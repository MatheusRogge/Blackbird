
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use engine_core::{
    entity::EntityKey,
    world::{EntityHandle, World},
};
use ultraviolet::{Mat4, Vec3, Vec4};

use crate::{
    arena::GeometryArena,
    camera::Camera,
    graph::{NodeId, RenderGraph},
    light::SkyLight,
    mesh::Mesh,
    pass::{Pass, PassContext, PassDesc, gbuffer::GBufferPass},
    resource::{ResourceDescriptor, ResourceId},
    shader::ShaderAsset,
    texture::TextureAsset,
};

pub const NUM_CASCADES: usize = 4;
pub const SHADOW_MAP_SIZE: u32 = 2048;

const CASCADE_LAMBDA: f32 = 0.75;
const CASCADE_VP_STRIDE: u64 = 256;
const VP_SIZE: u64 = std::mem::size_of::<[[f32; 4]; 4]>() as u64;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ShadowParams {
    pub view_to_shadow: [[[f32; 4]; 4]; NUM_CASCADES],
    pub cascade_splits: [f32; NUM_CASCADES],
    pub bias: f32,
    pub inv_shadow_map_size: f32,
    pub _pad: [f32; 2],
}

pub struct ShadowOutputs {
    pub shadow_map_id: ResourceId,
    pub shadow_params_id: ResourceId,
}

pub struct ShadowPass {
    node_id: Option<NodeId>,
    shader: ShaderAsset,
    pipeline: Option<wgpu::RenderPipeline>,
    /// Unculled pipeline that discards cut-out texels, for `Mesh::is_masked` meshes.
    masked_pipeline: Option<wgpu::RenderPipeline>,
    masked_texture_layout: Option<wgpu::BindGroupLayout>,
    masked_sampler: Option<wgpu::Sampler>,
    /// Albedo textures of masked meshes, keyed by `TextureAsset` pointer; key 0 is opaque white.
    masked_textures: HashMap<usize, (wgpu::Texture, wgpu::BindGroup)>,
    bgl: Option<wgpu::BindGroupLayout>,
    instance_layout: Option<wgpu::BindGroupLayout>,
    instance_bind_group: Option<(u64, wgpu::BindGroup)>,

    shadow_map_id: ResourceId,
    shadow_params_id: ResourceId,
    cascade_vp_id: ResourceId,

    cascade_views: Vec<wgpu::TextureView>,
    cascade_bind_group: Option<wgpu::BindGroup>,

    local_bounds: HashMap<EntityKey, Aabb>,
}

#[derive(Clone, Copy)]
struct Aabb {
    min: Vec3,
    max: Vec3,
}

impl Aabb {
    fn empty() -> Self {
        Self { min: Vec3::broadcast(f32::MAX), max: Vec3::broadcast(f32::MIN) }
    }

    fn is_empty(&self) -> bool {
        self.min.x > self.max.x
    }

    fn extend(&mut self, p: Vec3) {
        self.min = self.min.min_by_component(p);
        self.max = self.max.max_by_component(p);
    }

    fn corners(&self) -> [Vec3; 8] {
        let (a, b) = (self.min, self.max);
        [
            Vec3::new(a.x, a.y, a.z), Vec3::new(b.x, a.y, a.z),
            Vec3::new(a.x, b.y, a.z), Vec3::new(b.x, b.y, a.z),
            Vec3::new(a.x, a.y, b.z), Vec3::new(b.x, a.y, b.z),
            Vec3::new(a.x, b.y, b.z), Vec3::new(b.x, b.y, b.z),
        ]
    }
}

/// Key of a mesh's albedo texture in `ShadowPass::masked_textures`.
fn albedo_key(mesh: &Mesh) -> usize {
    mesh.albedo_texture.as_ref().map_or(0, |t| Arc::as_ptr(t) as usize)
}

impl ShadowPass {
    /// Uploads the albedo texture of every masked mesh not seen before (plus the white
    /// fallback), so the cascades can alpha-test against it.
    fn upload_masked_textures(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, meshes: &[EntityHandle<'_, Mesh>]) {
        let (Some(layout), Some(sampler)) = (&self.masked_texture_layout, &self.masked_sampler) else {
            return;
        };
        let white = Arc::new(TextureAsset::new(1, 1, vec![255; 4]));
        let fallback = (!self.masked_textures.contains_key(&0)).then_some((0, &white));
        let needed = meshes
            .iter()
            .filter(|h| h.entity.is_masked())
            .filter_map(|h| h.entity.albedo_texture.as_ref().map(|t| (albedo_key(h.entity), t)))
            .chain(fallback);

        for (key, asset) in needed {
            if self.masked_textures.contains_key(&key) {
                continue;
            }
            let (texture, view) = GBufferPass::upload_texture(device, queue, asset, wgpu::TextureFormat::Rgba8Unorm);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shadow_masked_texture"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
                ],
            });
            self.masked_textures.insert(key, (texture, bind_group));
        }
    }

    pub fn new(graph: &mut RenderGraph, shader: ShaderAsset) -> (Self, ShadowOutputs) {
        let shadow_map_id = graph.alloc_resource_id(ResourceDescriptor::FixedTexture {
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: NUM_CASCADES as u32,
            },
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        let shadow_params_id = graph.alloc_resource_id(ResourceDescriptor::UploadBuffer {
            size: std::mem::size_of::<ShadowParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let cascade_vp_id = graph.alloc_resource_id(ResourceDescriptor::UploadBuffer {
            size: CASCADE_VP_STRIDE * NUM_CASCADES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let outputs = ShadowOutputs { shadow_map_id, shadow_params_id };

        (
            Self {
                node_id: None,
                shader,
                pipeline: None,
                masked_pipeline: None,
                masked_texture_layout: None,
                masked_sampler: None,
                masked_textures: HashMap::new(),
                bgl: None,
                instance_layout: None,
                instance_bind_group: None,
                shadow_map_id,
                shadow_params_id,
                cascade_vp_id,
                cascade_views: Vec::new(),
                cascade_bind_group: None,
                local_bounds: HashMap::new(),
            },
            outputs,
        )
    }

    /// World-space bounds of every mesh. Local bounds are computed once per
    /// entity, so only the transforms are re-applied each frame.
    fn scene_bounds(&mut self, meshes: &[EntityHandle<'_, Mesh>]) -> Aabb {
        let live: HashSet<EntityKey> = meshes.iter().map(|h| h.id).collect();
        self.local_bounds.retain(|key, _| live.contains(key));

        let mut scene = Aabb::empty();
        for handle in meshes {
            let local = self.local_bounds.entry(handle.id).or_insert_with(|| {
                let mut bounds = Aabb::empty();
                for v in &handle.entity.vertices {
                    bounds.extend(Vec3::from(v.position));
                }
                bounds
            });
            if local.is_empty() {
                continue;
            }
            for corner in local.corners() {
                scene.extend(handle.entity.transform.transform_point3(corner));
            }
        }
        scene
    }
}

fn ortho_rh_wgpu(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> Mat4 {
    let rml = r - l;
    let tmb = t - b;
    let fmn = f - n;
    Mat4::new(
        Vec4::new(2.0 / rml, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 2.0 / tmb, 0.0, 0.0),
        Vec4::new(0.0, 0.0, -1.0 / fmn, 0.0),
        Vec4::new(-(r + l) / rml, -(t + b) / tmb, -n / fmn, 1.0),
    )
}

/// Builds a stable light-space VP matrix for one cascade slice.
///
/// Uses a bounding sphere (not an AABB) to fix the XY extents so the ortho
/// projection size is rotation-invariant — the camera rotating no longer changes
/// the projection bounds, eliminating the warping artefact. The sphere centre is
/// then snapped to the nearest shadow texel so the grid only shifts by whole-
/// texel steps during camera translation. `depth_range` is the light-space
/// near/far of the whole scene, so every caster lands in every cascade.
fn cascade_vp(camera: &Camera, aspect: f32, light_view: &Mat4, near_z: f32, far_z: f32, depth_range: (f32, f32)) -> Mat4 {
    let th = (camera.fovy * 0.5).tan();
    let nh = near_z * th;
    let nw = nh * aspect;
    let fh = far_z * th;
    let fw = fh * aspect;

    let corners_vs = [
        Vec3::new(-nw, -nh, -near_z), Vec3::new( nw, -nh, -near_z),
        Vec3::new(-nw,  nh, -near_z), Vec3::new( nw,  nh, -near_z),
        Vec3::new(-fw, -fh, -far_z),  Vec3::new( fw, -fh, -far_z),
        Vec3::new(-fw,  fh, -far_z),  Vec3::new( fw,  fh, -far_z),
    ];

    let inv_cam_view = camera.view_matrix().inversed();

    // Convert corners to world space.
    let mut corners_ws = [Vec3::zero(); 8];
    let mut center_ws  = Vec3::zero();
    for (i, c) in corners_vs.iter().enumerate() {
        let w4 = inv_cam_view * Vec4::new(c.x, c.y, c.z, 1.0);
        corners_ws[i] = Vec3::new(w4.x, w4.y, w4.z);
        center_ws += corners_ws[i];
    }
    center_ws /= 8.0;

    // Bounding sphere radius — constant for a given (near_z, far_z, fov, aspect)
    // regardless of camera orientation, so the XY projection bounds never change
    // size as the camera rotates.
    let radius = corners_ws
        .iter()
        .map(|c| (*c - center_ws).mag())
        .fold(0f32, f32::max);

    // Project sphere centre into light space and snap to whole-texel steps.
    let c_ls = *light_view * Vec4::new(center_ws.x, center_ws.y, center_ws.z, 1.0);
    let texel = 2.0 * radius / SHADOW_MAP_SIZE as f32;
    let sx = (c_ls.x / texel).round() * texel;
    let sy = (c_ls.y / texel).round() * texel;

    // Fixed square XY ortho bounds around the snapped centre.
    let l = sx - radius;
    let r = sx + radius;
    let b = sy - radius;
    let t = sy + radius;

    ortho_rh_wgpu(l, r, b, t, depth_range.0, depth_range.1) * *light_view
}

/// PSSM split scheme blending logarithmic and uniform distributions.
fn compute_cascade_splits(near: f32, far: f32) -> [f32; NUM_CASCADES] {
    let mut splits = [0f32; NUM_CASCADES];
    for (i, split) in splits.iter_mut().enumerate() {
        let t = (i + 1) as f32 / NUM_CASCADES as f32;
        let log = near * (far / near).powf(t);
        let uni = near + (far - near) * t;
        *split = CASCADE_LAMBDA * log + (1.0 - CASCADE_LAMBDA) * uni;
    }
    splits
}

impl PassDesc for ShadowPass {
    fn name(&self) -> &'static str {
        "shadow"
    }

    fn reads(&self) -> Vec<ResourceId> {
        vec![]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![self.shadow_map_id, self.shadow_params_id, self.cascade_vp_id]
    }
}

impl Pass for ShadowPass {
    fn bind_node_id(&mut self, node_id: NodeId) {
        self.node_id = Some(node_id);
    }

    fn execute(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &PassContext<'_>,
        world: &World,
    ) {
        let Some(&params_buf) = ctx.buffers.get(&self.shadow_params_id) else {
            return;
        };
        let Some(&cascade_vp_buf) = ctx.buffers.get(&self.cascade_vp_id) else {
            return;
        };

        let cameras = world.get_entities::<Camera>();
        let Some(camera) = cameras.first() else {
            return;
        };

        let sky_lights = world.get_entities::<SkyLight>();
        let dir = sky_lights
            .first()
            .map(|l| l.direction.normalized())
            .unwrap_or(Vec3::new(0.0, -1.0, 0.0));

        let up = if dir.dot(Vec3::unit_y()).abs() > 0.99 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::unit_y()
        };
        let light_view = Mat4::look_at(camera.eye - dir, camera.eye, up);

        let meshes = world.get_entity_handles::<Mesh>();
        let scene = self.scene_bounds(&meshes);
        if scene.is_empty() {
            return;
        }

        // Cascades only need to reach the farthest scene point, and their depth
        // range only needs to span the scene, so precision and bias follow the
        // scene's scale rather than the camera's far plane.
        let view = camera.view_matrix();
        let (mut min_z, mut max_z) = (f32::MAX, f32::MIN);
        let mut shadow_far = 0f32;
        for corner in scene.corners() {
            shadow_far = shadow_far.max(-view.transform_point3(corner).z);
            let z = light_view.transform_point3(corner).z;
            min_z = min_z.min(z);
            max_z = max_z.max(z);
        }
        let shadow_far = shadow_far.clamp(camera.near * 2.0, camera.far);
        let pad = (max_z - min_z) * 0.01 + 1e-3;
        let depth_range = (-max_z - pad, -min_z + pad);

        let splits = compute_cascade_splits(camera.near, shadow_far);

        // Lazy init: pipeline + cascade bind group + per-layer views.
        if self.pipeline.is_none() {
            let module = self.shader.compile(device).expect("shadow shader compile failed");

            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(VP_SIZE),
                    },
                    count: None,
                }],
            });

            let instance_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow_instance_layout"),
                entries: &[GeometryArena::instance_layout_entry(0)],
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow_pipeline_layout"),
                bind_group_layouts: &[Some(&bgl), Some(&instance_layout)],
                immediate_size: 0,
            });

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("shadow_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    buffers: &[Some(crate::mesh::Vertex::buffer_descriptor())],
                    compilation_options: Default::default(),
                },
                fragment: None,
                primitive: wgpu::PrimitiveState {
                    cull_mode: Some(wgpu::Face::Front),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

            let masked_texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow_masked_texture_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

            let masked_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow_masked_pipeline_layout"),
                bind_group_layouts: &[Some(&bgl), Some(&instance_layout), Some(&masked_texture_layout)],
                immediate_size: 0,
            });

            let masked_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("shadow_masked_pipeline"),
                layout: Some(&masked_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_masked"),
                    buffers: &[Some(crate::mesh::Vertex::buffer_descriptor())],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_masked"),
                    targets: &[],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

            self.masked_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("shadow_masked_sampler"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }));
            self.masked_texture_layout = Some(masked_texture_layout);
            self.masked_pipeline = Some(masked_pipeline);

            self.cascade_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shadow_cascade_bg"),
                layout: &bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        size: wgpu::BufferSize::new(VP_SIZE),
                        ..cascade_vp_buf.buffer_binding()
                    }),
                }],
            }));

            self.bgl = Some(bgl);
            self.instance_layout = Some(instance_layout);
            self.pipeline = Some(pipeline);
        }

        // Create per-layer render attachment views once, from the graph-allocated texture.
        if self.cascade_views.is_empty()
            && let Some(&tex) = ctx.textures.get(&self.shadow_map_id)
        {
            for i in 0..NUM_CASCADES {
                let view = tex.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i as u32,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
                self.cascade_views.push(view);
            }
        }

        if self.cascade_views.is_empty() {
            return;
        }

        let generation = ctx.geometry.generation();
        if self.instance_bind_group.as_ref().is_none_or(|(g, _)| *g != generation)
            && let Some(layout) = self.instance_layout.as_ref()
        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shadow_instances"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ctx.geometry.instance_buffer().as_entire_binding(),
                }],
            });
            self.instance_bind_group = Some((generation, bind_group));
        }
        self.upload_masked_textures(device, queue, &meshes);

        let Some((_, instance_bg)) = self.instance_bind_group.as_ref() else {
            return;
        };

        let (Some(pipeline), Some(masked_pipeline)) = (&self.pipeline, &self.masked_pipeline) else {
            return;
        };
        let mut masked: Vec<_> = meshes.iter().filter(|h| h.entity.is_masked()).collect();
        masked.sort_unstable_by_key(|h| albedo_key(h.entity));
        let inv_cam_view = camera.view_matrix().inversed();
        let (w, h) = ctx.surface_size;
        let aspect = camera.aspect_or(w as f32 / h as f32);

        let mut view_to_shadow_raw = [[[0f32; 4]; 4]; NUM_CASCADES];

        for i in 0..NUM_CASCADES {
            let near_z = if i == 0 { camera.near } else { splits[i - 1] };
            let far_z  = splits[i];
            let light_vp = cascade_vp(camera, aspect, &light_view, near_z, far_z, depth_range);

            let v2s = light_vp * inv_cam_view;
            let cols = v2s.cols;
            view_to_shadow_raw[i] = [cols[0].into(), cols[1].into(), cols[2].into(), cols[3].into()];

            let lp_cols = light_vp.cols;
            let vp_raw: [[f32; 4]; 4] = [
                lp_cols[0].into(), lp_cols[1].into(), lp_cols[2].into(), lp_cols[3].into(),
            ];
            let vp_offset = CASCADE_VP_STRIDE * i as u64;
            cascade_vp_buf.write_at(queue, vp_offset, bytemuck::bytes_of(&vp_raw));

            let layer_view = &self.cascade_views[i];
            let bg = self.cascade_bind_group.as_ref().expect("shadow cascade bind group");

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&format!("shadow_cascade{i}")),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: layer_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bg, &[vp_offset as u32]);
            pass.set_bind_group(1, instance_bg, &[]);
            pass.set_vertex_buffer(0, ctx.geometry.vertex_buffer().slice(..));
            pass.set_index_buffer(ctx.geometry.index_buffer().slice(..), wgpu::IndexFormat::Uint32);

            let opaque = meshes.iter().filter(|h| !h.entity.is_masked());
            for draw in opaque.filter_map(|h| ctx.geometry.get(h.id)) {
                pass.draw_indexed(draw.indices.clone(), draw.base_vertex, draw.instances());
            }

            if masked.is_empty() {
                continue;
            }
            pass.set_pipeline(masked_pipeline);
            pass.set_bind_group(0, bg, &[vp_offset as u32]);
            pass.set_bind_group(1, instance_bg, &[]);
            let mut current_texture = None;
            for handle in &masked {
                let Some(draw) = ctx.geometry.get(handle.id) else { continue };
                let key = albedo_key(handle.entity);
                if current_texture != Some(key) {
                    let Some((_, texture_bg)) = self.masked_textures.get(&key).or(self.masked_textures.get(&0)) else {
                        continue;
                    };
                    pass.set_bind_group(2, texture_bg, &[]);
                    current_texture = Some(key);
                }
                pass.draw_indexed(draw.indices.clone(), draw.base_vertex, draw.instances());
            }
        }

        let params = ShadowParams {
            view_to_shadow: view_to_shadow_raw,
            cascade_splits: splits,
            bias: 0.002,
            inv_shadow_map_size: 1.0 / SHADOW_MAP_SIZE as f32,
            _pad: [0.0; 2],
        };
        params_buf.write(queue, bytemuck::bytes_of(&params));
    }
}
