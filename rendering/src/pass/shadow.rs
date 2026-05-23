use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use engine_core::world::World;
use ultraviolet::{Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

use crate::{
    camera::Camera,
    graph::{NodeId, RenderGraph},
    light::SkyLight,
    mesh::Mesh,
    pass::{Pass, PassContext, PassDesc},
    resource::{ResourceDescriptor, ResourceId},
    shader::ShaderAsset,
};

pub const NUM_CASCADES: usize = 4;
pub const SHADOW_MAP_SIZE: u32 = 2048;

const CASCADE_LAMBDA: f32 = 0.75;

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
    bgl: Option<wgpu::BindGroupLayout>,

    shadow_map_id: ResourceId,
    shadow_params_id: ResourceId,

    cascade_views: Vec<wgpu::TextureView>,
    cascade_vp_bufs: Vec<wgpu::Buffer>,
    cascade_bind_groups: Vec<wgpu::BindGroup>,

    vertex_buffer_cache: HashMap<usize, wgpu::Buffer>,
    index_buffer_cache: HashMap<usize, wgpu::Buffer>,
}

impl ShadowPass {
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

        let shadow_params_id = graph.alloc_resource_id(ResourceDescriptor::Buffer {
            size: std::mem::size_of::<ShadowParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let outputs = ShadowOutputs { shadow_map_id, shadow_params_id };

        (
            Self {
                node_id: None,
                shader,
                pipeline: None,
                bgl: None,
                shadow_map_id,
                shadow_params_id,
                cascade_views: Vec::new(),
                cascade_vp_bufs: Vec::new(),
                cascade_bind_groups: Vec::new(),
                vertex_buffer_cache: HashMap::new(),
                index_buffer_cache: HashMap::new(),
            },
            outputs,
        )
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
/// texel steps during camera translation.
fn cascade_vp(camera: &Camera, light_view: &Mat4, near_z: f32, far_z: f32) -> Mat4 {
    let th = (camera.fovy * 0.5).tan();
    let nh = near_z * th;
    let nw = nh * camera.aspect;
    let fh = far_z * th;
    let fw = fh * camera.aspect;

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

    // Z range from actual corners; extend backwards to capture casters outside the slice.
    let mut min_z = f32::MAX;
    let mut max_z = f32::MIN;
    for c in &corners_ws {
        let ls = *light_view * Vec4::new(c.x, c.y, c.z, 1.0);
        min_z = min_z.min(ls.z);
        max_z = max_z.max(ls.z);
    }
    let ortho_near = (-max_z - far_z * 0.5).max(0.1);
    let ortho_far  = -min_z + 10.0;

    ortho_rh_wgpu(l, r, b, t, ortho_near, ortho_far) * *light_view
}

/// PSSM split scheme blending logarithmic and uniform distributions.
fn compute_cascade_splits(near: f32, far: f32) -> [f32; NUM_CASCADES] {
    let mut splits = [0f32; NUM_CASCADES];
    for i in 0..NUM_CASCADES {
        let t = (i + 1) as f32 / NUM_CASCADES as f32;
        let log = near * (far / near).powf(t);
        let uni = near + (far - near) * t;
        splits[i] = CASCADE_LAMBDA * log + (1.0 - CASCADE_LAMBDA) * uni;
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
        vec![self.shadow_map_id, self.shadow_params_id]
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
        let light_push = camera.far;
        let light_view = Mat4::look_at(camera.eye - dir * light_push, camera.eye, up);

        let splits = compute_cascade_splits(camera.near, camera.far);

        // Lazy init: pipeline + per-cascade buffers + per-layer views.
        if self.pipeline.is_none() {
            let module = self.shader.compile(device).expect("shadow shader compile failed");

            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow_pipeline_layout"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("shadow_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    buffers: &[crate::mesh::Vertex::buffer_descriptor()],
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

            for i in 0..NUM_CASCADES {
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("shadow_vp_cascade{i}")),
                    size: std::mem::size_of::<[[f32; 4]; 4]>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("shadow_bg_cascade{i}")),
                    layout: &bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                });
                self.cascade_vp_bufs.push(buf);
                self.cascade_bind_groups.push(bg);
            }

            self.bgl = Some(bgl);
            self.pipeline = Some(pipeline);
        }

        // Create per-layer render attachment views once, from the graph-allocated texture.
        if self.cascade_views.is_empty() {
            if let Some(&tex) = ctx.textures.get(&self.shadow_map_id) {
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
        }

        if self.cascade_views.is_empty() {
            return;
        }

        // Upload mesh buffers once.
        let meshes = world.get_entities::<Mesh>();
        for (mesh_idx, mesh) in meshes.iter().enumerate() {
            self.vertex_buffer_cache.entry(mesh_idx).or_insert_with(|| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    usage: wgpu::BufferUsages::VERTEX,
                    contents: mesh.get_vertex_buffer_content(),
                })
            });
            self.index_buffer_cache.entry(mesh_idx).or_insert_with(|| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    usage: wgpu::BufferUsages::INDEX,
                    contents: mesh.get_indices_buffer_content(),
                })
            });
        }

        let pipeline = self.pipeline.as_ref().unwrap();
        let inv_cam_view = camera.view_matrix().inversed();

        let mut view_to_shadow_raw = [[[0f32; 4]; 4]; NUM_CASCADES];

        for i in 0..NUM_CASCADES {
            let near_z = if i == 0 { camera.near } else { splits[i - 1] };
            let far_z  = splits[i];
            let light_vp = cascade_vp(camera, &light_view, near_z, far_z);

            let v2s = light_vp * inv_cam_view;
            let cols = v2s.cols;
            view_to_shadow_raw[i] = [cols[0].into(), cols[1].into(), cols[2].into(), cols[3].into()];

            let lp_cols = light_vp.cols;
            let vp_raw: [[f32; 4]; 4] = [
                lp_cols[0].into(), lp_cols[1].into(), lp_cols[2].into(), lp_cols[3].into(),
            ];
            queue.write_buffer(&self.cascade_vp_bufs[i], 0, bytemuck::bytes_of(&vp_raw));

            let layer_view = &self.cascade_views[i];
            let bg = &self.cascade_bind_groups[i];

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
            pass.set_bind_group(0, bg, &[]);

            for mesh_idx in 0..meshes.len() {
                if let (Some(vb), Some(ib)) = (
                    self.vertex_buffer_cache.get(&mesh_idx),
                    self.index_buffer_cache.get(&mesh_idx),
                ) {
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..meshes[mesh_idx].get_indices_count() as u32, 0, 0..1);
                }
            }
        }

        let params = ShadowParams {
            view_to_shadow: view_to_shadow_raw,
            cascade_splits: splits,
            bias: 0.002,
            inv_shadow_map_size: 1.0 / SHADOW_MAP_SIZE as f32,
            _pad: [0.0; 2],
        };
        queue.write_buffer(params_buf, 0, bytemuck::bytes_of(&params));
    }
}
