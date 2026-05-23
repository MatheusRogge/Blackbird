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

pub const SHADOW_MAP_SIZE: u32 = 2048;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ShadowParams {
    pub view_to_shadow: [[f32; 4]; 4],
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

    shadow_map_id: ResourceId,
    shadow_params_id: ResourceId,

    light_vp_buf: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,

    vertex_buffer_cache: HashMap<usize, wgpu::Buffer>,
    index_buffer_cache: HashMap<usize, wgpu::Buffer>,
}

impl ShadowPass {
    pub fn new(graph: &mut RenderGraph, shader: ShaderAsset) -> (Self, ShadowOutputs) {
        let shadow_map_id = graph.alloc_resource_id(ResourceDescriptor::FixedTexture {
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: 1,
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
                shadow_map_id,
                shadow_params_id,
                light_vp_buf: None,
                bind_group: None,
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

fn shadow_vp_stable(camera: &Camera, light_view: &Mat4, shadow_distance: f32) -> Mat4 {
    let near = camera.near;
    let far = shadow_distance.min(camera.far);
    let th = (camera.fovy * 0.5).tan();
    let nh = near * th;
    let nw = nh * camera.aspect;
    let fh = far * th;
    let fw = fh * camera.aspect;

    let corners_vs = [
        Vec3::new(-nw, -nh, -near), Vec3::new( nw, -nh, -near),
        Vec3::new(-nw,  nh, -near), Vec3::new( nw,  nh, -near),
        Vec3::new(-fw, -fh, -far),  Vec3::new( fw, -fh, -far),
        Vec3::new(-fw,  fh, -far),  Vec3::new( fw,  fh, -far),
    ];

    let inv_cam_view = camera.view_matrix().inversed();

    let mut min_x = f32::MAX; let mut max_x = f32::MIN;
    let mut min_y = f32::MAX; let mut max_y = f32::MIN;
    let mut min_z = f32::MAX; let mut max_z = f32::MIN;

    for c in &corners_vs {
        let world = inv_cam_view * Vec4::new(c.x, c.y, c.z, 1.0);
        let ls    = *light_view  * Vec4::new(world.x, world.y, world.z, 1.0);
        min_x = min_x.min(ls.x); max_x = max_x.max(ls.x);
        min_y = min_y.min(ls.y); max_y = max_y.max(ls.y);
        min_z = min_z.min(ls.z); max_z = max_z.max(ls.z);
    }

    let ortho_near = (-max_z - shadow_distance * 0.5).max(0.1);
    let ortho_far  = -min_z + 10.0;

    let l = min_x;
    let r = max_x;
    let b = min_y;
    let t = max_y;
    let mut vp = ortho_rh_wgpu(l, r, b, t, ortho_near, ortho_far) * *light_view;

    // The AABB bounds (l,r,b,t) are constant for pure camera translation so
    // snapping them does nothing. What moves is the combined VP translation
    // column. Snap it to the nearest shadow-texel in NDC space so the shadow
    // grid only moves in whole-texel steps for every direction of movement.
    let half = SHADOW_MAP_SIZE as f32 * 0.5; // texels per NDC unit
    vp.cols[3].x = (vp.cols[3].x * half).round() / half;
    vp.cols[3].y = (vp.cols[3].y * half).round() / half;

    vp
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
        let Some(&shadow_view) = ctx.views.get(&self.shadow_map_id) else {
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
        // Place the light far behind the camera so all scene geometry (including
        // overhead casters) sits in front of the light's near plane.
        let light_push = camera.far;
        let light_view = Mat4::look_at(camera.eye - dir * light_push, camera.eye, up);
        let shadow_distance = camera.far * 0.5;
        let light_vp = shadow_vp_stable(camera, &light_view, shadow_distance);

        let view_to_shadow = light_vp * camera.view_matrix().inversed();
        let cols = view_to_shadow.cols;
        let params = ShadowParams {
            view_to_shadow: [cols[0].into(), cols[1].into(), cols[2].into(), cols[3].into()],
            bias: 0.002,
            inv_shadow_map_size: 1.0 / SHADOW_MAP_SIZE as f32,
            _pad: [0.0; 2],
        };
        queue.write_buffer(params_buf, 0, bytemuck::bytes_of(&params));

        // Lazy pipeline + light VP buffer init
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

            let light_vp_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("shadow_light_vp_buf"),
                size: std::mem::size_of::<[[f32; 4]; 4]>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shadow_bg"),
                layout: &bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: light_vp_buf.as_entire_binding(),
                }],
            });

            self.pipeline = Some(pipeline);
            self.light_vp_buf = Some(light_vp_buf);
            self.bind_group = Some(bg);
        }

        let light_vp_cols = light_vp.cols;
        let light_vp_raw: [[f32; 4]; 4] = [
            light_vp_cols[0].into(),
            light_vp_cols[1].into(),
            light_vp_cols[2].into(),
            light_vp_cols[3].into(),
        ];
        if let Some(buf) = &self.light_vp_buf {
            queue.write_buffer(buf, 0, bytemuck::bytes_of(&light_vp_raw));
        }

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
        let bg = self.bind_group.as_ref().unwrap();

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shadow"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: shadow_view,
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
                let mesh = &meshes[mesh_idx];
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.get_indices_count() as u32, 0, 0..1);
            }
        }
    }
}
