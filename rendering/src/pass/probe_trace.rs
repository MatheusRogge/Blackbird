use bytemuck::{Pod, Zeroable};
use engine_core::world::World;
use ultraviolet::Mat4;

use crate::{
    camera::Camera,
    graph::{NodeId, RenderGraph},
    pass::{
        Pass, PassContext, PassDesc,
        bvh_upload::BvhOutputs,
        probe_atlas::{ProbeAtlasOutputs, PROBE_COUNT, RAYS_PER_PROBE},
        light_upload::LightBuffers,
    },
    resource::{ResourceDescriptor, ResourceId},
    shader::ShaderAsset,
};

const RAY_BUFFER_SIZE: u64 = (PROBE_COUNT * RAYS_PER_PROBE * 16) as u64;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ProbeTraceFrame {
    frame_rotation: [[f32; 4]; 3], // mat3x3 col-major, each col padded → 48 bytes
    inv_view: [[f32; 4]; 4],       // mat4x4 → 64 bytes
    frame_index: u32,              // offset 112
    _pad: [u32; 3],                // offset 116 → total 128 bytes
}

pub struct ProbeTraceOutputs {
    pub ray_radiance_id: ResourceId,
    pub ray_direction_id: ResourceId,
}

pub struct ProbeTracePass {
    node_id: Option<NodeId>,
    bvh_nodes_id: ResourceId,
    bvh_tris_id: ResourceId,
    bvh_info_id: ResourceId,
    probe_params_id: ResourceId,
    sky_buffer_id: ResourceId,
    counts_buffer_id: ResourceId,
    ray_radiance_id: ResourceId,
    ray_direction_id: ResourceId,
    pipeline: Option<wgpu::ComputePipeline>,
    bind_group: Option<wgpu::BindGroup>,
    frame_uniform_buf: Option<wgpu::Buffer>,
    frame_index: u32,
    shader: ShaderAsset,
}

impl ProbeTracePass {
    pub fn new(
        graph: &mut RenderGraph,
        bvh: &BvhOutputs,
        probe_atlas: &ProbeAtlasOutputs,
        light_buffers: &LightBuffers,
        shader: ShaderAsset,
    ) -> (Self, ProbeTraceOutputs) {
        let ray_radiance_id = graph.alloc_resource_id(ResourceDescriptor::Buffer {
            size: RAY_BUFFER_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let ray_direction_id = graph.alloc_resource_id(ResourceDescriptor::Buffer {
            size: RAY_BUFFER_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let outputs = ProbeTraceOutputs { ray_radiance_id, ray_direction_id };

        (
            Self {
                node_id: None,
                bvh_nodes_id: bvh.nodes_id,
                bvh_tris_id: bvh.tris_id,
                bvh_info_id: bvh.info_id,
                probe_params_id: probe_atlas.probe_params_id,
                sky_buffer_id: light_buffers.sky_buffer_id,
                counts_buffer_id: light_buffers.counts_buffer_id,
                ray_radiance_id,
                ray_direction_id,
                pipeline: None,
                bind_group: None,
                frame_uniform_buf: None,
                frame_index: 0,
                shader,
            },
            outputs,
        )
    }
}

fn make_frame_rotation(frame_index: u32) -> [[f32; 4]; 3] {
    let angle = (frame_index as f32) * 2.399_963;
    let t = frame_index as f32;
    let x = (t * 0.7132).sin();
    let y = (t * 0.3541).cos();
    let z = (t * 0.5234).sin();
    let len = (x * x + y * y + z * z).sqrt().max(1e-6);
    let (x, y, z) = (x / len, y / len, z / len);
    let c = angle.cos();
    let s = angle.sin();
    let t1 = 1.0 - c;
    [
        [t1 * x * x + c,     t1 * x * y + s * z, t1 * x * z - s * y, 0.0],
        [t1 * x * y - s * z, t1 * y * y + c,     t1 * y * z + s * x, 0.0],
        [t1 * x * z + s * y, t1 * y * z - s * x, t1 * z * z + c,     0.0],
    ]
}

fn mat4_to_cols(m: Mat4) -> [[f32; 4]; 4] {
    let c = m.cols;
    [c[0].into(), c[1].into(), c[2].into(), c[3].into()]
}

impl PassDesc for ProbeTracePass {
    fn name(&self) -> &'static str {
        "probe_trace"
    }

    fn reads(&self) -> Vec<ResourceId> {
        vec![
            self.bvh_nodes_id,
            self.bvh_tris_id,
            self.bvh_info_id,
            self.probe_params_id,
            self.sky_buffer_id,
            self.counts_buffer_id,
        ]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![self.ray_radiance_id, self.ray_direction_id]
    }
}

impl Pass for ProbeTracePass {
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
        let current = self.frame_index;
        self.frame_index = self.frame_index.wrapping_add(1);
        if current % 4 != 0 {
            return;
        }

        let Some(&bvh_nodes_buf) = ctx.buffers.get(&self.bvh_nodes_id) else { return };
        let Some(&bvh_tris_buf) = ctx.buffers.get(&self.bvh_tris_id) else { return };
        let Some(&bvh_info_buf) = ctx.buffers.get(&self.bvh_info_id) else { return };
        let Some(&probe_params_buf) = ctx.buffers.get(&self.probe_params_id) else { return };
        let Some(&sky_buf) = ctx.buffers.get(&self.sky_buffer_id) else { return };
        let Some(&counts_buf) = ctx.buffers.get(&self.counts_buffer_id) else { return };
        let Some(&radiance_buf) = ctx.buffers.get(&self.ray_radiance_id) else { return };
        let Some(&direction_buf) = ctx.buffers.get(&self.ray_direction_id) else { return };

        let cameras = world.get_entities::<Camera>();
        let inv_view = cameras
            .first()
            .map(|c| c.view_matrix().inversed())
            .unwrap_or(Mat4::identity());

        if self.frame_uniform_buf.is_none() {
            self.frame_uniform_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("probe_trace_frame"),
                size: std::mem::size_of::<ProbeTraceFrame>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        let frame_data = ProbeTraceFrame {
            frame_rotation: make_frame_rotation(self.frame_index),
            inv_view: mat4_to_cols(inv_view),
            frame_index: self.frame_index,
            _pad: [0; 3],
        };
        queue.write_buffer(
            self.frame_uniform_buf.as_ref().unwrap(),
            0,
            bytemuck::bytes_of(&frame_data),
        );

        if self.pipeline.is_none() {
            let frame_buf = self.frame_uniform_buf.as_ref().unwrap();

            let module = self
                .shader
                .compile(device)
                .expect("probe_trace shader compile failed");

            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("probe_trace_bgl"),
                entries: &[
                    // 0: BVH nodes
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 1: BVH triangles
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 2: BVH info
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 3: probe params
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 4: sky lights (storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 5: light counts
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 6: frame uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 7: ray_radiance output
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 8: ray_direction output
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            let pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("probe_trace_pipeline_layout"),
                    bind_group_layouts: &[Some(&bgl)],
                    immediate_size: 0,
                });

            self.pipeline = Some(device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("probe_trace_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                },
            ));

            self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("probe_trace_bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: bvh_nodes_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: bvh_tris_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: bvh_info_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: probe_params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: sky_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: counts_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: frame_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: radiance_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: direction_buf.as_entire_binding() },
                ],
            }));
        }

        let pipeline = self.pipeline.as_ref().unwrap();
        let bg = self.bind_group.as_ref().unwrap();

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("probe_trace"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, bg, &[]);
        compute_pass.dispatch_workgroups(RAYS_PER_PROBE / 32, PROBE_COUNT, 1);
    }
}
