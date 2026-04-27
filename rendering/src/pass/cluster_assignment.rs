use bytemuck::{Pod, Zeroable};
use engine_core::world::World;

use crate::{
    camera::Camera,
    graph::{NodeId, RenderGraph},
    light::{MAX_POINT_LIGHTS, PointLight},
    pass::light_upload::LightBuffers,
    pass::{Pass, PassContext, PassDesc},
    resource::{ResourceDescriptor, ResourceId},
    shader::ShaderAsset,
};

pub const CLUSTER_X: u32 = 16;
pub const CLUSTER_Y: u32 = 9;
pub const CLUSTER_Z: u32 = 24;
pub const TOTAL_CLUSTERS: usize = (CLUSTER_X * CLUSTER_Y * CLUSTER_Z) as usize;
pub const MAX_LIGHTS_PER_CLUSTER: usize = 128;

const LIGHT_GRID_BYTES: u64 = (TOTAL_CLUSTERS * std::mem::size_of::<u32>()) as u64;
const LIGHT_INDICES_BYTES: u64 =
    (TOTAL_CLUSTERS * MAX_LIGHTS_PER_CLUSTER * std::mem::size_of::<u32>()) as u64;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ClusterParams {
    pub tile_w: f32,
    pub tile_h: f32,
    pub z_near: f32,
    pub z_far: f32,
    pub log_ratio_recip: f32,
    pub num_point_lights: u32,
    pub inv_proj_00: f32,
    pub inv_proj_11: f32,
    pub screen_w: f32,
    pub screen_h: f32,
    pub debug_mode: u32,
    pub _pad: u32,
}

pub struct ClusterOutputs {
    pub cluster_params_id: ResourceId,
    pub light_grid_id: ResourceId,
    pub light_indices_id: ResourceId,
}

pub struct ClusterAssignmentPass {
    node_id: Option<NodeId>,
    point_buffer_id: ResourceId,
    cluster_params_id: ResourceId,
    light_grid_id: ResourceId,
    light_indices_id: ResourceId,
    shader: ShaderAsset,
    pipeline: Option<wgpu::ComputePipeline>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    // Cached once — all bound resources are buffers that never move.
    bind_group: Option<wgpu::BindGroup>,
}

impl ClusterAssignmentPass {
    pub fn new(
        graph: &mut RenderGraph,
        light_buffers: &LightBuffers,
        shader: ShaderAsset,
    ) -> (Self, ClusterOutputs) {
        let cluster_params_id = graph.alloc_resource_id(ResourceDescriptor::Buffer {
            size: std::mem::size_of::<ClusterParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let light_grid_id = graph.alloc_resource_id(ResourceDescriptor::Buffer {
            size: LIGHT_GRID_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let light_indices_id = graph.alloc_resource_id(ResourceDescriptor::Buffer {
            size: LIGHT_INDICES_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let outputs = ClusterOutputs {
            cluster_params_id,
            light_grid_id,
            light_indices_id,
        };

        (
            Self {
                node_id: None,
                point_buffer_id: light_buffers.point_buffer_id,
                cluster_params_id,
                light_grid_id,
                light_indices_id,
                shader,
                pipeline: None,
                bind_group_layout: None,
                bind_group: None,
            },
            outputs,
        )
    }
}

impl PassDesc for ClusterAssignmentPass {
    fn name(&self) -> &'static str {
        "cluster_assignment"
    }

    fn reads(&self) -> Vec<ResourceId> {
        vec![self.point_buffer_id]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![
            self.cluster_params_id,
            self.light_grid_id,
            self.light_indices_id,
        ]
    }
}

impl Pass for ClusterAssignmentPass {
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
        let cameras = world.get_entities::<Camera>();
        let Some(camera) = cameras.first() else {
            return;
        };

        let (sw, sh) = ctx.surface_size;
        let w = sw as f32;
        let h = sh as f32;
        let tan_half_fov = (camera.fovy / 2.0).tan();

        let num_lights = world
            .get_entities::<PointLight>()
            .len()
            .min(MAX_POINT_LIGHTS) as u32;

        let params = ClusterParams {
            tile_w: w / CLUSTER_X as f32,
            tile_h: h / CLUSTER_Y as f32,
            z_near: camera.near,
            z_far: camera.far,
            log_ratio_recip: CLUSTER_Z as f32 / (camera.far / camera.near).ln(),
            num_point_lights: num_lights,
            inv_proj_00: camera.aspect * tan_half_fov,
            inv_proj_11: tan_half_fov,
            screen_w: w,
            screen_h: h,
            debug_mode: 0,
            _pad: 0,
        };

        let Some(&params_buf) = ctx.buffers.get(&self.cluster_params_id) else {
            return;
        };
        let Some(&grid_buf) = ctx.buffers.get(&self.light_grid_id) else {
            return;
        };
        let Some(&indices_buf) = ctx.buffers.get(&self.light_indices_id) else {
            return;
        };
        let Some(&point_buf) = ctx.buffers.get(&self.point_buffer_id) else {
            return;
        };

        queue.write_buffer(params_buf, 0, bytemuck::bytes_of(&params));

        if self.pipeline.is_none() {
            let module = self
                .shader
                .compile(device)
                .expect("cluster_assignment shader compile failed");

            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cluster_assignment_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
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

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("cluster_assignment_pipeline_layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cluster_assignment_pipeline"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cluster_assignment_bg"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: point_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: grid_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: indices_buf.as_entire_binding(),
                    },
                ],
            });

            self.pipeline = Some(pipeline);
            self.bind_group_layout = Some(layout);
            self.bind_group = Some(bg);
        }

        let pipeline = self.pipeline.as_ref().unwrap();
        let bg = self.bind_group.as_ref().unwrap();

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cluster_assignment"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, bg, &[]);
        // 3456 clusters / 64 threads per workgroup = 54 workgroups exactly.
        const THREADS: u32 = 64;
        let workgroups = (CLUSTER_X * CLUSTER_Y * CLUSTER_Z).div_ceil(THREADS);
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }
}
