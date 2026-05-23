use engine_core::world::World;

use crate::{
    graph::{NodeId, RenderGraph},
    pass::{
        Pass, PassContext, PassDesc,
        probe_atlas::{ProbeAtlasOutputs, PROBE_COUNT},
        probe_trace::ProbeTraceOutputs,
    },
    resource::ResourceId,
    shader::ShaderAsset,
};

pub struct ProbeUpdatePass {
    node_id: Option<NodeId>,
    ray_radiance_id: ResourceId,
    ray_direction_id: ResourceId,
    probe_params_id: ResourceId,
    irradiance_atlas_id: ResourceId,
    visibility_atlas_id: ResourceId,
    irradiance_pipeline: Option<wgpu::ComputePipeline>,
    visibility_pipeline: Option<wgpu::ComputePipeline>,
    bind_group: Option<wgpu::BindGroup>,
    frame_index: u32,
    shader: ShaderAsset,
}

impl ProbeUpdatePass {
    pub fn new(
        _graph: &mut RenderGraph,
        probe_atlas: &ProbeAtlasOutputs,
        probe_trace: &ProbeTraceOutputs,
        shader: ShaderAsset,
    ) -> Self {
        Self {
            node_id: None,
            ray_radiance_id: probe_trace.ray_radiance_id,
            ray_direction_id: probe_trace.ray_direction_id,
            probe_params_id: probe_atlas.probe_params_id,
            irradiance_atlas_id: probe_atlas.irradiance_atlas_id,
            visibility_atlas_id: probe_atlas.visibility_atlas_id,
            irradiance_pipeline: None,
            visibility_pipeline: None,
            bind_group: None,
            frame_index: 0,
            shader,
        }
    }
}

impl PassDesc for ProbeUpdatePass {
    fn name(&self) -> &'static str {
        "probe_update"
    }

    fn reads(&self) -> Vec<ResourceId> {
        vec![self.ray_radiance_id, self.ray_direction_id, self.probe_params_id]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![self.irradiance_atlas_id, self.visibility_atlas_id]
    }
}

impl Pass for ProbeUpdatePass {
    fn bind_node_id(&mut self, node_id: NodeId) {
        self.node_id = Some(node_id);
    }

    fn execute(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &PassContext<'_>,
        _world: &World,
    ) {
        let current = self.frame_index;
        self.frame_index = self.frame_index.wrapping_add(1);
        if current % 4 != 0 {
            return;
        }

        let Some(&radiance_buf) = ctx.buffers.get(&self.ray_radiance_id) else {
            return;
        };
        let Some(&direction_buf) = ctx.buffers.get(&self.ray_direction_id) else {
            return;
        };
        let Some(&probe_params_buf) = ctx.buffers.get(&self.probe_params_id) else {
            return;
        };
        let Some(&irradiance_view) = ctx.views.get(&self.irradiance_atlas_id) else {
            return;
        };
        let Some(&visibility_view) = ctx.views.get(&self.visibility_atlas_id) else {
            return;
        };

        if self.irradiance_pipeline.is_none() {
            let module = self
                .shader
                .compile(device)
                .expect("probe_update shader compile failed");

            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("probe_update_bgl"),
                entries: &[
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
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

            let pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("probe_update_pipeline_layout"),
                    bind_group_layouts: &[Some(&bgl)],
                    immediate_size: 0,
                });

            self.irradiance_pipeline = Some(device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("probe_update_irradiance_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some("update_irradiance"),
                    compilation_options: Default::default(),
                    cache: None,
                },
            ));

            self.visibility_pipeline = Some(device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("probe_update_visibility_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some("update_visibility"),
                    compilation_options: Default::default(),
                    cache: None,
                },
            ));

            self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("probe_update_bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: radiance_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: direction_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: probe_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(irradiance_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(visibility_view),
                    },
                ],
            }));
        }

        let irradiance_pipeline = self.irradiance_pipeline.as_ref().unwrap();
        let visibility_pipeline = self.visibility_pipeline.as_ref().unwrap();
        let bg = self.bind_group.as_ref().unwrap();

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("probe_update_irradiance"),
                timestamp_writes: None,
            });
            pass.set_pipeline(irradiance_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(1, 1, PROBE_COUNT);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("probe_update_visibility"),
                timestamp_writes: None,
            });
            pass.set_pipeline(visibility_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(1, 1, PROBE_COUNT);
        }
    }
}
