use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use engine_core::world::World;

use crate::{
    arena::BufferRange,
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
    ray_radiance_id:      ResourceId,
    ray_direction_id:     ResourceId,
    probe_params_id:      ResourceId,
    irradiance_atlas_ids: [ResourceId; 2],
    visibility_atlas_ids: [ResourceId; 2],
    current_atlas:        Arc<AtomicUsize>,

    irradiance_pipelines: [Option<wgpu::ComputePipeline>; 2],
    visibility_pipelines:  [Option<wgpu::ComputePipeline>; 2],
    // bind_groups[i] reads atlas[i] and writes atlas[1-i]
    bind_groups: [Option<wgpu::BindGroup>; 2],

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
            ray_radiance_id:      probe_trace.ray_radiance_id,
            ray_direction_id:     probe_trace.ray_direction_id,
            probe_params_id:      probe_atlas.probe_params_id,
            irradiance_atlas_ids: probe_atlas.irradiance_atlas_ids,
            visibility_atlas_ids: probe_atlas.visibility_atlas_ids,
            current_atlas:        Arc::clone(&probe_atlas.current_atlas),

            irradiance_pipelines: [None, None],
            visibility_pipelines:  [None, None],
            bind_groups:          [None, None],

            frame_index: 0,
            shader,
        }
    }

    fn build_bind_groups(
        &mut self,
        device: &wgpu::Device,
        radiance_buf:    BufferRange,
        direction_buf:   BufferRange,
        probe_params_buf: BufferRange,
        irr_views:  [&wgpu::TextureView; 2],
        vis_views:  [&wgpu::TextureView; 2],
    ) {
        let module = self.shader.compile(device).expect("probe_update shader compile failed");

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("probe_update_bgl"),
            entries: &[
                // 0: ray_radiance
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
                // 1: ray_direction
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
                // 2: probe_params
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
                // 3: irradiance_prev (sampled, previous frame's atlas)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                // 4: irradiance_next (write-only storage)
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
                // 5: visibility_prev
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                // 6: visibility_next (write-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("probe_update_pipeline_layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        for slot in 0..2usize {
            let irr_prev = irr_views[slot];
            let irr_next = irr_views[1 - slot];
            let vis_prev = vis_views[slot];
            let vis_next = vis_views[1 - slot];

            self.bind_groups[slot] = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("probe_update_bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: radiance_buf.binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: direction_buf.binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: probe_params_buf.binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(irr_prev) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(irr_next) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(vis_prev) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(vis_next) },
                ],
            }));
        }

        let make_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        self.irradiance_pipelines = [Some(make_pipeline("update_irradiance")), Some(make_pipeline("update_irradiance"))];
        self.visibility_pipelines  = [Some(make_pipeline("update_visibility")),  Some(make_pipeline("update_visibility"))];
    }
}

impl PassDesc for ProbeUpdatePass {
    fn name(&self) -> &'static str { "probe_update" }

    fn reads(&self) -> Vec<ResourceId> {
        vec![self.ray_radiance_id, self.ray_direction_id, self.probe_params_id]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![
            self.irradiance_atlas_ids[0], self.irradiance_atlas_ids[1],
            self.visibility_atlas_ids[0], self.visibility_atlas_ids[1],
        ]
    }
}

impl Pass for ProbeUpdatePass {
    fn bind_node_id(&mut self, node_id: NodeId) { self.node_id = Some(node_id); }

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
        if !current.is_multiple_of(4) {
            return;
        }

        let Some(&radiance_buf)    = ctx.buffers.get(&self.ray_radiance_id)  else { return };
        let Some(&direction_buf)   = ctx.buffers.get(&self.ray_direction_id) else { return };
        let Some(&probe_params_buf) = ctx.buffers.get(&self.probe_params_id)  else { return };

        let Some(&irr0) = ctx.views.get(&self.irradiance_atlas_ids[0]) else { return };
        let Some(&irr1) = ctx.views.get(&self.irradiance_atlas_ids[1]) else { return };
        let Some(&vis0) = ctx.views.get(&self.visibility_atlas_ids[0]) else { return };
        let Some(&vis1) = ctx.views.get(&self.visibility_atlas_ids[1]) else { return };

        if self.bind_groups[0].is_none() {
            self.build_bind_groups(device, radiance_buf, direction_buf, probe_params_buf,
                                   [irr0, irr1], [vis0, vis1]);
        }

        // read_idx is the slot that was last written (the "current" good data).
        // We read from it and write to the other slot.
        let read_idx  = self.current_atlas.load(Ordering::Relaxed);
        let write_idx = 1 - read_idx;

        let irr_pipeline = self.irradiance_pipelines[read_idx].as_ref().unwrap();
        let vis_pipeline  = self.visibility_pipelines[read_idx].as_ref().unwrap();
        let bg = self.bind_groups[read_idx].as_ref().unwrap();

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("probe_update_irradiance"),
                timestamp_writes: None,
            });
            pass.set_pipeline(irr_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(1, 1, PROBE_COUNT);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("probe_update_visibility"),
                timestamp_writes: None,
            });
            pass.set_pipeline(vis_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(1, 1, PROBE_COUNT);
        }

        // Flip: write_idx is now the freshly-written atlas.
        self.current_atlas.store(write_idx, Ordering::Relaxed);
    }
}
