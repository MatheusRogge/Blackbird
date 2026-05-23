use engine_core::world::World;

use crate::{
    graph::{NodeId, RenderGraph},
    pass::cluster_assignment::ClusterOutputs,
    pass::gbuffer::GBufferOutputs,
    pass::light_upload::LightBuffers,
    pass::probe_atlas::ProbeAtlasOutputs,
    pass::shadow::ShadowOutputs,
    pass::{Pass, PassContext, PassDesc},
    resource::{ResourceDescriptor, ResourceId},
    shader::ShaderAsset,
};

pub struct LightingOutputs {
    pub lit_id: ResourceId,
}

pub struct LightingPass {
    node_id: Option<NodeId>,
    shader: ShaderAsset,
    pipeline: Option<wgpu::RenderPipeline>,
    gbuffer_bgl: Option<wgpu::BindGroupLayout>,
    cluster_bgl: Option<wgpu::BindGroupLayout>,
    sampler: Option<wgpu::Sampler>,

    albedo_id: ResourceId,
    normal_id: ResourceId,
    material_id: ResourceId,
    depth_id: ResourceId,

    cluster_params_id: ResourceId,
    point_buffer_id: ResourceId,
    light_grid_id: ResourceId,
    light_indices_id: ResourceId,
    sky_buffer_id: ResourceId,

    shadow_map_id: ResourceId,
    shadow_params_id: ResourceId,

    lit_id: ResourceId,
    surface_format: wgpu::TextureFormat,

    // cluster_bg only binds buffers — cached indefinitely.
    // gbuffer_bg binds texture views — recreated when albedo is reallocated (resize).
    cluster_bg: Option<wgpu::BindGroup>,
    gbuffer_bg: Option<wgpu::BindGroup>,
    cached_albedo_ptr: usize,

    shadow_bgl: Option<wgpu::BindGroupLayout>,
    shadow_bg: Option<wgpu::BindGroup>,
    shadow_sampler: Option<wgpu::Sampler>,
    cached_shadow_ptr: usize,

    irradiance_atlas_id: ResourceId,
    visibility_atlas_id: ResourceId,
    probe_params_id: ResourceId,
    camera_buffer_id: ResourceId,
    probe_bgl: Option<wgpu::BindGroupLayout>,
    probe_bg: Option<wgpu::BindGroup>,
    probe_sampler: Option<wgpu::Sampler>,
}

impl LightingPass {
    pub fn new(
        graph: &mut RenderGraph,
        gbuffer: &GBufferOutputs,
        cluster: &ClusterOutputs,
        light_buffers: &LightBuffers,
        shadow: &ShadowOutputs,
        probe_atlas: &ProbeAtlasOutputs,
        camera_buffer_id: ResourceId,
        _surface_config: &wgpu::SurfaceConfiguration,
        shader: ShaderAsset,
    ) -> (Self, LightingOutputs) {
        let lit_id = graph.alloc_resource_id(ResourceDescriptor::ScreenTexture {
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        let outputs = LightingOutputs { lit_id };

        (
            Self {
                node_id: None,
                shader,
                pipeline: None,
                gbuffer_bgl: None,
                cluster_bgl: None,
                sampler: None,
                albedo_id: gbuffer.albedo_id,
                normal_id: gbuffer.normal_id,
                material_id: gbuffer.material_id,
                depth_id: gbuffer.depth_id,
                cluster_params_id: cluster.cluster_params_id,
                point_buffer_id: light_buffers.point_buffer_id,
                light_grid_id: cluster.light_grid_id,
                light_indices_id: cluster.light_indices_id,
                sky_buffer_id: light_buffers.sky_buffer_id,
                shadow_map_id: shadow.shadow_map_id,
                shadow_params_id: shadow.shadow_params_id,
                lit_id,
                surface_format: wgpu::TextureFormat::Rgba16Float,
                cluster_bg: None,
                gbuffer_bg: None,
                cached_albedo_ptr: 0,
                shadow_bgl: None,
                shadow_bg: None,
                shadow_sampler: None,
                cached_shadow_ptr: 0,
                irradiance_atlas_id: probe_atlas.irradiance_atlas_id,
                visibility_atlas_id: probe_atlas.visibility_atlas_id,
                probe_params_id: probe_atlas.probe_params_id,
                camera_buffer_id,
                probe_bgl: None,
                probe_bg: None,
                probe_sampler: None,
            },
            outputs,
        )
    }
}

impl PassDesc for LightingPass {
    fn name(&self) -> &'static str {
        "lighting"
    }

    fn reads(&self) -> Vec<ResourceId> {
        vec![
            self.albedo_id,
            self.normal_id,
            self.material_id,
            self.depth_id,
            self.cluster_params_id,
            self.point_buffer_id,
            self.light_grid_id,
            self.light_indices_id,
            self.sky_buffer_id,
            self.shadow_map_id,
            self.shadow_params_id,
            self.irradiance_atlas_id,
            self.visibility_atlas_id,
            self.probe_params_id,
            self.camera_buffer_id,
        ]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![self.lit_id]
    }
}

impl Pass for LightingPass {
    fn bind_node_id(&mut self, node_id: NodeId) {
        self.node_id = Some(node_id);
    }

    fn on_resize(&mut self, _width: u32, _height: u32) {
        self.gbuffer_bg = None;
        self.cached_albedo_ptr = 0;
    }

    fn execute(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &PassContext<'_>,
        _world: &World,
    ) {
        let Some(&lit_view) = ctx.views.get(&self.lit_id) else {
            return;
        };

        let Some(&albedo_view) = ctx.views.get(&self.albedo_id) else {
            return;
        };

        let Some(&normal_view) = ctx.views.get(&self.normal_id) else {
            return;
        };

        let Some(&material_view) = ctx.views.get(&self.material_id) else {
            return;
        };

        let Some(&depth_view) = ctx.views.get(&self.depth_id) else {
            return;
        };

        let Some(&params_buf) = ctx.buffers.get(&self.cluster_params_id) else {
            return;
        };

        let Some(&point_buf) = ctx.buffers.get(&self.point_buffer_id) else {
            return;
        };

        let Some(&grid_buf) = ctx.buffers.get(&self.light_grid_id) else {
            return;
        };

        let Some(&indices_buf) = ctx.buffers.get(&self.light_indices_id) else {
            return;
        };

        let Some(&sky_buf) = ctx.buffers.get(&self.sky_buffer_id) else {
            return;
        };

        let Some(&shadow_params_buf) = ctx.buffers.get(&self.shadow_params_id) else {
            return;
        };

        let Some(&shadow_map_view) = ctx.views.get(&self.shadow_map_id) else {
            return;
        };

        let Some(&irradiance_view) = ctx.views.get(&self.irradiance_atlas_id) else {
            return;
        };

        let Some(&visibility_view) = ctx.views.get(&self.visibility_atlas_id) else {
            return;
        };

        let Some(&probe_params_buf) = ctx.buffers.get(&self.probe_params_id) else {
            return;
        };

        let Some(&camera_buf) = ctx.buffers.get(&self.camera_buffer_id) else {
            return;
        };

        if self.pipeline.is_none() {
            let module = self
                .shader
                .compile(device)
                .expect("lighting shader compile failed");

            let gbuffer_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("lighting_gbuffer_bgl"),
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
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Depth,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

            let cluster_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("lighting_cluster_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            let shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("lighting_shadow_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Depth,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
                ],
            });

            let probe_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("lighting_probe_bgl"),
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
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            let probe_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("probe_atlas_sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });

            self.probe_bgl = Some(probe_bgl);
            self.probe_sampler = Some(probe_sampler);

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("lighting_pipeline_layout"),
                bind_group_layouts: &[
                    Some(&gbuffer_bgl),
                    Some(&cluster_bgl),
                    Some(&shadow_bgl),
                    Some(self.probe_bgl.as_ref().unwrap()),
                ],
                immediate_size: 0,
            });

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("lighting_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.surface_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("lighting_sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });

            self.pipeline = Some(pipeline);
            self.gbuffer_bgl = Some(gbuffer_bgl);
            self.cluster_bgl = Some(cluster_bgl);
            self.sampler = Some(sampler);
            self.shadow_bgl = Some(shadow_bgl);
        }

        let pipeline = self.pipeline.as_ref().unwrap();
        let gbuffer_bgl = self.gbuffer_bgl.as_ref().unwrap();
        let cluster_bgl = self.cluster_bgl.as_ref().unwrap();
        let sampler = self.sampler.as_ref().unwrap();

        // Cluster bind group: all buffer bindings, never changes.
        if self.cluster_bg.is_none() {
            self.cluster_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lighting_cluster_bg"),
                layout: cluster_bgl,
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
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: sky_buf.as_entire_binding(),
                    },
                ],
            }));
        }

        // GBuffer bind group: texture view bindings, recreated only when the
        // textures are reallocated (i.e. on resize).
        let albedo_ptr = albedo_view as *const wgpu::TextureView as usize;
        if self.gbuffer_bg.is_none() || albedo_ptr != self.cached_albedo_ptr {
            self.gbuffer_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lighting_gbuffer_bg"),
                layout: gbuffer_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(albedo_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(normal_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(material_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            }));
            self.cached_albedo_ptr = albedo_ptr;
        }

        let shadow_bgl = self.shadow_bgl.as_ref().unwrap();
        let shadow_ptr = shadow_map_view as *const wgpu::TextureView as usize;
        if self.shadow_bg.is_none() || shadow_ptr != self.cached_shadow_ptr {
            let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("shadow_comparison_sampler"),
                compare: Some(wgpu::CompareFunction::LessEqual),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
            self.shadow_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lighting_shadow_bg"),
                layout: shadow_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: shadow_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(shadow_map_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                    },
                ],
            }));
            self.shadow_sampler = Some(shadow_sampler);
            self.cached_shadow_ptr = shadow_ptr;
        }

        if self.probe_bg.is_none() {
            let probe_bgl = self.probe_bgl.as_ref().unwrap();
            let probe_sampler = self.probe_sampler.as_ref().unwrap();
            self.probe_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lighting_probe_bg"),
                layout: probe_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(irradiance_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(visibility_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: probe_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(probe_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: camera_buf.as_entire_binding(),
                    },
                ],
            }));
        }

        let cluster_bg: &wgpu::BindGroup = self.cluster_bg.as_ref().unwrap();
        let gbuffer_bg: &wgpu::BindGroup = self.gbuffer_bg.as_ref().unwrap();
        let shadow_bg: &wgpu::BindGroup = self.shadow_bg.as_ref().unwrap();
        let probe_bg: &wgpu::BindGroup = self.probe_bg.as_ref().unwrap();

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("lighting"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: lit_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    store: wgpu::StoreOp::Store,
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, gbuffer_bg, &[]);
        pass.set_bind_group(1, cluster_bg, &[]);
        pass.set_bind_group(2, shadow_bg, &[]);
        pass.set_bind_group(3, probe_bg, &[]);
        pass.draw(0..3, 0..1);
    }
}
