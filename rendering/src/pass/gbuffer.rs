use engine_core::world::World;

use crate::{
    camera::Camera,
    graph::{NodeId, RenderGraph},
    mesh::{Mesh, Vertex},
    arena::GeometryArena,
    pass::material::{MaterialKey, MaterialTextures, material_key},
    pass::{Pass, PassContext, PassDesc},
    resource::{BindingResource, ResourceDescriptor, ResourceId},
    shader::ShaderAsset,
};

pub struct GBufferOutputs {
    pub albedo_id: ResourceId,
    pub normal_id: ResourceId,
    pub material_id: ResourceId,
    /// Rgba16Float, linear emitted radiance (texture × factor × strength), not yet exposed.
    pub emissive_id: ResourceId,
    pub depth_id: ResourceId,
}

pub struct GBufferPass {
    node_id: Option<NodeId>,

    shader: ShaderAsset,
    pipeline: Option<wgpu::RenderPipeline>,
    /// Same as `pipeline` without back-face culling, for `Mesh::double_sided`.
    double_sided_pipeline: Option<wgpu::RenderPipeline>,
    camera_bind_group_layout: Option<wgpu::BindGroupLayout>,
    instance_layout: Option<wgpu::BindGroupLayout>,
    instance_bind_group: Option<(u64, wgpu::BindGroup)>,
    materials: Option<MaterialTextures>,
    camera_bind_group: Option<wgpu::BindGroup>,

    camera_buffer_id: ResourceId,
    albedo_id: ResourceId,
    normal_id: ResourceId,
    material_id: ResourceId,
    emissive_id: ResourceId,
    depth_id: ResourceId,
}

impl GBufferPass {
    pub fn new(
        graph: &mut RenderGraph,
        _camera_node_id: NodeId,
        camera_buffer_id: ResourceId,
        _surface_config: &wgpu::SurfaceConfiguration,
        shader: ShaderAsset,
    ) -> (Self, GBufferOutputs) {
        let albedo_id = graph.alloc_resource_id(ResourceDescriptor::ScreenTexture {
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        let normal_id = graph.alloc_resource_id(ResourceDescriptor::ScreenTexture {
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        let material_id = graph.alloc_resource_id(ResourceDescriptor::ScreenTexture {
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        let emissive_id = graph.alloc_resource_id(ResourceDescriptor::ScreenTexture {
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        let depth_id = graph.alloc_resource_id(ResourceDescriptor::ScreenTexture {
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        let outputs = GBufferOutputs {
            albedo_id,
            normal_id,
            material_id,
            emissive_id,
            depth_id,
        };

        (
            Self {
                node_id: None,
                shader,
                pipeline: None,
                double_sided_pipeline: None,
                camera_bind_group_layout: None,
                instance_layout: None,
                instance_bind_group: None,
                materials: None,
                camera_bind_group: None,
                camera_buffer_id,
                albedo_id,
                normal_id,
                material_id,
                emissive_id,
                depth_id,
            },
            outputs,
        )
    }
}

impl PassDesc for GBufferPass {
    fn name(&self) -> &'static str {
        "gbuffer"
    }

    fn reads(&self) -> Vec<ResourceId> {
        vec![self.camera_buffer_id]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![
            self.albedo_id,
            self.normal_id,
            self.material_id,
            self.emissive_id,
            self.depth_id,
        ]
    }

    fn layout_entries(&self) -> Vec<wgpu::BindGroupLayoutEntry> {
        vec![
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                count: None,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                count: None,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                count: None,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                count: None,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Depth,
                },
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                count: None,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            },
        ]
    }

    fn binding_resources(&self) -> Vec<BindingResource> {
        vec![
            BindingResource {
                slot: 0,
                resource_id: self.albedo_id,
                descriptor: ResourceDescriptor::ScreenTexture {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                },
            },
            BindingResource {
                slot: 1,
                resource_id: self.normal_id,
                descriptor: ResourceDescriptor::ScreenTexture {
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                },
            },
            BindingResource {
                slot: 2,
                resource_id: self.material_id,
                descriptor: ResourceDescriptor::ScreenTexture {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                },
            },
            BindingResource {
                slot: 3,
                resource_id: self.depth_id,
                descriptor: ResourceDescriptor::ScreenTexture {
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                },
            },
        ]
    }

    fn samplers(&self) -> Vec<(u32, wgpu::SamplerDescriptor<'static>)> {
        vec![(
            4,
            wgpu::SamplerDescriptor {
                label: Some("gbuffer_sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            },
        )]
    }
}

impl Pass for GBufferPass {
    fn bind_node_id(&mut self, node_id: NodeId) {
        self.node_id = Some(node_id);
    }

    fn on_resize(&mut self, _width: u32, _height: u32) {
        self.camera_bind_group = None;
    }

    fn execute(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &PassContext<'_>,
        world: &World,
    ) {
        let depth_view = ctx.views[&self.depth_id];
        let albedo_view = ctx.views[&self.albedo_id];
        let normal_view = ctx.views[&self.normal_id];
        let material_view = ctx.views[&self.material_id];
        let emissive_view = ctx.views[&self.emissive_id];

        if self.pipeline.is_none() {
            let shader_module = self
                .shader
                .compile(device)
                .expect("shader compilation failed");

            let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gbuffer_camera_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    count: None,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                }],
            });

            let materials = MaterialTextures::new(device, queue, "gbuffer_materials");

            let instance_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gbuffer_instance_layout"),
                entries: &[GeometryArena::instance_layout_entry(0)],
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gbuffer_pipeline_layout"),
                bind_group_layouts: &[Some(&camera_layout), Some(materials.layout()), Some(&instance_layout)],
                immediate_size: 0,
            });

            let create_pipeline = |label: &str, cull_mode: Option<wgpu::Face>| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader_module,
                        entry_point: Some("vs_main"),
                        buffers: &[Some(Vertex::buffer_descriptor())],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader_module,
                        entry_point: Some("fs_main"),
                        targets: &[
                            Some(wgpu::ColorTargetState {
                                format: wgpu::TextureFormat::Rgba8Unorm,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            }),
                            Some(wgpu::ColorTargetState {
                                format: wgpu::TextureFormat::Rgba16Float,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            }),
                            Some(wgpu::ColorTargetState {
                                format: wgpu::TextureFormat::Rgba8Unorm,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            }),
                            Some(wgpu::ColorTargetState {
                                format: wgpu::TextureFormat::Rgba16Float,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            }),
                        ],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode,
                        unclipped_depth: false,
                        polygon_mode: wgpu::PolygonMode::Fill,
                        conservative: false,
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                })
            };
            let pipeline = create_pipeline("gbuffer_pipeline", Some(wgpu::Face::Back));
            let double_sided_pipeline = create_pipeline("gbuffer_double_sided_pipeline", None);

            self.pipeline = Some(pipeline);
            self.double_sided_pipeline = Some(double_sided_pipeline);
            self.camera_bind_group_layout = Some(camera_layout);
            self.instance_layout = Some(instance_layout);
            self.materials = Some(materials);
        }

        let meshes: Vec<_> = world
            .get_entity_handles::<Mesh>()
            .into_iter()
            .filter(|h| !h.entity.is_blended())
            .collect();

        let (Some(pipeline), Some(double_sided_pipeline), Some(materials)) =
            (&self.pipeline, &self.double_sided_pipeline, &mut self.materials)
        else {
            return;
        };
        // Before the render pass opens, so no resource creation happens during command recording.
        materials.prepare(device, queue, encoder, meshes.iter().map(|h| h.entity));

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("gbuffer"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: albedo_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        store: wgpu::StoreOp::Store,
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: normal_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        store: wgpu::StoreOp::Store,
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: material_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        store: wgpu::StoreOp::Store,
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: emissive_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        store: wgpu::StoreOp::Store,
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    store: wgpu::StoreOp::Store,
                    load: wgpu::LoadOp::Clear(0.0),
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });

        if self.camera_bind_group.is_none()
            && let (Some(layout), Some(buf)) = (
                self.camera_bind_group_layout.as_ref(),
                ctx.buffers.get(&self.camera_buffer_id),
            )
        {
            self.camera_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gbuffer_camera_bg"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.binding(),
                }],
            }));
        }

        let generation = ctx.geometry.generation();
        if self.instance_bind_group.as_ref().is_none_or(|(g, _)| *g != generation)
            && let Some(layout) = self.instance_layout.as_ref()
        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gbuffer_instances"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ctx.geometry.instance_buffer().as_entire_binding(),
                }],
            });
            self.instance_bind_group = Some((generation, bind_group));
        }

        // Group draws by pipeline, then by bind group key, so consecutive meshes
        // that share a material don't trigger a GPU state switch.
        let (w, h) = ctx.surface_size;
        let view_proj = world.get_entities::<Camera>().first().map(|c| c.view_proj_matrix(w as f32 / h as f32));
        let in_view = |i: usize| {
            let handle = &meshes[i];
            view_proj.is_none_or(|vp| {
                ctx.geometry.get(handle.id).is_some_and(|d| !d.bounds.outside_clip(vp * handle.entity.transform))
            })
        };
        let mut draw_order: Vec<usize> = (0..meshes.len()).filter(|&i| in_view(i)).collect();
        draw_order.sort_unstable_by_key(|&i| {
            let m = meshes[i].entity;
            (m.double_sided, material_key(m))
        });

        pass.set_pipeline(pipeline);
        let mut double_sided = false;
        if let Some((_, instance_bg)) = self.instance_bind_group.as_ref() {
            pass.set_bind_group(2, instance_bg, &[]);
        }
        pass.set_vertex_buffer(0, ctx.geometry.vertex_buffer().slice(..));
        pass.set_index_buffer(ctx.geometry.index_buffer().slice(..), wgpu::IndexFormat::Uint32);

        if let Some(camera_bg) = self.camera_bind_group.as_ref() {
            pass.set_bind_group(0, camera_bg, &[]);
        }

        let mut current_bg_key: Option<MaterialKey> = None;

        for i in draw_order {
            let handle = &meshes[i];
            let Some(draw) = ctx.geometry.get(handle.id) else {
                continue;
            };
            let mesh = handle.entity;
            let bg_key = material_key(mesh);

            if mesh.double_sided != double_sided {
                double_sided = mesh.double_sided;
                pass.set_pipeline(if double_sided { double_sided_pipeline } else { pipeline });
            }

            if current_bg_key != Some(bg_key) {
                pass.set_bind_group(1, materials.bind_group(bg_key), &[]);
                current_bg_key = Some(bg_key);
            }

            pass.draw_indexed(draw.indices.clone(), draw.base_vertex, draw.instances());
        }
    }
}
