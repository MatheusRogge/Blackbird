use std::{collections::HashMap, sync::Arc};

use engine_core::world::World;

use crate::{
    graph::{NodeId, RenderGraph},
    mesh::{Mesh, Vertex},
    pass::{PassContext, RenderPass, RenderPassDesc},
    resource::{BindingResource, ResourceDescriptor, ResourceId},
    shader::ShaderAsset,
    texture::TextureAsset,
};

pub struct GBufferOutputs {
    pub albedo_id: ResourceId,
    pub normal_id: ResourceId,
    pub material_id: ResourceId,
    pub depth_id: ResourceId,
}

pub struct GBufferPass {
    node_id: Option<NodeId>,
    surface_size: wgpu::Extent3d,

    shader: ShaderAsset,
    pipeline: Option<Arc<wgpu::RenderPipeline>>,
    camera_bind_group_layout: Option<wgpu::BindGroupLayout>,
    texture_bind_group_layout: Option<wgpu::BindGroupLayout>,

    texture_sampler: Option<wgpu::Sampler>,
    texture_cache: HashMap<usize, (wgpu::Texture, wgpu::TextureView)>,
    texture_bind_groups: HashMap<usize, wgpu::BindGroup>,
    fallback_texture: Option<(wgpu::Texture, wgpu::TextureView)>,
    fallback_bind_group: Option<wgpu::BindGroup>,

    camera_buffer_id: ResourceId,
    albedo_id: ResourceId,
    normal_id: ResourceId,
    material_id: ResourceId,
    depth_id: ResourceId,
}

impl GBufferPass {
    pub fn new(
        graph: &mut RenderGraph,
        _camera_node_id: NodeId,
        camera_buffer_id: ResourceId,
        surface_config: &wgpu::SurfaceConfiguration,
        shader: ShaderAsset,
    ) -> (Self, GBufferOutputs) {
        let size = wgpu::Extent3d {
            width: surface_config.width,
            height: surface_config.height,
            depth_or_array_layers: 1,
        };

        let albedo_id = graph.alloc_resource_id(ResourceDescriptor::Texture {
            size,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        let normal_id = graph.alloc_resource_id(ResourceDescriptor::Texture {
            size,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        let material_id = graph.alloc_resource_id(ResourceDescriptor::Texture {
            size,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        let depth_id = graph.alloc_resource_id(ResourceDescriptor::Texture {
            size,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        let outputs = GBufferOutputs {
            albedo_id,
            normal_id,
            material_id,
            depth_id,
        };

        (
            Self {
                node_id: None,
                surface_size: size,
                shader,
                pipeline: None,
                camera_bind_group_layout: None,
                texture_bind_group_layout: None,
                texture_sampler: None,
                texture_cache: HashMap::new(),
                texture_bind_groups: HashMap::new(),
                fallback_texture: None,
                fallback_bind_group: None,
                camera_buffer_id,
                albedo_id,
                normal_id,
                material_id,
                depth_id,
            },
            outputs,
        )
    }

    fn upload_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        asset: &TextureAsset,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let size = wgpu::Extent3d {
            width: asset.width,
            height: asset.height,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // bytes_per_row must be a multiple of COPY_BYTES_PER_ROW_ALIGNMENT (256).
        let unpadded = 4 * asset.width;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let bytes_per_row = unpadded.div_ceil(align) * align;

        if bytes_per_row == unpadded {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &asset.data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
                size,
            );
        } else {
            // Pad each row to meet alignment requirements.
            let padded: Vec<u8> = asset
                .data
                .chunks_exact(unpadded as usize)
                .flat_map(|row| {
                    let padding = bytes_per_row as usize - unpadded as usize;
                    row.iter().copied().chain(std::iter::repeat_n(0, padding))
                })
                .collect();

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &padded,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
                size,
            );
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_texture_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    }
}

impl RenderPassDesc for GBufferPass {
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
                descriptor: ResourceDescriptor::Texture {
                    size: self.surface_size,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                },
            },
            BindingResource {
                slot: 1,
                resource_id: self.normal_id,
                descriptor: ResourceDescriptor::Texture {
                    size: self.surface_size,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                },
            },
            BindingResource {
                slot: 2,
                resource_id: self.material_id,
                descriptor: ResourceDescriptor::Texture {
                    size: self.surface_size,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                },
            },
            BindingResource {
                slot: 3,
                resource_id: self.depth_id,
                descriptor: ResourceDescriptor::Texture {
                    size: self.surface_size,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
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

impl RenderPass for GBufferPass {
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
        let depth_view = ctx.views[&self.depth_id];
        let albedo_view = ctx.views[&self.albedo_id];
        let normal_view = ctx.views[&self.normal_id];
        let material_view = ctx.views[&self.material_id];

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

            let texture_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("gbuffer_texture_layout"),
                    entries: &[
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
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        },
                    ],
                });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gbuffer_pipeline_layout"),
                bind_group_layouts: &[Some(&camera_layout), Some(&texture_layout)],
                immediate_size: 0,
            });

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("gbuffer_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader_module,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_descriptor()],
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
                    ],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
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
            });

            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("gbuffer_texture_sampler"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                address_mode_w: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });

            // 1×1 white fallback texture
            let fallback_asset = TextureAsset::new(1, 1, vec![255, 255, 255, 255]);
            let (fallback_tex, fallback_view) =
                Self::upload_texture(device, queue, &fallback_asset);
            let fallback_bg =
                Self::create_texture_bind_group(device, &texture_layout, &sampler, &fallback_view);

            self.pipeline = Some(Arc::new(pipeline));
            self.camera_bind_group_layout = Some(camera_layout);
            self.texture_bind_group_layout = Some(texture_layout);
            self.texture_sampler = Some(sampler);
            self.fallback_texture = Some((fallback_tex, fallback_view));
            self.fallback_bind_group = Some(fallback_bg);
        }

        let pipeline = self.pipeline.as_ref().unwrap();

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

        pass.set_pipeline(pipeline);

        if let (Some(layout), Some(buf)) = (
            self.camera_bind_group_layout.as_ref(),
            ctx.buffers.get(&self.camera_buffer_id),
        ) {
            let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gbuffer_camera_bg"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            pass.set_bind_group(0, &camera_bg, &[]);
        }

        let meshes = world.get_entities::<Mesh>();

        for mesh in meshes {
            if let Some(tex_arc) = &mesh.albedo_texture {
                let key = Arc::as_ptr(tex_arc) as usize;
                if !self.texture_cache.contains_key(&key) {
                    let (tex, view) = Self::upload_texture(device, queue, tex_arc);
                    if let (Some(layout), Some(sampler)) = (
                        self.texture_bind_group_layout.as_ref(),
                        self.texture_sampler.as_ref(),
                    ) {
                        let bg = Self::create_texture_bind_group(device, layout, sampler, &view);
                        self.texture_bind_groups.insert(key, bg);
                    }
                    self.texture_cache.insert(key, (tex, view));
                }
            }

            let texture_bg = match &mesh.albedo_texture {
                Some(tex_arc) => {
                    let key = Arc::as_ptr(tex_arc) as usize;
                    self.texture_bind_groups.get(&key)
                }
                None => None,
            }
            .or(self.fallback_bind_group.as_ref());

            if let Some(bg) = texture_bg {
                pass.set_bind_group(1, bg, &[]);
            }

            let vertex_buffer = mesh.get_vertex_buffer(device);
            let index_buffer = mesh.get_index_buffer(device);
            let index_count = mesh.get_indices_count() as u32;

            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..index_count, 0, 0..1);
        }
    }
}
