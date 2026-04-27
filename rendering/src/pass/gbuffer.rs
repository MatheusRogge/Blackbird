use std::{collections::HashMap, sync::Arc};
use wgpu::util::DeviceExt;

use engine_core::world::World;

use crate::{
    graph::{NodeId, RenderGraph},
    mesh::{Mesh, Vertex},
    pass::{Pass, PassContext, PassDesc},
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

    shader: ShaderAsset,
    pipeline: Option<wgpu::RenderPipeline>,
    camera_bind_group_layout: Option<wgpu::BindGroupLayout>,
    texture_bind_group_layout: Option<wgpu::BindGroupLayout>,

    texture_sampler: Option<wgpu::Sampler>,
    texture_cache: HashMap<usize, (wgpu::Texture, wgpu::TextureView)>,
    texture_bind_groups: HashMap<(usize, usize), wgpu::BindGroup>,
    fallback_albedo: Option<(wgpu::Texture, wgpu::TextureView)>,
    fallback_normal: Option<(wgpu::Texture, wgpu::TextureView)>,
    fallback_bind_group: Option<wgpu::BindGroup>,
    camera_bind_group: Option<wgpu::BindGroup>,
    // Vertex/index buffers keyed by mesh index. Meshes are only ever appended,
    // so indices are stable and each buffer is uploaded exactly once.
    vertex_buffer_cache: HashMap<usize, wgpu::Buffer>,
    index_buffer_cache: HashMap<usize, wgpu::Buffer>,

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

        let depth_id = graph.alloc_resource_id(ResourceDescriptor::ScreenTexture {
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
                shader,
                pipeline: None,
                camera_bind_group_layout: None,
                texture_bind_group_layout: None,
                texture_sampler: None,
                texture_cache: HashMap::new(),
                texture_bind_groups: HashMap::new(),
                fallback_albedo: None,
                fallback_normal: None,
                fallback_bind_group: None,
                camera_bind_group: None,
                vertex_buffer_cache: HashMap::new(),
                index_buffer_cache: HashMap::new(),
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
        format: wgpu::TextureFormat,
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
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

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
        albedo_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
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
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
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

            let fallback_albedo_asset = TextureAsset::new(1, 1, vec![255, 255, 255, 255]);
            let (fallback_albedo_tex, fallback_albedo_view) = Self::upload_texture(
                device,
                queue,
                &fallback_albedo_asset,
                wgpu::TextureFormat::Rgba8UnormSrgb,
            );

            // [128, 128, 255] decodes to (0, 0, 1) in tangent space — a flat normal map
            let fallback_normal_asset = TextureAsset::new(1, 1, vec![128, 128, 255, 255]);
            let (fallback_normal_tex, fallback_normal_view) = Self::upload_texture(
                device,
                queue,
                &fallback_normal_asset,
                wgpu::TextureFormat::Rgba8Unorm,
            );

            let fallback_bg = Self::create_texture_bind_group(
                device,
                &texture_layout,
                &sampler,
                &fallback_albedo_view,
                &fallback_normal_view,
            );

            self.pipeline = Some(pipeline);
            self.camera_bind_group_layout = Some(camera_layout);
            self.texture_bind_group_layout = Some(texture_layout);
            self.texture_sampler = Some(sampler);
            self.fallback_albedo = Some((fallback_albedo_tex, fallback_albedo_view));
            self.fallback_normal = Some((fallback_normal_tex, fallback_normal_view));
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
                    resource: buf.as_entire_binding(),
                }],
            }));
        }

        let meshes = world.get_entities::<Mesh>();

        // Upload any new textures and build missing bind groups before the render
        // pass opens, so no resource creation happens during command recording.
        for mesh in meshes.iter() {
            let albedo_key = mesh.albedo_texture.as_ref().map(|t| Arc::as_ptr(t) as usize);
            let normal_key = mesh.normal_texture.as_ref().map(|t| Arc::as_ptr(t) as usize);

            if let (Some(k), Some(tex)) = (albedo_key, &mesh.albedo_texture) {
                self.texture_cache.entry(k).or_insert_with(|| {
                    Self::upload_texture(device, queue, tex, wgpu::TextureFormat::Rgba8UnormSrgb)
                });
            }
            if let (Some(k), Some(tex)) = (normal_key, &mesh.normal_texture) {
                self.texture_cache.entry(k).or_insert_with(|| {
                    Self::upload_texture(device, queue, tex, wgpu::TextureFormat::Rgba8Unorm)
                });
            }

            let bg_key = (albedo_key.unwrap_or(0), normal_key.unwrap_or(0));
            if !self.texture_bind_groups.contains_key(&bg_key) {
                let av = albedo_key
                    .and_then(|k| self.texture_cache.get(&k))
                    .map(|(_, v)| v)
                    .or_else(|| self.fallback_albedo.as_ref().map(|(_, v)| v));
                let nv = normal_key
                    .and_then(|k| self.texture_cache.get(&k))
                    .map(|(_, v)| v)
                    .or_else(|| self.fallback_normal.as_ref().map(|(_, v)| v));

                if let (Some(av), Some(nv), Some(layout), Some(sampler)) = (
                    av,
                    nv,
                    self.texture_bind_group_layout.as_ref(),
                    self.texture_sampler.as_ref(),
                ) {
                    let bg = Self::create_texture_bind_group(device, layout, sampler, av, nv);
                    self.texture_bind_groups.insert(bg_key, bg);
                }
            }
        }

        // Upload vertex/index buffers for any new meshes before the render pass.
        for (mesh_idx, mesh) in meshes.iter().enumerate() {
            self.vertex_buffer_cache.entry(mesh_idx).or_insert_with(|| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("vertex_buffer"),
                    usage: wgpu::BufferUsages::VERTEX,
                    contents: mesh.get_vertex_buffer_content(),
                })
            });
            self.index_buffer_cache.entry(mesh_idx).or_insert_with(|| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("index_buffer"),
                    usage: wgpu::BufferUsages::INDEX,
                    contents: mesh.get_indices_buffer_content(),
                })
            });
        }

        // Sort draw order by bind group key so consecutive meshes that share a
        // material don't trigger a GPU bind group switch.
        let mut draw_order: Vec<usize> = (0..meshes.len()).collect();
        draw_order.sort_unstable_by_key(|&i| {
            let m = &meshes[i];
            let ak = m.albedo_texture.as_ref().map(|t| Arc::as_ptr(t) as usize).unwrap_or(0);
            let nk = m.normal_texture.as_ref().map(|t| Arc::as_ptr(t) as usize).unwrap_or(0);
            (ak, nk)
        });

        pass.set_pipeline(pipeline);

        if let Some(camera_bg) = self.camera_bind_group.as_ref() {
            pass.set_bind_group(0, camera_bg, &[]);
        }

        let mut current_bg_key: Option<(usize, usize)> = None;

        for mesh_idx in draw_order {
            let mesh = &meshes[mesh_idx];
            let albedo_key = mesh.albedo_texture.as_ref().map(|t| Arc::as_ptr(t) as usize);
            let normal_key = mesh.normal_texture.as_ref().map(|t| Arc::as_ptr(t) as usize);
            let bg_key = (albedo_key.unwrap_or(0), normal_key.unwrap_or(0));

            if current_bg_key != Some(bg_key) {
                let texture_bg = self
                    .texture_bind_groups
                    .get(&bg_key)
                    .or(self.fallback_bind_group.as_ref());

                if let Some(bg) = texture_bg {
                    pass.set_bind_group(1, bg, &[]);
                    current_bg_key = Some(bg_key);
                }
            }

            let vertex_buffer = &self.vertex_buffer_cache[&mesh_idx];
            let index_buffer = &self.index_buffer_cache[&mesh_idx];
            let index_count = mesh.get_indices_count() as u32;

            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..index_count, 0, 0..1);
        }
    }
}
