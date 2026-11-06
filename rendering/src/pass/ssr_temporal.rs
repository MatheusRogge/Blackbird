//! Temporal accumulation of SSR: blends the jittered per-frame `ssr` output into a reprojected
//! history (`ssr_temporal.wgsl`). `composite` reads the result instead of raw `ssr`.

use engine_core::world::World;

use crate::{
    graph::{NodeId, RenderGraph},
    pass::{Pass, PassContext, PassDesc},
    resource::{ResourceDescriptor, ResourceId},
    shader::ShaderAsset,
};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Last frame's output. Pass-owned, since graph screen textures may be aliased between frames.
struct History {
    size: (u32, u32),
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

pub struct SsrTemporalPass {
    node_id: Option<NodeId>,
    shader: ShaderAsset,
    ssr_id: ResourceId,
    depth_id: ResourceId,
    specular_weight_id: ResourceId,
    camera_buffer_id: ResourceId,
    output_id: ResourceId,
    pipeline: Option<(wgpu::RenderPipeline, wgpu::BindGroupLayout, wgpu::Sampler)>,
    history: Option<History>,
    /// Keyed by the input view addresses (screen textures move on resize).
    bind_group: Option<(Vec<usize>, wgpu::BindGroup)>,
}

impl SsrTemporalPass {
    pub fn new(
        graph: &mut RenderGraph,
        ssr_id: ResourceId,
        depth_id: ResourceId,
        specular_weight_id: ResourceId,
        camera_buffer_id: ResourceId,
        shader: ShaderAsset,
    ) -> (Self, ResourceId) {
        let output_id = graph.alloc_resource_id(ResourceDescriptor::ScreenTexture {
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        });
        let pass = Self {
            node_id: None,
            shader,
            ssr_id,
            depth_id,
            specular_weight_id,
            camera_buffer_id,
            output_id,
            pipeline: None,
            history: None,
            bind_group: None,
        };
        (pass, output_id)
    }

    fn create_pipeline(&self, device: &wgpu::Device) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout, wgpu::Sampler) {
        let module = self.shader.compile(device).expect("ssr_temporal shader compilation failed");
        let texture = |binding, sample_type| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                view_dimension: wgpu::TextureViewDimension::D2,
                sample_type,
            },
            count: None,
        };
        let float = wgpu::TextureSampleType::Float { filterable: true };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssr_temporal"),
            entries: &[
                texture(0, float),
                texture(1, float),
                texture(2, wgpu::TextureSampleType::Depth),
                texture(3, float),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssr_temporal"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ssr_temporal"),
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
                targets: &[Some(FORMAT.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ssr_temporal"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        (pipeline, layout, sampler)
    }

    /// A cleared history (zero confidence), so the first frames fade in from the current SSR.
    fn create_history(device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, (width, height): (u32, u32)) -> History {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ssr_history"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ssr_history_clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        History { size: (width, height), texture, view }
    }
}

impl PassDesc for SsrTemporalPass {
    fn name(&self) -> &'static str {
        "ssr_temporal"
    }

    fn reads(&self) -> Vec<ResourceId> {
        vec![self.ssr_id, self.depth_id, self.specular_weight_id, self.camera_buffer_id]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![self.output_id]
    }
}

impl Pass for SsrTemporalPass {
    fn bind_node_id(&mut self, node_id: NodeId) {
        self.node_id = Some(node_id);
    }

    fn on_resize(&mut self, _width: u32, _height: u32) {
        self.history = None;
        self.bind_group = None;
    }

    fn execute(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &PassContext<'_>,
        _world: &World,
    ) {
        let (Some(&output_view), Some(&output), Some(&ssr), Some(&depth), Some(&weight), Some(&camera)) = (
            ctx.views.get(&self.output_id),
            ctx.textures.get(&self.output_id),
            ctx.views.get(&self.ssr_id),
            ctx.views.get(&self.depth_id),
            ctx.views.get(&self.specular_weight_id),
            ctx.buffers.get(&self.camera_buffer_id),
        ) else {
            return;
        };
        if self.pipeline.is_none() {
            self.pipeline = Some(self.create_pipeline(device));
        }
        if self.history.as_ref().is_none_or(|h| h.size != ctx.surface_size) {
            self.history = Some(Self::create_history(device, encoder, ctx.surface_size));
            self.bind_group = None;
        }
        let (Some((pipeline, layout, sampler)), Some(history)) = (&self.pipeline, &self.history) else { return };

        let key: Vec<usize> = [ssr, depth, weight].iter().map(|v| std::ptr::from_ref(*v) as usize).collect();
        if self.bind_group.as_ref().is_none_or(|(k, _)| *k != key) {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ssr_temporal"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(ssr) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&history.view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(depth) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(weight) },
                    wgpu::BindGroupEntry { binding: 4, resource: camera.binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(sampler) },
                ],
            });
            self.bind_group = Some((key, bind_group));
        }
        let Some((_, bind_group)) = &self.bind_group else { return };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ssr_temporal"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
        drop(pass);

        let (width, height) = history.size;
        encoder.copy_texture_to_texture(
            output.as_image_copy(),
            history.texture.as_image_copy(),
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
    }
}
