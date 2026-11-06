use engine_core::world::World;

use crate::{
    graph::{NodeId, RenderGraph},
    pass::{Pass, PassContext, PassDesc},
    resource::{ResourceDescriptor, ResourceId},
    shader::ShaderAsset,
};

#[derive(Clone, Copy)]
pub enum FullscreenInput {
    Color(ResourceId),
    Depth(ResourceId),
}

impl FullscreenInput {
    fn id(self) -> ResourceId {
        match self {
            Self::Color(id) | Self::Depth(id) => id,
        }
    }
}

/// A fullscreen triangle that reads screen textures and uniforms and writes one screen texture.
/// The shader's group 0 holds the inputs at bindings `0..n`, then the uniforms, then a linear
/// clamp-to-edge sampler. Its entry points are `vs_main` and `fs_main`.
pub struct FullscreenPass {
    name: &'static str,
    node_id: Option<NodeId>,
    shader: ShaderAsset,
    inputs: Vec<FullscreenInput>,
    uniforms: Vec<ResourceId>,
    output_id: ResourceId,
    output_format: wgpu::TextureFormat,
    pipeline: Option<(wgpu::RenderPipeline, wgpu::BindGroupLayout, wgpu::Sampler)>,
    /// Keyed by the input views' addresses, which change when screen textures are reallocated.
    bind_group: Option<(Vec<usize>, wgpu::BindGroup)>,
}

impl FullscreenPass {
    pub fn new(
        graph: &mut RenderGraph,
        name: &'static str,
        inputs: Vec<FullscreenInput>,
        uniforms: Vec<ResourceId>,
        output_format: wgpu::TextureFormat,
        shader: ShaderAsset,
    ) -> (Self, ResourceId) {
        let output_id = graph.alloc_resource_id(ResourceDescriptor::ScreenTexture {
            format: output_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
        });
        let pass = Self {
            name,
            node_id: None,
            shader,
            inputs,
            uniforms,
            output_id,
            output_format,
            pipeline: None,
            bind_group: None,
        };
        (pass, output_id)
    }

    fn create_pipeline(&self, device: &wgpu::Device) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout, wgpu::Sampler) {
        let module = self.shader.compile(device).expect("fullscreen shader compilation failed");

        let textures = self.inputs.iter().map(|input| match input {
            FullscreenInput::Color(_) => wgpu::TextureSampleType::Float { filterable: true },
            FullscreenInput::Depth(_) => wgpu::TextureSampleType::Depth,
        });
        let entries: Vec<wgpu::BindGroupLayoutEntry> = textures
            .map(|sample_type| wgpu::BindingType::Texture {
                multisampled: false,
                view_dimension: wgpu::TextureViewDimension::D2,
                sample_type,
            })
            .chain(self.uniforms.iter().map(|_| wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            }))
            .chain([wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)])
            .enumerate()
            .map(|(binding, ty)| wgpu::BindGroupLayoutEntry {
                binding: binding as u32,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty,
                count: None,
            })
            .collect();

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(self.name),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(self.name),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(self.name),
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
                    format: self.output_format,
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
            label: Some(self.name),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        (pipeline, layout, sampler)
    }
}

impl PassDesc for FullscreenPass {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reads(&self) -> Vec<ResourceId> {
        self.inputs.iter().map(|i| i.id()).chain(self.uniforms.iter().copied()).collect()
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![self.output_id]
    }
}

impl Pass for FullscreenPass {
    fn bind_node_id(&mut self, node_id: NodeId) {
        self.node_id = Some(node_id);
    }

    fn on_resize(&mut self, _width: u32, _height: u32) {
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
        let Some(&output_view) = ctx.views.get(&self.output_id) else { return };
        let Some(views) = self.inputs.iter().map(|i| ctx.views.get(&i.id()).copied()).collect::<Option<Vec<_>>>()
        else {
            return;
        };
        let Some(uniforms) = self.uniforms.iter().map(|id| ctx.buffers.get(id)).collect::<Option<Vec<_>>>() else {
            return;
        };

        if self.pipeline.is_none() {
            self.pipeline = Some(self.create_pipeline(device));
        }
        let Some((pipeline, layout, sampler)) = &self.pipeline else { return };

        let key: Vec<usize> = views.iter().map(|v| std::ptr::from_ref(*v) as usize).collect();
        if self.bind_group.as_ref().is_none_or(|(k, _)| *k != key) {
            let resources = views
                .iter()
                .map(|view| wgpu::BindingResource::TextureView(view))
                .chain(uniforms.iter().map(|buffer| buffer.binding()))
                .chain([wgpu::BindingResource::Sampler(sampler)]);
            let entries: Vec<wgpu::BindGroupEntry> = resources
                .enumerate()
                .map(|(binding, resource)| wgpu::BindGroupEntry { binding: binding as u32, resource })
                .collect();
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(self.name),
                layout,
                entries: &entries,
            });
            self.bind_group = Some((key, bind_group));
        }
        let Some((_, bind_group)) = &self.bind_group else { return };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(self.name),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    store: wgpu::StoreOp::Store,
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
