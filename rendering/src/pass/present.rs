use std::sync::Arc;

use engine::world::World;

use crate::{
    graph::NodeId,
    pass::{PassContext, RenderPass, RenderPassDesc},
    resource::ResourceId,
    shader::ShaderAsset,
};

pub struct PresentPass {
    node_id: Option<NodeId>,
    shader: Arc<ShaderAsset>,
    pipeline: Option<Arc<wgpu::RenderPipeline>>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    albedo_id: ResourceId,
    surface_id: ResourceId,
    surface_format: wgpu::TextureFormat,
}

impl PresentPass {
    pub fn new(
        albedo_id: ResourceId,
        surface_id: ResourceId,
        surface_format: wgpu::TextureFormat,
        shader: Arc<ShaderAsset>,
    ) -> Self {
        Self {
            node_id: None,
            shader,
            pipeline: None,
            bind_group_layout: None,
            albedo_id,
            surface_id,
            surface_format,
        }
    }
}

impl RenderPassDesc for PresentPass {
    fn name(&self) -> &'static str {
        "present"
    }

    fn reads(&self) -> Vec<ResourceId> {
        vec![self.albedo_id]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![self.surface_id]
    }
}

impl RenderPass for PresentPass {
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
        let surface_view = ctx.views[&self.surface_id];
        let albedo_view = ctx.views[&self.albedo_id];

        // Lazily create pipeline + layout (once)
        if self.pipeline.is_none() {
            let shader_module = self
                .shader
                .compile(device)
                .expect("present shader compilation failed");

            let layout_entries = [
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
            ];

            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("present_bind_group_layout"),
                    entries: &layout_entries,
                });

            let pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("present_pipeline_layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("present_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader_module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_module,
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

            self.pipeline = Some(Arc::new(pipeline));
            self.bind_group_layout = Some(bind_group_layout);
        }

        let pipeline = self.pipeline.as_ref().unwrap();
        let bgl = self.bind_group_layout.as_ref().unwrap();

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("present_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("present_bind_group"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("present"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: surface_view,
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
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
