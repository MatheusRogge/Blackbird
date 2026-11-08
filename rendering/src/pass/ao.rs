//! Screen-space ambient occlusion (`ao.wgsl`): GTAO over the G-buffer depth and normals into a raw
//! target, then a depth-aware 4×4 blur into the output `lighting` reads. With the
//! `ambient_occlusion` debug flag off the output is cleared to 1.

use std::sync::{Arc, RwLock};

use engine_core::world::World;

use crate::{
    debug_flags::RenderDebugFlags,
    graph::{NodeId, RenderGraph},
    pass::{Pass, PassContext, PassDesc},
    resource::{ResourceDescriptor, ResourceId},
    shader::ShaderAsset,
};

pub const AO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

struct Pipelines {
    ao: wgpu::RenderPipeline,
    ao_layout: wgpu::BindGroupLayout,
    blur: wgpu::RenderPipeline,
    blur_layout: wgpu::BindGroupLayout,
}

pub struct AoPass {
    node_id: Option<NodeId>,
    shader: ShaderAsset,
    depth_id: ResourceId,
    normal_id: ResourceId,
    camera_buffer_id: ResourceId,
    raw_id: ResourceId,
    output_id: ResourceId,
    debug_flags: Arc<RwLock<RenderDebugFlags>>,
    pipelines: Option<Pipelines>,
    /// Keyed by the input view addresses (screen textures move on resize).
    bind_groups: Option<(Vec<usize>, wgpu::BindGroup, wgpu::BindGroup)>,
}

impl AoPass {
    pub fn new(
        graph: &mut RenderGraph,
        depth_id: ResourceId,
        normal_id: ResourceId,
        camera_buffer_id: ResourceId,
        shader: ShaderAsset,
        debug_flags: Arc<RwLock<RenderDebugFlags>>,
    ) -> (Self, ResourceId) {
        let mut target = || {
            graph.alloc_resource_id(ResourceDescriptor::ScreenTexture {
                format: AO_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            })
        };
        let raw_id = target();
        let output_id = target();
        let pass = Self {
            node_id: None,
            shader,
            depth_id,
            normal_id,
            camera_buffer_id,
            raw_id,
            output_id,
            debug_flags,
            pipelines: None,
            bind_groups: None,
        };
        (pass, output_id)
    }

    fn create_pipelines(&self, device: &wgpu::Device) -> Pipelines {
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
        let mut entries = vec![
            texture(0, wgpu::TextureSampleType::Depth),
            texture(1, float),
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
        ];
        let ao_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("ao"), entries: &entries });
        entries.push(texture(3, float));
        let blur_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("ao_blur"), entries: &entries });

        let module = self.shader.compile(device).expect("ao shader compilation failed");
        let pipeline = |label, layout: &wgpu::BindGroupLayout, fs| {
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(fs),
                    targets: &[Some(AO_FORMAT.into())],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        Pipelines {
            ao: pipeline("ao", &ao_layout, "fs_ao"),
            blur: pipeline("ao_blur", &blur_layout, "fs_blur"),
            ao_layout,
            blur_layout,
        }
    }
}

fn fullscreen(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    target: &wgpu::TextureView,
    draw: Option<(&wgpu::RenderPipeline, &wgpu::BindGroup)>,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store },
        })],
        ..Default::default()
    });
    if let Some((pipeline, bind_group)) = draw {
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

impl PassDesc for AoPass {
    fn name(&self) -> &'static str {
        "ao"
    }

    fn reads(&self) -> Vec<ResourceId> {
        vec![self.depth_id, self.normal_id, self.camera_buffer_id]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![self.raw_id, self.output_id]
    }
}

impl Pass for AoPass {
    fn bind_node_id(&mut self, node_id: NodeId) {
        self.node_id = Some(node_id);
    }

    fn on_resize(&mut self, _width: u32, _height: u32) {
        self.bind_groups = None;
    }

    fn execute(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &PassContext<'_>,
        _world: &World,
    ) {
        let view = |id| ctx.views.get(id).copied();
        let (Some(output), Some(raw), Some(depth), Some(normal)) =
            (view(&self.output_id), view(&self.raw_id), view(&self.depth_id), view(&self.normal_id))
        else {
            return;
        };
        let Some(camera) = ctx.buffers.get(&self.camera_buffer_id) else { return };
        if !self.debug_flags.read().expect("debug_flags poisoned").ambient_occlusion {
            fullscreen(encoder, "ao_blur", output, None);
            return;
        }

        if self.pipelines.is_none() {
            self.pipelines = Some(self.create_pipelines(device));
        }
        let Some(pipelines) = &self.pipelines else { return };

        let key: Vec<usize> = [raw, depth, normal].iter().map(|v| std::ptr::from_ref(*v) as usize).collect();
        if self.bind_groups.as_ref().is_none_or(|(k, ..)| *k != key) {
            let mut entries = vec![
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(normal) },
                wgpu::BindGroupEntry { binding: 2, resource: camera.binding() },
            ];
            let ao = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ao"),
                layout: &pipelines.ao_layout,
                entries: &entries,
            });
            entries.push(wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(raw) });
            let blur = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ao_blur"),
                layout: &pipelines.blur_layout,
                entries: &entries,
            });
            self.bind_groups = Some((key, ao, blur));
        }
        let Some((_, ao_bg, blur_bg)) = &self.bind_groups else { return };
        fullscreen(encoder, "ao", raw, Some((&pipelines.ao, ao_bg)));
        fullscreen(encoder, "ao_blur", output, Some((&pipelines.blur, blur_bg)));
    }
}
