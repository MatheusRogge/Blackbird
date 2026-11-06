//! Screen-space reflections: builds a Hi-Z pyramid of the G-buffer depth (`hiz.wgsl`, embedded),
//! then traces each reflection ray through it (`ssr.wgsl`). Writes premultiplied hit colour with
//! confidence in alpha, which `ssr_temporal` accumulates and `composite` blends.

use engine_core::world::World;

use crate::{
    graph::{NodeId, RenderGraph},
    pass::{Pass, PassContext, PassDesc},
    resource::{ResourceDescriptor, ResourceId},
    shader::ShaderAsset,
};

const HIZ_SHADER: &str = include_str!("hiz.wgsl");
const HIZ_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
const OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub struct SsrInputs {
    pub depth_id: ResourceId,
    pub normal_id: ResourceId,
    pub lit_id: ResourceId,
    pub specular_weight_id: ResourceId,
    pub camera_buffer_id: ResourceId,
    /// `lighting`'s debug flags uniform (`disable_ssr`).
    pub debug_flags_id: ResourceId,
}

struct Pipelines {
    trace: wgpu::RenderPipeline,
    trace_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    hiz_copy: wgpu::RenderPipeline,
    hiz_reduce: wgpu::RenderPipeline,
    /// Binding 0: the G-buffer depth.
    hiz_copy_layout: wgpu::BindGroupLayout,
    /// Binding 1: the level above.
    hiz_reduce_layout: wgpu::BindGroupLayout,
}

/// The pyramid, sized to the surface. Pass-owned: graph textures have a single mip.
struct Hiz {
    size: (u32, u32),
    /// Whole chain, for the trace.
    view: wgpu::TextureView,
    /// One view per level: render target, and source of the next level.
    levels: Vec<wgpu::TextureView>,
    /// Entry `i` reads level `i` to build level `i + 1`.
    reduce_bgs: Vec<wgpu::BindGroup>,
    /// Keyed by the depth view address.
    copy_bg: Option<(usize, wgpu::BindGroup)>,
}

pub struct SsrPass {
    node_id: Option<NodeId>,
    shader: ShaderAsset,
    inputs: SsrInputs,
    output_id: ResourceId,
    pipelines: Option<Pipelines>,
    hiz: Option<Hiz>,
    /// Keyed by the input view addresses (screen textures move on resize).
    trace_bg: Option<(Vec<usize>, wgpu::BindGroup)>,
}

impl SsrPass {
    pub fn new(graph: &mut RenderGraph, inputs: SsrInputs, shader: ShaderAsset) -> (Self, ResourceId) {
        let output_id = graph.alloc_resource_id(ResourceDescriptor::ScreenTexture {
            format: OUTPUT_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        });
        let pass = Self {
            node_id: None,
            shader,
            inputs,
            output_id,
            pipelines: None,
            hiz: None,
            trace_bg: None,
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
        let uniform = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let unfilterable = wgpu::TextureSampleType::Float { filterable: false };
        let float = wgpu::TextureSampleType::Float { filterable: true };

        let pipeline = |label, module: &wgpu::ShaderModule, layout: &wgpu::BindGroupLayout, fs, format: wgpu::TextureFormat| {
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some(fs),
                    targets: &[Some(format.into())],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        let trace_module = self.shader.compile(device).expect("ssr shader compilation failed");
        let trace_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssr"),
            entries: &[
                texture(0, unfilterable),
                texture(1, float),
                texture(2, float),
                texture(3, float),
                uniform(4),
                uniform(5),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let hiz_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hiz"),
            source: wgpu::ShaderSource::Wgsl(HIZ_SHADER.into()),
        });
        let hiz_copy_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hiz_copy"),
            entries: &[texture(0, wgpu::TextureSampleType::Depth)],
        });
        let hiz_reduce_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hiz_reduce"),
            entries: &[texture(1, unfilterable)],
        });
        Pipelines {
            trace: pipeline("ssr", &trace_module, &trace_layout, "fs_main", OUTPUT_FORMAT),
            trace_layout,
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("ssr"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            hiz_copy: pipeline("hiz_copy", &hiz_module, &hiz_copy_layout, "fs_copy", HIZ_FORMAT),
            hiz_reduce: pipeline("hiz_reduce", &hiz_module, &hiz_reduce_layout, "fs_reduce", HIZ_FORMAT),
            hiz_copy_layout,
            hiz_reduce_layout,
        }
    }

    fn create_hiz(device: &wgpu::Device, pipelines: &Pipelines, (width, height): (u32, u32)) -> Hiz {
        let level_count = width.max(height).max(1).ilog2() + 1;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hiz"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HIZ_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let levels: Vec<_> = (0..level_count)
            .map(|level| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let reduce_bgs = levels
            .iter()
            .take(levels.len() - 1)
            .map(|source| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("hiz_reduce"),
                    layout: &pipelines.hiz_reduce_layout,
                    entries: &[wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(source) }],
                })
            })
            .collect();
        Hiz {
            size: (width, height),
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
            levels,
            reduce_bgs,
            copy_bg: None,
        }
    }
}

fn fullscreen(encoder: &mut wgpu::CommandEncoder, label: &str, target: &wgpu::TextureView, pipeline: &wgpu::RenderPipeline, bind_group: &wgpu::BindGroup) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
        })],
        ..Default::default()
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}

impl PassDesc for SsrPass {
    fn name(&self) -> &'static str {
        "ssr"
    }

    fn reads(&self) -> Vec<ResourceId> {
        let i = &self.inputs;
        vec![i.depth_id, i.normal_id, i.lit_id, i.specular_weight_id, i.camera_buffer_id, i.debug_flags_id]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![self.output_id]
    }
}

impl Pass for SsrPass {
    fn bind_node_id(&mut self, node_id: NodeId) {
        self.node_id = Some(node_id);
    }

    fn on_resize(&mut self, _width: u32, _height: u32) {
        self.hiz = None;
        self.trace_bg = None;
    }

    fn execute(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &PassContext<'_>,
        _world: &World,
    ) {
        let i = &self.inputs;
        let view = |id| ctx.views.get(id).copied();
        let (Some(output), Some(depth), Some(normal), Some(lit), Some(weight)) =
            (view(&self.output_id), view(&i.depth_id), view(&i.normal_id), view(&i.lit_id), view(&i.specular_weight_id))
        else {
            return;
        };
        let (Some(camera), Some(flags)) = (ctx.buffers.get(&i.camera_buffer_id), ctx.buffers.get(&i.debug_flags_id)) else {
            return;
        };
        if self.pipelines.is_none() {
            self.pipelines = Some(self.create_pipelines(device));
        }
        let Some(pipelines) = &self.pipelines else { return };
        if self.hiz.as_ref().is_none_or(|h| h.size != ctx.surface_size) {
            self.hiz = Some(Self::create_hiz(device, pipelines, ctx.surface_size));
            self.trace_bg = None;
        }
        let Some(hiz) = self.hiz.as_mut() else { return };

        let depth_key = std::ptr::from_ref(depth) as usize;
        if hiz.copy_bg.as_ref().is_none_or(|(k, _)| *k != depth_key) {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hiz_copy"),
                layout: &pipelines.hiz_copy_layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth) }],
            });
            hiz.copy_bg = Some((depth_key, bind_group));
        }
        let Some((_, copy_bg)) = &hiz.copy_bg else { return };
        fullscreen(encoder, "hiz_copy", &hiz.levels[0], &pipelines.hiz_copy, copy_bg);
        for (level, bind_group) in hiz.reduce_bgs.iter().enumerate() {
            fullscreen(encoder, "hiz_reduce", &hiz.levels[level + 1], &pipelines.hiz_reduce, bind_group);
        }

        let key: Vec<usize> = [normal, lit, weight].iter().map(|v| std::ptr::from_ref(*v) as usize).collect();
        if self.trace_bg.as_ref().is_none_or(|(k, _)| *k != key) {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ssr"),
                layout: &pipelines.trace_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&hiz.view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(normal) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(lit) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(weight) },
                    wgpu::BindGroupEntry { binding: 4, resource: camera.binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: flags.binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&pipelines.sampler) },
                ],
            });
            self.trace_bg = Some((key, bind_group));
        }
        let Some((_, trace_bg)) = &self.trace_bg else { return };
        fullscreen(encoder, "ssr", output, &pipelines.trace, trace_bg);
    }
}
