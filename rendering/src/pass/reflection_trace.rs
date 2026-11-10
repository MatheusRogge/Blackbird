use engine_core::world::World;

use crate::{
    graph::{NodeId, RenderGraph},
    pass::bvh_upload::BvhOutputs,
    pass::cluster_assignment::ClusterOutputs,
    pass::gbuffer::GBufferOutputs,
    pass::light_upload::LightBuffers,
    pass::lighting::{LightingBindings, LightingOutputs},
    pass::probe_atlas::ProbeAtlasOutputs,
    pass::reflection_probe::ReflectionProbeOutputs,
    pass::shadow::ShadowOutputs,
    pass::{Pass, PassContext, PassDesc},
    resource::{ResourceDescriptor, ResourceId},
    shader::ShaderAsset,
};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Matches `ssr`'s ray length target, which the resolve stands in for.
const RAY_LENGTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;

/// Inputs from earlier passes.
pub struct ReflectionTraceInputs<'a> {
    pub gbuffer: &'a GBufferOutputs,
    pub lighting: &'a LightingOutputs,
    pub bvh: &'a BvhOutputs,
    /// `ssr`'s colour (pixels it hit aren't traced) and ray length.
    pub ssr_id: ResourceId,
    pub ssr_ray_length_id: ResourceId,
    pub cluster: &'a ClusterOutputs,
    pub light_buffers: &'a LightBuffers,
    pub shadow: &'a ShadowOutputs,
    pub probe_atlas: &'a ProbeAtlasOutputs,
    pub camera_buffer_id: ResourceId,
    pub reflection: &'a ReflectionProbeOutputs,
}

/// Traces the reflection ray of pixels SSR missed through the GI's BVH (`lighting.wgsl`'s
/// `fs_reflection_trace` / `fs_reflection_trace_rough`, a software traversal: no ray-tracing
/// hardware), one pixel per 2×2 block on glossy surfaces and per 4×4 on rough ones, so reflections of things off screen keep their place and sharpness
/// instead of falling back to the camera-centred probe. `fs_reflection_resolve` then upsamples
/// the traces into SSR's outputs where SSR missed, so `ssr_temporal` accumulates them and
/// `composite` blurs them like SSR hits.
pub struct ReflectionTracePass {
    node_id: Option<NodeId>,
    shader: ShaderAsset,
    pipelines: Option<Pipelines>,
    /// The half-resolution (glossy) and quarter-resolution (rough) traces, for this surface size.
    traced: Option<((u32, u32), wgpu::TextureView, wgpu::TextureView)>,
    /// Keyed by the addresses of every view and buffer in them.
    trace_bg: Option<([usize; 11], wgpu::BindGroup)>,
    resolve_bg: Option<([usize; 7], wgpu::BindGroup)>,
    bindings: LightingBindings,

    normal_id: ResourceId,
    depth_id: ResourceId,
    lit_id: ResourceId,
    specular_weight_id: ResourceId,
    ssr_id: ResourceId,
    ssr_ray_length_id: ResourceId,
    bvh_nodes_id: ResourceId,
    bvh_tris_id: ResourceId,
    bvh_info_id: ResourceId,
    debug_flags_id: ResourceId,
    output_id: ResourceId,
    ray_length_id: ResourceId,
}

impl ReflectionTracePass {
    /// Returns the pass and its colour and ray length outputs, in place of `ssr`'s.
    pub fn new(
        graph: &mut RenderGraph,
        inputs: ReflectionTraceInputs<'_>,
        lighting_shader: ShaderAsset,
    ) -> (Self, ResourceId, ResourceId) {
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let output_id = graph.alloc_resource_id(ResourceDescriptor::ScreenTexture { format: FORMAT, usage });
        let ray_length_id = graph.alloc_resource_id(ResourceDescriptor::ScreenTexture { format: RAY_LENGTH_FORMAT, usage });
        let pass = Self {
            node_id: None,
            shader: lighting_shader,
            pipelines: None,
            traced: None,
            trace_bg: None,
            resolve_bg: None,
            bindings: LightingBindings::new(
                inputs.cluster,
                inputs.light_buffers,
                inputs.shadow,
                inputs.probe_atlas,
                inputs.camera_buffer_id,
                inputs.lighting.debug_flags_id,
                inputs.reflection,
            ),
            normal_id: inputs.gbuffer.normal_id,
            depth_id: inputs.gbuffer.depth_id,
            lit_id: inputs.lighting.lit_id,
            specular_weight_id: inputs.lighting.specular_weight_id,
            ssr_id: inputs.ssr_id,
            ssr_ray_length_id: inputs.ssr_ray_length_id,
            bvh_nodes_id: inputs.bvh.nodes_id,
            bvh_tris_id: inputs.bvh.tris_id,
            bvh_info_id: inputs.bvh.info_id,
            debug_flags_id: inputs.lighting.debug_flags_id,
            output_id,
            ray_length_id,
        };
        (pass, output_id, ray_length_id)
    }

    fn create_pipelines(&mut self, device: &wgpu::Device) -> Pipelines {
        let module = self.shader.compile(device).expect("lighting shader compile failed");
        let float = wgpu::TextureSampleType::Float { filterable: true };
        let trace_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reflection_trace_bgl"),
            entries: &[
                texture_entry(1, float),
                texture_entry(3, wgpu::TextureSampleType::Depth),
                texture_entry(20, float),
                texture_entry(21, float),
                buffer_entry(22, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(23, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(24, wgpu::BufferBindingType::Uniform),
                texture_entry(25, float),
            ],
        });
        let resolve_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reflection_resolve_bgl"),
            entries: &[
                texture_entry(1, float),
                texture_entry(3, wgpu::TextureSampleType::Depth),
                texture_entry(21, float),
                texture_entry(25, float),
                texture_entry(27, float),
                texture_entry(28, wgpu::TextureSampleType::Float { filterable: false }),
                texture_entry(29, float),
            ],
        });
        let [cluster, shadow, probe, debug_flags] = self.bindings.layouts(device);
        let pipeline = |label: &str, scene_layout: &wgpu::BindGroupLayout, entry_point: &str, formats: &[wgpu::TextureFormat]| {
            let targets: Vec<_> = formats.iter().map(|&format| Some(format.into())).collect();
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(scene_layout), Some(cluster), Some(shadow), Some(probe), Some(debug_flags)],
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry_point),
                    targets: &targets,
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
            trace: pipeline("reflection_trace", &trace_layout, "fs_reflection_trace", &[FORMAT]),
            trace_rough: pipeline("reflection_trace_rough", &trace_layout, "fs_reflection_trace_rough", &[FORMAT]),
            resolve: pipeline("reflection_resolve", &resolve_layout, "fs_reflection_resolve", &[FORMAT, RAY_LENGTH_FORMAT]),
            trace_layout,
            resolve_layout,
        }
    }
}

struct Pipelines {
    trace: wgpu::RenderPipeline,
    trace_rough: wgpu::RenderPipeline,
    resolve: wgpu::RenderPipeline,
    trace_layout: wgpu::BindGroupLayout,
    resolve_layout: wgpu::BindGroupLayout,
}

fn texture_entry(binding: u32, sample_type: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            multisampled: false,
            view_dimension: wgpu::TextureViewDimension::D2,
            sample_type,
        },
        count: None,
    }
}

fn buffer_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

impl PassDesc for ReflectionTracePass {
    fn name(&self) -> &'static str {
        "reflection_trace"
    }

    fn reads(&self) -> Vec<ResourceId> {
        let mut reads = vec![
            self.normal_id,
            self.depth_id,
            self.lit_id,
            self.specular_weight_id,
            self.ssr_id,
            self.ssr_ray_length_id,
            self.bvh_nodes_id,
            self.bvh_tris_id,
            self.bvh_info_id,
            self.debug_flags_id,
        ];
        reads.extend(self.bindings.reads());
        reads
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![self.output_id, self.ray_length_id]
    }
}

impl Pass for ReflectionTracePass {
    fn bind_node_id(&mut self, node_id: NodeId) {
        self.node_id = Some(node_id);
    }

    fn on_resize(&mut self, _width: u32, _height: u32) {
        self.traced = None;
        self.trace_bg = None;
        self.resolve_bg = None;
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
        let buffer = |id| ctx.buffers.get(id).copied();
        let (
            Some(output),
            Some(ray_length),
            Some(normal),
            Some(depth),
            Some(lit),
            Some(specular_weight),
            Some(ssr),
            Some(ssr_ray_length),
        ) = (
            view(&self.output_id),
            view(&self.ray_length_id),
            view(&self.normal_id),
            view(&self.depth_id),
            view(&self.lit_id),
            view(&self.specular_weight_id),
            view(&self.ssr_id),
            view(&self.ssr_ray_length_id),
        )
        else {
            return;
        };
        let (Some(nodes), Some(tris), Some(info)) =
            (buffer(&self.bvh_nodes_id), buffer(&self.bvh_tris_id), buffer(&self.bvh_info_id))
        else {
            return;
        };
        if self.pipelines.is_none() {
            self.pipelines = Some(self.create_pipelines(device));
        }
        let Some(lighting_bgs) = self.bindings.bind_groups(device, ctx) else { return };
        let Some(pipelines) = &self.pipelines else { return };

        let size = ctx.surface_size;
        if self.traced.as_ref().is_none_or(|(s, ..)| *s != size) {
            let target = |label, divisor: u32| {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: size.0.div_ceil(divisor).max(1),
                        height: size.1.div_ceil(divisor).max(1),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                texture.create_view(&wgpu::TextureViewDescriptor::default())
            };
            self.traced = Some((size, target("reflection_traced", 2), target("reflection_traced_rough", 4)));
            self.resolve_bg = None;
        }
        let Some((_, traced, traced_rough)) = &self.traced else { return };

        let address = |v: &wgpu::TextureView| v as *const wgpu::TextureView as usize;
        let texture = |binding, view| wgpu::BindGroupEntry { binding, resource: wgpu::BindingResource::TextureView(view) };
        let trace_key = [
            address(normal),
            address(depth),
            address(lit),
            address(specular_weight),
            address(ssr),
            nodes.buffer as *const wgpu::Buffer as usize,
            tris.buffer as *const wgpu::Buffer as usize,
            info.buffer as *const wgpu::Buffer as usize,
            nodes.offset as usize,
            tris.offset as usize,
            info.offset as usize,
        ];
        if self.trace_bg.as_ref().is_none_or(|(k, _)| *k != trace_key) {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("reflection_trace_bg"),
                layout: &pipelines.trace_layout,
                entries: &[
                    texture(1, normal),
                    texture(3, depth),
                    texture(20, lit),
                    texture(21, specular_weight),
                    wgpu::BindGroupEntry { binding: 22, resource: nodes.binding() },
                    wgpu::BindGroupEntry { binding: 23, resource: tris.binding() },
                    wgpu::BindGroupEntry { binding: 24, resource: info.binding() },
                    texture(25, ssr),
                ],
            });
            self.trace_bg = Some((trace_key, bind_group));
        }
        let resolve_key = [
            address(normal),
            address(depth),
            address(specular_weight),
            address(ssr),
            address(traced),
            address(ssr_ray_length),
            address(traced_rough),
        ];
        if self.resolve_bg.as_ref().is_none_or(|(k, _)| *k != resolve_key) {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("reflection_resolve_bg"),
                layout: &pipelines.resolve_layout,
                entries: &[
                    texture(1, normal),
                    texture(3, depth),
                    texture(21, specular_weight),
                    texture(25, ssr),
                    texture(27, traced),
                    texture(28, ssr_ray_length),
                    texture(29, traced_rough),
                ],
            });
            self.resolve_bg = Some((resolve_key, bind_group));
        }
        let (Some((_, trace_bg)), Some((_, resolve_bg))) = (&self.trace_bg, &self.resolve_bg) else { return };

        let attachment = |view, load| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
            })
        };
        let clear = wgpu::LoadOp::Clear(wgpu::Color::BLACK);
        for (targets, pipeline, scene_bg, label) in [
            (vec![attachment(traced, clear)], &pipelines.trace, trace_bg, "reflection_trace"),
            (vec![attachment(traced_rough, clear)], &pipelines.trace_rough, trace_bg, "reflection_trace_rough"),
            (vec![attachment(output, clear), attachment(ray_length, clear)], &pipelines.resolve, resolve_bg, "reflection_resolve"),
        ] {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &targets,
                ..Default::default()
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, scene_bg, &[]);
            for (i, bind_group) in lighting_bgs.iter().enumerate() {
                pass.set_bind_group(i as u32 + 1, *bind_group, &[]);
            }
            pass.draw(0..3, 0..1);
        }
    }
}
