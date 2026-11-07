use std::sync::{
    Arc, RwLock,
    atomic::{AtomicU32, Ordering},
};

use engine_core::world::World;

use crate::{
    debug_flags::{GBufferView, RenderDebugFlags},
    graph::{NodeId, RenderGraph},
    pass::cluster_assignment::ClusterOutputs,
    pass::gbuffer::GBufferOutputs,
    pass::light_upload::LightBuffers,
    pass::lighting::{LightingBindings, LightingOutputs},
    pass::particle_sim::ParticleOutputs,
    pass::probe_atlas::ProbeAtlasOutputs,
    pass::reflection_probe::ReflectionProbeOutputs,
    pass::shadow::ShadowOutputs,
    pass::{Pass, PassContext, PassDesc},
    resource::{ResourceDescriptor, ResourceId},
    shader::ShaderAsset,
};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Draws the particles `particle_sim` simulated over the finished HDR scene: copies `scene` into
/// its own target, then draws every live particle (an indirect draw over `particle_sim`'s draw
/// list, so expired slots cost nothing) as an additive camera-facing disc
/// (`lighting.wgsl`'s `vs_particle` / `fs_particle`), lit with the same bindings as `lighting`.
/// No depth attachment: the fragment shader fades particles out against the G-buffer depth itself
/// (soft particles). Particles draw over glass. Skipped in G-buffer debug views.
pub struct ParticlesPass {
    node_id: Option<NodeId>,
    shader: ShaderAsset,
    pipeline: Option<(wgpu::RenderPipeline, wgpu::BindGroupLayout)>,
    /// Keyed by the depth view address.
    particle_bg: Option<(usize, wgpu::BindGroup)>,
    bindings: LightingBindings,
    debug_flags: Arc<RwLock<RenderDebugFlags>>,
    slots_in_use: Arc<AtomicU32>,

    scene_id: ResourceId,
    depth_id: ResourceId,
    particles_id: ResourceId,
    params_id: ResourceId,
    state_id: ResourceId,
    draw_list_id: ResourceId,
    debug_flags_id: ResourceId,
    output_id: ResourceId,
}

impl ParticlesPass {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        graph: &mut RenderGraph,
        scene_id: ResourceId,
        particles: &ParticleOutputs,
        gbuffer: &GBufferOutputs,
        lighting: &LightingOutputs,
        cluster: &ClusterOutputs,
        light_buffers: &LightBuffers,
        shadow: &ShadowOutputs,
        probe_atlas: &ProbeAtlasOutputs,
        camera_buffer_id: ResourceId,
        reflection: &ReflectionProbeOutputs,
        lighting_shader: ShaderAsset,
        debug_flags: Arc<RwLock<RenderDebugFlags>>,
    ) -> (Self, ResourceId) {
        let output_id = graph.alloc_resource_id(ResourceDescriptor::ScreenTexture {
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
        });
        let pass = Self {
            node_id: None,
            shader: lighting_shader,
            pipeline: None,
            particle_bg: None,
            bindings: LightingBindings::new(
                cluster,
                light_buffers,
                shadow,
                probe_atlas,
                camera_buffer_id,
                lighting.debug_flags_id,
                reflection,
            ),
            debug_flags,
            slots_in_use: Arc::clone(&particles.slots_in_use),
            scene_id,
            depth_id: gbuffer.depth_id,
            particles_id: particles.particles_id,
            params_id: particles.params_id,
            state_id: particles.state_id,
            draw_list_id: particles.draw_list_id,
            debug_flags_id: lighting.debug_flags_id,
            output_id,
        };
        (pass, output_id)
    }

    fn create_pipeline(&mut self, device: &wgpu::Device) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
        let module = self.shader.compile(device).expect("lighting shader compile failed");
        let vertex_fragment = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
        let particle_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particles_bgl"),
            entries: &[
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
                    binding: 12,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: vertex_fragment,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let [cluster, shadow, probe, debug_flags] = self.bindings.layouts(device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particles_pipeline_layout"),
            bind_group_layouts: &[Some(&particle_layout), Some(cluster), Some(shadow), Some(probe), Some(debug_flags)],
            immediate_size: 0,
        });
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particles"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_particle"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_particle"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: FORMAT,
                    blend: Some(additive),
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
        (pipeline, particle_layout)
    }
}

impl PassDesc for ParticlesPass {
    fn name(&self) -> &'static str {
        "particles"
    }

    fn reads(&self) -> Vec<ResourceId> {
        let mut reads = vec![
            self.scene_id,
            self.depth_id,
            self.particles_id,
            self.params_id,
            self.state_id,
            self.draw_list_id,
            self.debug_flags_id,
        ];
        reads.extend(self.bindings.reads());
        reads
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![self.output_id]
    }
}

impl Pass for ParticlesPass {
    fn bind_node_id(&mut self, node_id: NodeId) {
        self.node_id = Some(node_id);
    }

    fn on_resize(&mut self, _width: u32, _height: u32) {
        self.particle_bg = None;
    }

    fn execute(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &PassContext<'_>,
        _world: &World,
    ) {
        let (Some(&scene), Some(&output), Some(&output_view), Some(&depth_view)) = (
            ctx.textures.get(&self.scene_id),
            ctx.textures.get(&self.output_id),
            ctx.views.get(&self.output_id),
            ctx.views.get(&self.depth_id),
        ) else {
            return;
        };
        let (w, h) = ctx.surface_size;
        encoder.copy_texture_to_texture(
            scene.as_image_copy(),
            output.as_image_copy(),
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );

        let lit_view = self.debug_flags.read().expect("debug_flags poisoned").view == GBufferView::Lit;
        if self.slots_in_use.load(Ordering::Relaxed) == 0 || !lit_view {
            return;
        }
        let buffer = |id| ctx.buffers.get(id).copied();
        let (Some(particles), Some(params), Some(state), Some(draw_list)) = (
            buffer(&self.particles_id),
            buffer(&self.params_id),
            buffer(&self.state_id),
            buffer(&self.draw_list_id),
        ) else {
            return;
        };

        if self.pipeline.is_none() {
            self.pipeline = Some(self.create_pipeline(device));
        }
        let Some((pipeline, particle_layout)) = &self.pipeline else { return };
        let depth_key = depth_view as *const wgpu::TextureView as usize;
        if self.particle_bg.as_ref().is_none_or(|(key, _)| *key != depth_key) {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("particles_bg"),
                layout: particle_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(depth_view) },
                    wgpu::BindGroupEntry { binding: 12, resource: particles.binding() },
                    wgpu::BindGroupEntry { binding: 13, resource: params.binding() },
                    wgpu::BindGroupEntry { binding: 14, resource: draw_list.binding() },
                ],
            });
            self.particle_bg = Some((depth_key, bind_group));
        }
        let Some((_, particle_bg)) = &self.particle_bg else { return };
        let Some(lighting_bgs) = self.bindings.bind_groups(device, ctx) else { return };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("particles"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, particle_bg, &[]);
        for (i, bind_group) in lighting_bgs.into_iter().enumerate() {
            pass.set_bind_group(i as u32 + 1, bind_group, &[]);
        }
        pass.draw_indirect(state.buffer, state.offset);
    }
}
