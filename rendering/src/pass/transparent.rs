use std::sync::{Arc, RwLock};

use engine_core::world::World;

use crate::{
    arena::GeometryArena,
    camera::Camera,
    debug_flags::{GBufferView, RenderDebugFlags},
    graph::{NodeId, RenderGraph},
    mesh::{Mesh, Vertex},
    pass::cluster_assignment::ClusterOutputs,
    pass::gbuffer::GBufferOutputs,
    pass::light_upload::LightBuffers,
    pass::lighting::{LightingBindings, LightingOutputs},
    pass::material::{MaterialTextures, material_key},
    pass::probe_atlas::ProbeAtlasOutputs,
    pass::shadow::ShadowOutputs,
    pass::{Pass, PassContext, PassDesc},
    resource::{ResourceDescriptor, ResourceId},
    shader::ShaderAsset,
};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

struct Pipelines {
    background: wgpu::RenderPipeline,
    /// Build the `t_behind` mip chain: level 0 copies the scene, each further level downsamples
    /// the one before.
    scene_copy: wgpu::RenderPipeline,
    scene_downsample: wgpu::RenderPipeline,
    blit_layout: wgpu::BindGroupLayout,
    /// Clamped trilinear, for `t_behind` and the chain build.
    sampler: wgpu::Sampler,
    /// Back faces of double-sided meshes, drawn before their front faces.
    back: wgpu::RenderPipeline,
    front: wgpu::RenderPipeline,
    scene_layout: wgpu::BindGroupLayout,
    materials: MaterialTextures,
}

/// The opaque scene with a mip chain, read by transmissive surfaces: refraction samples it off the
/// fragment's own pixel, and rough surfaces sample a blurrier level.
struct Behind {
    size: (u32, u32),
    view: wgpu::TextureView,
    mips: Vec<wgpu::TextureView>,
    /// One per level: level 0 reads the scene, level `i` reads level `i - 1`. Keyed by the scene
    /// view address.
    blit_bgs: Option<(usize, Vec<wgpu::BindGroup>)>,
}

impl Behind {
    fn new(device: &wgpu::Device, (width, height): (u32, u32)) -> Self {
        let mip_level_count = width.max(height).max(1).ilog2() + 1;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("transparent_behind"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let mips = (0..mip_level_count)
            .map(|level| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        Self {
            size: (width, height),
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
            mips,
            blit_bgs: None,
        }
    }
}

/// Forward-shaded `AlphaMode::Blend` meshes over the composited opaque scene. Copies `scene` into
/// its own target, then draws the meshes back to front (by world-space bounds center) with
/// premultiplied alpha, depth-tested against the G-buffer depth without writing it. Shading uses
/// the `vs_forward` / `fs_forward` entry points of `lighting.wgsl` and the same light, shadow and
/// probe bindings as `lighting`. Skipped in G-buffer debug views.
pub struct TransparentPass {
    node_id: Option<NodeId>,
    shader: ShaderAsset,
    pipelines: Option<Pipelines>,
    /// Keyed by the scene view address and the geometry generation.
    scene_bg: Option<((usize, usize, u64), wgpu::BindGroup)>,
    behind: Option<Behind>,
    bindings: LightingBindings,
    debug_flags: Arc<RwLock<RenderDebugFlags>>,

    scene_id: ResourceId,
    depth_id: ResourceId,
    debug_flags_id: ResourceId,
    output_id: ResourceId,
}

impl TransparentPass {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        graph: &mut RenderGraph,
        scene_id: ResourceId,
        gbuffer: &GBufferOutputs,
        lighting: &LightingOutputs,
        cluster: &ClusterOutputs,
        light_buffers: &LightBuffers,
        shadow: &ShadowOutputs,
        probe_atlas: &ProbeAtlasOutputs,
        camera_buffer_id: ResourceId,
        lighting_shader: ShaderAsset,
        debug_flags: Arc<RwLock<RenderDebugFlags>>,
    ) -> (Self, ResourceId) {
        let output_id = graph.alloc_resource_id(ResourceDescriptor::ScreenTexture {
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        });
        let pass = Self {
            node_id: None,
            shader: lighting_shader,
            pipelines: None,
            scene_bg: None,
            behind: None,
            bindings: LightingBindings::new(
                cluster,
                light_buffers,
                shadow,
                probe_atlas,
                camera_buffer_id,
                lighting.debug_flags_id,
            ),
            debug_flags,
            scene_id,
            depth_id: gbuffer.depth_id,
            debug_flags_id: lighting.debug_flags_id,
            output_id,
        };
        (pass, output_id)
    }

    fn create_pipelines(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Pipelines {
        let module = self.shader.compile(device).expect("lighting shader compile failed");

        let scene_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("transparent_scene_bgl"),
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
                sampler_entry(4),
                color_entry(6),
                wgpu::BindGroupLayoutEntry {
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ..GeometryArena::instance_layout_entry(7)
                },
                color_entry(8),
            ],
        });
        let blit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("transparent_blit_bgl"),
            entries: &[sampler_entry(4), color_entry(10)],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("transparent_behind_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let materials = MaterialTextures::new(device, queue, "transparent_materials");
        let [cluster, shadow, probe, debug_flags] = self.bindings.layouts(device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("transparent_pipeline_layout"),
            bind_group_layouts: &[
                Some(&scene_layout),
                Some(cluster),
                Some(shadow),
                Some(probe),
                Some(debug_flags),
                Some(materials.layout()),
            ],
            immediate_size: 0,
        });

        let background = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("transparent_background"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_background"),
                targets: &[Some(FORMAT.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("transparent_blit_pipeline_layout"),
            bind_group_layouts: &[Some(&blit_layout)],
            immediate_size: 0,
        });
        let blit = |entry_point| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&blit_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry_point),
                    targets: &[Some(FORMAT.into())],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        let forward = |label, cull_mode| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_forward"),
                    buffers: &[Some(Vertex::buffer_descriptor())],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_forward"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: FORMAT,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState { cull_mode: Some(cull_mode), ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        Pipelines {
            background,
            scene_copy: blit("fs_scene_copy"),
            scene_downsample: blit("fs_scene_downsample"),
            blit_layout,
            sampler,
            back: forward("transparent_back_faces", wgpu::Face::Front),
            front: forward("transparent_front_faces", wgpu::Face::Back),
            scene_layout,
            materials,
        }
    }
}

fn color_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            multisampled: false,
            view_dimension: wgpu::TextureViewDimension::D2,
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

/// Fills `behind` from `scene`: a copy into level 0, then each level downsampled from the last.
fn build_behind_chain(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &Pipelines,
    behind: &mut Behind,
    scene_view: &wgpu::TextureView,
) {
    let scene_key = scene_view as *const wgpu::TextureView as usize;
    if behind.blit_bgs.as_ref().is_none_or(|(k, _)| *k != scene_key) {
        let sources = std::iter::once(scene_view).chain(&behind.mips);
        let bind_groups = sources
            .take(behind.mips.len())
            .map(|source| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("transparent_blit_bg"),
                    layout: &pipelines.blit_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&pipelines.sampler) },
                        wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(source) },
                    ],
                })
            })
            .collect();
        behind.blit_bgs = Some((scene_key, bind_groups));
    }
    let Some((_, bind_groups)) = behind.blit_bgs.as_ref() else { return };

    for (level, (target, bind_group)) in behind.mips.iter().zip(bind_groups).enumerate() {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("transparent_behind_mip"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        pass.set_pipeline(if level == 0 { &pipelines.scene_copy } else { &pipelines.scene_downsample });
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

impl PassDesc for TransparentPass {
    fn name(&self) -> &'static str {
        "transparent"
    }

    fn reads(&self) -> Vec<ResourceId> {
        let mut reads = vec![self.scene_id, self.depth_id, self.debug_flags_id];
        reads.extend(self.bindings.reads());
        reads
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![self.output_id]
    }
}

impl Pass for TransparentPass {
    fn bind_node_id(&mut self, node_id: NodeId) {
        self.node_id = Some(node_id);
    }

    fn on_resize(&mut self, _width: u32, _height: u32) {
        self.scene_bg = None;
        self.behind = None;
    }

    fn execute(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &PassContext<'_>,
        world: &World,
    ) {
        let (Some(&output_view), Some(&scene_view), Some(&depth_view)) = (
            ctx.views.get(&self.output_id),
            ctx.views.get(&self.scene_id),
            ctx.views.get(&self.depth_id),
        ) else {
            return;
        };

        if self.pipelines.is_none() {
            self.pipelines = Some(self.create_pipelines(device, queue));
        }

        let lit_view = self.debug_flags.read().expect("debug_flags poisoned").view == GBufferView::Lit;
        let (w, h) = ctx.surface_size;
        let camera = world.get_entities::<Camera>().first().map(|c| (c.eye, c.view_proj_matrix(w as f32 / h as f32)));
        let mut meshes: Vec<_> = world
            .get_entity_handles::<Mesh>()
            .into_iter()
            .filter(|h| lit_view && h.entity.is_blended())
            .filter_map(|h| {
                let draw = ctx.geometry.get(h.id)?;
                let (eye, view_proj) = camera?;
                if draw.bounds.outside_clip(view_proj * h.entity.transform) {
                    return None;
                }
                let bounds = draw.bounds.transformed(h.entity.transform);
                let distance = ((bounds.min + bounds.max) * 0.5 - eye).mag_sq();
                Some((distance, h, draw))
            })
            .collect();
        meshes.sort_by(|a, b| b.0.total_cmp(&a.0));

        let Some(pipelines) = self.pipelines.as_mut() else { return };
        pipelines.materials.prepare(device, queue, meshes.iter().map(|(_, h, _)| h.entity));

        if self.behind.as_ref().is_none_or(|b| b.size != ctx.surface_size) {
            self.behind = Some(Behind::new(device, ctx.surface_size));
            self.scene_bg = None;
        }
        let Some(behind) = self.behind.as_mut() else { return };
        if meshes.iter().any(|(_, h, _)| h.entity.material.transmission > 0.0) {
            build_behind_chain(device, encoder, pipelines, behind, scene_view);
        }

        let key = (
            scene_view as *const wgpu::TextureView as usize,
            depth_view as *const wgpu::TextureView as usize,
            ctx.geometry.generation(),
        );
        if self.scene_bg.as_ref().is_none_or(|(k, _)| *k != key) {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("transparent_scene_bg"),
                layout: &pipelines.scene_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(depth_view) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&pipelines.sampler) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(scene_view) },
                    wgpu::BindGroupEntry { binding: 7, resource: ctx.geometry.instance_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&behind.view) },
                ],
            });
            self.scene_bg = Some((key, bind_group));
        }
        let Some((_, scene_bg)) = self.scene_bg.as_ref() else { return };
        let Some(lighting_bgs) = self.bindings.bind_groups(device, ctx) else { return };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("transparent"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: None,
                stencil_ops: None,
            }),
            ..Default::default()
        });

        pass.set_bind_group(0, scene_bg, &[]);
        for (i, bind_group) in lighting_bgs.into_iter().enumerate() {
            pass.set_bind_group(i as u32 + 1, bind_group, &[]);
        }
        pass.set_pipeline(&pipelines.background);
        pass.set_bind_group(5, pipelines.materials.bind_group(Default::default()), &[]);
        pass.draw(0..3, 0..1);

        if meshes.is_empty() {
            return;
        }
        pass.set_vertex_buffer(0, ctx.geometry.vertex_buffer().slice(..));
        pass.set_index_buffer(ctx.geometry.index_buffer().slice(..), wgpu::IndexFormat::Uint32);
        for (_, handle, draw) in &meshes {
            let mesh = handle.entity;
            pass.set_bind_group(5, pipelines.materials.bind_group(material_key(mesh)), &[]);
            if mesh.double_sided {
                pass.set_pipeline(&pipelines.back);
                pass.draw_indexed(draw.indices.clone(), draw.base_vertex, draw.instances());
            }
            pass.set_pipeline(&pipelines.front);
            pass.draw_indexed(draw.indices.clone(), draw.base_vertex, draw.instances());
        }
    }
}
