//! Screen-space ambient occlusion (`ao.wgsl`): draws the nearest visible glass into its own depth
//! and transmittance targets (`Blend` meshes aren't in the G-buffer), then GTAO over the G-buffer
//! depth, normals and that glass into a half-resolution raw target, then a depth-aware 4×4 blur
//! that upsamples it into the output `lighting` reads. With the `ambient_occlusion` debug flag off the output is cleared to 1. The
//! glass targets are also outputs: contact shadows in `lighting` march them and `ssr` reflects them,
//! so they are drawn while any of the three is on.

use std::sync::{Arc, RwLock};

use engine_core::world::World;

use crate::{
    arena::GeometryArena,
    camera::Camera,
    debug_flags::RenderDebugFlags,
    mesh::{Mesh, Vertex},
    graph::{NodeId, RenderGraph},
    pass::{Pass, PassContext, PassDesc},
    resource::{ResourceDescriptor, ResourceId},
    shader::ShaderAsset,
};

pub const AO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
const GLASS_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const GLASS_TRANSMITTANCE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
pub const CONTACT_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Float;

#[derive(Clone, Copy)]
pub struct AoOutputs {
    pub ao_id: ResourceId,
    /// Nearest glass in front of the opaque depth (reversed-Z, cleared to 0).
    pub glass_depth_id: ResourceId,
    /// That glass's transmittance in rgb, cleared to 1.
    pub glass_transmittance_id: ResourceId,
    /// `CONTACT_DEPTH_FORMAT`: linear depth of the opaque surface (r) and of that glass (g), 0 for
    /// none, so contact shadows march both with one load.
    pub contact_depth_id: ResourceId,
}

struct Pipelines {
    contact_depth: wgpu::RenderPipeline,
    glass: wgpu::RenderPipeline,
    glass_layout: wgpu::BindGroupLayout,
    ao: wgpu::RenderPipeline,
    /// `fs_ao` with `HAS_GLASS` off, for frames with no visible glass.
    ao_no_glass: wgpu::RenderPipeline,
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
    /// Half-resolution GTAO, recreated on resize.
    raw: Option<wgpu::TextureView>,
    output_id: ResourceId,
    glass_depth_id: ResourceId,
    glass_transmittance_id: ResourceId,
    contact_depth_id: ResourceId,
    debug_flags: Arc<RwLock<RenderDebugFlags>>,
    pipelines: Option<Pipelines>,
    /// Keyed by the input view addresses (screen textures move on resize).
    bind_groups: Option<(Vec<usize>, wgpu::BindGroup, wgpu::BindGroup)>,
    /// Keyed by the depth view address and the geometry generation (instance buffer).
    glass_bg: Option<((usize, u64), wgpu::BindGroup)>,
}

impl AoPass {
    pub fn new(
        graph: &mut RenderGraph,
        depth_id: ResourceId,
        normal_id: ResourceId,
        camera_buffer_id: ResourceId,
        shader: ShaderAsset,
        debug_flags: Arc<RwLock<RenderDebugFlags>>,
    ) -> (Self, AoOutputs) {
        let mut target = |format| {
            graph.alloc_resource_id(ResourceDescriptor::ScreenTexture {
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            })
        };
        let output_id = target(AO_FORMAT);
        let glass_depth_id = target(GLASS_DEPTH_FORMAT);
        let glass_transmittance_id = target(GLASS_TRANSMITTANCE_FORMAT);
        let contact_depth_id = target(CONTACT_DEPTH_FORMAT);
        let pass = Self {
            node_id: None,
            shader,
            depth_id,
            normal_id,
            camera_buffer_id,
            raw: None,
            output_id,
            glass_depth_id,
            glass_transmittance_id,
            contact_depth_id,
            debug_flags,
            pipelines: None,
            bind_groups: None,
            glass_bg: None,
        };
        (pass, AoOutputs { ao_id: output_id, glass_depth_id, glass_transmittance_id, contact_depth_id })
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
        let camera = wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let glass_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ao_glass"),
            entries: &[texture(0, wgpu::TextureSampleType::Depth), camera, GeometryArena::instance_layout_entry(4)],
        });
        let mut entries = vec![
            texture(0, wgpu::TextureSampleType::Depth),
            texture(1, float),
            camera,
            texture(6, float),
            texture(7, wgpu::TextureSampleType::Float { filterable: false }),
        ];
        let ao_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("ao"), entries: &entries });
        entries.push(texture(3, float));
        let blur_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("ao_blur"), entries: &entries });

        let module = self.shader.compile(device).expect("ao shader compilation failed");
        let pipeline = |label, layout: &wgpu::BindGroupLayout, fs, constants: &[(&str, f64)]| {
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
                    compilation_options: wgpu::PipelineCompilationOptions { constants, ..Default::default() },
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let glass_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ao_glass"),
            bind_group_layouts: &[Some(&glass_layout)],
            immediate_size: 0,
        });
        // Both draw into the glass pass's targets: transmittance, then contact depth.
        let glass_pass_pipeline = |label, vs, fs, buffers: &[Option<wgpu::VertexBufferLayout<'_>>], masks: [wgpu::ColorWrites; 2], depth: (bool, wgpu::CompareFunction)| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&glass_pipeline_layout),
                vertex: wgpu::VertexState { module: &module, entry_point: Some(vs), buffers, compilation_options: Default::default() },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(fs),
                    targets: &[
                        Some(wgpu::ColorTargetState { format: GLASS_TRANSMITTANCE_FORMAT, blend: None, write_mask: masks[0] }),
                        Some(wgpu::ColorTargetState { format: CONTACT_DEPTH_FORMAT, blend: None, write_mask: masks[1] }),
                    ],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: GLASS_DEPTH_FORMAT,
                    depth_write_enabled: Some(depth.0),
                    depth_compare: Some(depth.1),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let contact_depth = glass_pass_pipeline(
            "ao_contact_depth",
            "vs_main",
            "fs_contact_depth",
            &[],
            [wgpu::ColorWrites::empty(), wgpu::ColorWrites::ALL],
            (false, wgpu::CompareFunction::Always),
        );
        // The depth test keeps the nearest glass, so the last write to a texel is its depth.
        let glass = glass_pass_pipeline(
            "ao_glass",
            "vs_glass",
            "fs_glass",
            &[Some(Vertex::buffer_descriptor())],
            [wgpu::ColorWrites::ALL, wgpu::ColorWrites::GREEN],
            (true, wgpu::CompareFunction::GreaterEqual),
        );
        Pipelines {
            contact_depth,
            glass,
            glass_layout,
            ao: pipeline("ao", &ao_layout, "fs_ao", &[]),
            ao_no_glass: pipeline("ao_no_glass", &ao_layout, "fs_ao", &[("HAS_GLASS", 0.0)]),
            blur: pipeline("ao_blur", &blur_layout, "fs_blur", &[]),
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
        vec![self.output_id, self.glass_depth_id, self.glass_transmittance_id, self.contact_depth_id]
    }
}

impl Pass for AoPass {
    fn bind_node_id(&mut self, node_id: NodeId) {
        self.node_id = Some(node_id);
    }

    fn on_resize(&mut self, _width: u32, _height: u32) {
        self.raw = None;
        self.bind_groups = None;
        self.glass_bg = None;
    }

    fn execute(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &PassContext<'_>,
        world: &World,
    ) {
        let view = |id| ctx.views.get(id).copied();
        let (
            Some(output),
            Some(depth),
            Some(normal),
            Some(glass_depth),
            Some(glass_transmittance),
            Some(contact_depth),
        ) = (
            view(&self.output_id),
            view(&self.depth_id),
            view(&self.normal_id),
            view(&self.glass_depth_id),
            view(&self.glass_transmittance_id),
            view(&self.contact_depth_id),
        )
        else {
            return;
        };
        let Some(camera) = ctx.buffers.get(&self.camera_buffer_id) else { return };
        let (ambient_occlusion, draw_glass) = {
            let flags = self.debug_flags.read().expect("debug_flags poisoned");
            let contact_shadows = flags.contact_shadows && flags.shadows;
            let ssr = flags.ssr && flags.reflections && flags.specular && flags.indirect;
            (flags.ambient_occlusion, flags.ambient_occlusion || contact_shadows || ssr)
        };

        if self.pipelines.is_none() {
            self.pipelines = Some(self.create_pipelines(device));
        }
        let Some(pipelines) = &self.pipelines else { return };
        let (w, h) = ctx.surface_size;
        let raw = self.raw.get_or_insert_with(|| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("ao_raw"),
                    size: wgpu::Extent3d { width: w.div_ceil(2), height: h.div_ceil(2), depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: AO_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        });

        let glass_key = (std::ptr::from_ref(depth) as usize, ctx.geometry.generation());
        if self.glass_bg.as_ref().is_none_or(|(k, _)| *k != glass_key) {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ao_glass"),
                layout: &pipelines.glass_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth) },
                    wgpu::BindGroupEntry { binding: 2, resource: camera.binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: ctx.geometry.instance_buffer().as_entire_binding() },
                ],
            });
            self.glass_bg = Some((glass_key, bind_group));
        }

        let key: Vec<usize> = [&*raw, depth, normal, glass_transmittance, contact_depth]
            .iter()
            .map(|v| std::ptr::from_ref(*v) as usize)
            .collect();
        if self.bind_groups.as_ref().is_none_or(|(k, ..)| *k != key) {
            let mut entries = vec![
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(normal) },
                wgpu::BindGroupEntry { binding: 2, resource: camera.binding() },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(glass_transmittance) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(contact_depth) },
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
        let Some((_, glass_bg)) = &self.glass_bg else { return };

        let view_proj = world.get_entities::<Camera>().first().map(|c| c.view_proj_matrix(w as f32 / h as f32));
        let glass: Vec<_> = world
            .get_entity_handles::<Mesh>()
            .into_iter()
            .filter(|h| draw_glass && h.entity.is_blended())
            .filter_map(|h| {
                let draw = ctx.geometry.get(h.id)?;
                (!draw.bounds.outside_clip(view_proj? * h.entity.transform)).then_some(draw)
            })
            .collect();
        let ao_pipeline = if glass.is_empty() { &pipelines.ao_no_glass } else { &pipelines.ao };
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ao_glass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: glass_transmittance,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: contact_depth,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: glass_depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_bind_group(0, glass_bg, &[]);
            pass.set_pipeline(&pipelines.contact_depth);
            pass.draw(0..3, 0..1);
            if !glass.is_empty() {
                pass.set_pipeline(&pipelines.glass);
                pass.set_vertex_buffer(0, ctx.geometry.vertex_buffer().slice(..));
                pass.set_index_buffer(ctx.geometry.index_buffer().slice(..), wgpu::IndexFormat::Uint32);
                for draw in glass {
                    pass.draw_indexed(draw.indices.clone(), draw.base_vertex, draw.instances());
                }
            }
        }
        if ambient_occlusion {
            fullscreen(encoder, "ao", raw, Some((ao_pipeline, ao_bg)));
            fullscreen(encoder, "ao_blur", output, Some((&pipelines.blur, blur_bg)));
        } else {
            fullscreen(encoder, "ao_blur", output, None);
        }
    }
}
