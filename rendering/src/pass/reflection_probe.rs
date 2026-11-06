//! Camera-following reflection probe: a cube capture of the lit opaque scene around the camera,
//! GGX-prefiltered into mips, which `lighting` and `transparent` use for reflections SSR can't
//! resolve (off-screen content).

use std::sync::{Arc, OnceLock, RwLock};

use bytemuck::{Pod, Zeroable};
use engine_core::world::World;
use ultraviolet::{Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

use crate::{
    arena::GeometryArena,
    camera::Camera,
    debug_flags::RenderDebugFlags,
    graph::{NodeId, RenderGraph},
    mesh::{Mesh, Vertex},
    pass::cluster_assignment::ClusterOutputs,
    pass::light_upload::LightBuffers,
    pass::lighting::LightingBindings,
    pass::material::{MaterialTextures, material_key},
    pass::probe_atlas::ProbeAtlasOutputs,
    pass::shadow::ShadowOutputs,
    pass::{Pass, PassContext, PassDesc},
    resource::{ResourceDescriptor, ResourceId},
    shader::ShaderAsset,
};

const FILTER_SHADER: &str = include_str!("reflection_probe.wgsl");

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Face size of the capture, which has a full box-filtered mip chain for filtered importance sampling.
pub const CAPTURE_SIZE: u32 = 256;
const CAPTURE_MIPS: u32 = CAPTURE_SIZE.ilog2() + 1;
/// Face size of the prefiltered cube `lighting` samples. Level `k` is GGX-filtered for perceptual
/// roughness `k / (PREFILTER_MIPS - 1)`.
pub const PREFILTER_SIZE: u32 = 128;
pub const PREFILTER_MIPS: u32 = 6;
const NEAR: f32 = 0.05;
const FAR: f32 = 500.0;
/// How far the camera can move from the probe center before the next capture cycle moves the
/// center to it.
const RECAPTURE_DISTANCE: f32 = 1.0;
const STRIDE: u64 = 256;

/// Cube faces in Vulkan order (+X −X +Y −Y +Z −Z) as (forward, right, up), where `right` is the
/// direction of increasing u and `up` of decreasing v. This basis is left-handed, so capture
/// pipelines use clockwise front faces.
const FACES: [(Vec3, Vec3, Vec3); 6] = [
    (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 1.0, 0.0)),
    (Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 1.0, 0.0)),
    (Vec3::new(0.0, 1.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0)),
    (Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
    (Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)),
    (Vec3::new(0.0, 0.0, -1.0), Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)),
];

/// Mirrors `ReflectionProbeParams` in `lighting.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ReflectionProbeParams {
    /// xyz: world-space probe center; w: 1 once captured (0 = use the irradiance fallback).
    pub center: [f32; 4],
    /// Prefiltered mip for roughness 1.
    pub max_lod: f32,
    pub _pad: [f32; 3],
}

/// Mirrors `ProbeCapture` in `lighting.wgsl`; one per face at `STRIDE`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CaptureView {
    view_proj: [[f32; 4]; 4],
    center: [f32; 4],
    forward: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
}

/// Mirrors `FilterParams` in `reflection_probe.wgsl`; one per (face, level) at `STRIDE`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FilterParams {
    face: u32,
    roughness: f32,
    src_size: f32,
    dst_size: f32,
}

/// The probe's GPU textures, created on first use by whichever pass gets there first.
pub struct ReflectionProbeTextures {
    capture: wgpu::Texture,
    prefiltered: wgpu::Texture,
    /// Cube view of every prefiltered mip, for shading.
    pub(crate) view: wgpu::TextureView,
    /// Clamped trilinear.
    pub(crate) sampler: wgpu::Sampler,
}

impl ReflectionProbeTextures {
    fn new(device: &wgpu::Device) -> Self {
        let cube = |label, size, mips| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 6 },
                mip_level_count: mips,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let capture = cube("reflection_probe_capture", CAPTURE_SIZE, CAPTURE_MIPS);
        let prefiltered = cube("reflection_probe_prefiltered", PREFILTER_SIZE, PREFILTER_MIPS);
        let view = prefiltered.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("reflection_probe_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        Self { capture, prefiltered, view, sampler }
    }
}

pub(crate) type SharedReflectionProbe = Arc<OnceLock<ReflectionProbeTextures>>;

pub(crate) fn probe_textures<'a>(shared: &'a SharedReflectionProbe, device: &wgpu::Device) -> &'a ReflectionProbeTextures {
    shared.get_or_init(|| ReflectionProbeTextures::new(device))
}

pub struct ReflectionProbeOutputs {
    /// Uniform `ReflectionProbeParams`, written every frame.
    pub params_id: ResourceId,
    pub(crate) textures: SharedReflectionProbe,
}

struct Filter {
    downsample: wgpu::RenderPipeline,
    prefilter: wgpu::RenderPipeline,
    /// [face][level]: level `l` of the capture reads level `l − 1`; index 0 is unused.
    downsample_bgs: Vec<wgpu::BindGroup>,
    prefilter_bg: wgpu::BindGroup,
    capture_targets: Vec<Vec<wgpu::TextureView>>,
    prefilter_targets: Vec<Vec<wgpu::TextureView>>,
    _params: wgpu::Buffer,
}

impl Filter {
    fn new(device: &wgpu::Device, textures: &ReflectionProbeTextures) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reflection_probe_filter"),
            source: wgpu::ShaderSource::Wgsl(FILTER_SHADER.into()),
        });
        let texture_entry = |binding, view_dimension| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                view_dimension,
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
            },
            count: None,
        };
        let common = [
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<FilterParams>() as u64),
                },
                count: None,
            },
        ];
        let downsample_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reflection_probe_downsample_bgl"),
            entries: &[texture_entry(0, wgpu::TextureViewDimension::D2Array), common[0], common[1]],
        });
        let prefilter_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reflection_probe_prefilter_bgl"),
            entries: &[texture_entry(1, wgpu::TextureViewDimension::Cube), common[0], common[1]],
        });
        let pipeline = |label, layout: &wgpu::BindGroupLayout, entry_point| {
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

        // Static per-(face, level) parameters: capture levels first, then prefiltered levels.
        let mut params = Vec::new();
        for face in 0..6u32 {
            for level in 0..CAPTURE_MIPS {
                params.push(FilterParams { face, roughness: 0.0, src_size: 0.0, dst_size: (CAPTURE_SIZE >> level) as f32 });
            }
            for level in 0..PREFILTER_MIPS {
                params.push(FilterParams {
                    face,
                    roughness: level as f32 / (PREFILTER_MIPS - 1) as f32,
                    src_size: CAPTURE_SIZE as f32,
                    dst_size: (PREFILTER_SIZE >> level) as f32,
                });
            }
        }
        let mut bytes = vec![0u8; params.len() * STRIDE as usize];
        for (i, p) in params.iter().enumerate() {
            let offset = i * STRIDE as usize;
            bytes[offset..offset + std::mem::size_of::<FilterParams>()].copy_from_slice(bytemuck::bytes_of(p));
        }
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("reflection_probe_filter_params"),
            contents: &bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let params_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &params_buffer,
            offset: 0,
            size: wgpu::BufferSize::new(std::mem::size_of::<FilterParams>() as u64),
        });

        let level_views = |texture: &wgpu::Texture, level, dimension, layers: std::ops::Range<u32>| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(dimension),
                base_mip_level: level,
                mip_level_count: Some(1),
                base_array_layer: layers.start,
                array_layer_count: Some(layers.end - layers.start),
                ..Default::default()
            })
        };
        let targets = |texture: &wgpu::Texture, mips: u32| -> Vec<Vec<wgpu::TextureView>> {
            (0..6)
                .map(|face| (0..mips).map(|l| level_views(texture, l, wgpu::TextureViewDimension::D2, face..face + 1)).collect())
                .collect()
        };

        let downsample_bgs = (0..CAPTURE_MIPS)
            .map(|level| {
                let source = level_views(&textures.capture, level.saturating_sub(1), wgpu::TextureViewDimension::D2Array, 0..6);
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("reflection_probe_downsample_bg"),
                    layout: &downsample_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&source) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&textures.sampler) },
                        wgpu::BindGroupEntry { binding: 3, resource: params_binding.clone() },
                    ],
                })
            })
            .collect();
        let capture_cube = textures.capture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        let prefilter_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reflection_probe_prefilter_bg"),
            layout: &prefilter_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&capture_cube) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&textures.sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: params_binding },
            ],
        });

        Self {
            downsample: pipeline("reflection_probe_downsample", &downsample_layout, "fs_downsample"),
            prefilter: pipeline("reflection_probe_prefilter", &prefilter_layout, "fs_prefilter"),
            downsample_bgs,
            prefilter_bg,
            capture_targets: targets(&textures.capture, CAPTURE_MIPS),
            prefilter_targets: targets(&textures.prefiltered, PREFILTER_MIPS),
            _params: params_buffer,
        }
    }

    fn params_offset(face: usize, slot: u32) -> u32 {
        ((face as u32 * (CAPTURE_MIPS + PREFILTER_MIPS) + slot) as u64 * STRIDE) as u32
    }

    fn draw(encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView, pipeline: &wgpu::RenderPipeline, bg: &wgpu::BindGroup, offset: u32) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("reflection_probe_filter"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[offset]);
        pass.draw(0..3, 0..1);
    }

    /// Rebuilds the capture mip chain of `faces`.
    fn downsample(&self, encoder: &mut wgpu::CommandEncoder, faces: &[usize]) {
        for &face in faces {
            for level in 1..CAPTURE_MIPS {
                let offset = Self::params_offset(face, level);
                Self::draw(encoder, &self.capture_targets[face][level as usize], &self.downsample, &self.downsample_bgs[level as usize], offset);
            }
        }
    }

    /// GGX-prefilters every face and level of the cube `lighting` samples.
    fn prefilter(&self, encoder: &mut wgpu::CommandEncoder) {
        for face in 0..6 {
            for level in 0..PREFILTER_MIPS {
                let offset = Self::params_offset(face, CAPTURE_MIPS + level);
                Self::draw(encoder, &self.prefilter_targets[face][level as usize], &self.prefilter, &self.prefilter_bg, offset);
            }
        }
    }
}

struct Capture {
    scene_layout: wgpu::BindGroupLayout,
    /// Keyed by the geometry generation.
    scene_bg: Option<(u64, wgpu::BindGroup)>,
    /// Depth only: back-face culled for single-sided opaque meshes, unculled and alpha-tested for
    /// masked and double-sided ones.
    prepass_opaque: wgpu::RenderPipeline,
    prepass_masked: wgpu::RenderPipeline,
    /// Shading against the prepass depth with an `Equal` test, so each texel is shaded once.
    shade_opaque: wgpu::RenderPipeline,
    shade_masked: wgpu::RenderPipeline,
    sky: wgpu::RenderPipeline,
    depth: wgpu::TextureView,
    materials: MaterialTextures,
    filter: Filter,
}

/// Renders the reflection probe and writes `ReflectionProbeParams`. The capture shades opaque
/// meshes with `lighting.wgsl`'s `vs_probe` / `fs_probe` (same lights, shadows and DDGI as the
/// deferred path, but with every point light instead of the camera's clusters, and the sky shadow
/// from whichever cascade covers the point), then the sky behind them.
pub struct ReflectionProbePass {
    node_id: Option<NodeId>,
    shader: ShaderAsset,
    textures: SharedReflectionProbe,
    capture: Option<Capture>,
    bindings: LightingBindings,
    debug_flags: Arc<RwLock<RenderDebugFlags>>,

    params_id: ResourceId,
    views_id: ResourceId,
    /// This pass's own copy of the debug flags uniform: `lighting` writes the main one, and
    /// `lighting` must run after this pass.
    debug_flags_id: ResourceId,

    /// Center of the six-face cycle in progress. One face is captured per frame, continuously, so
    /// moved geometry, changed lights and converging GI show up within two cycles, at a constant cost.
    cycle_center: Option<Vec3>,
    /// Center of the last completed cycle: what the prefiltered cube shows.
    published: Option<Vec3>,
    next_face: usize,
}

impl ReflectionProbePass {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        graph: &mut RenderGraph,
        cluster: &ClusterOutputs,
        light_buffers: &LightBuffers,
        shadow: &ShadowOutputs,
        probe_atlas: &ProbeAtlasOutputs,
        camera_buffer_id: ResourceId,
        lighting_shader: ShaderAsset,
        debug_flags: Arc<RwLock<RenderDebugFlags>>,
    ) -> (Self, ReflectionProbeOutputs) {
        let upload = |graph: &mut RenderGraph, size| {
            graph.alloc_resource_id(ResourceDescriptor::UploadBuffer {
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        let params_id = upload(graph, std::mem::size_of::<ReflectionProbeParams>() as u64);
        let views_id = upload(graph, STRIDE * 6);
        let debug_flags_id = upload(graph, std::mem::size_of::<crate::debug_flags::DebugFlagsGpu>() as u64);
        let textures = SharedReflectionProbe::default();
        let outputs = ReflectionProbeOutputs { params_id, textures: Arc::clone(&textures) };
        let bindings = LightingBindings::new(cluster, light_buffers, shadow, probe_atlas, camera_buffer_id, debug_flags_id, &outputs);
        let pass = Self {
            node_id: None,
            shader: lighting_shader,
            textures,
            capture: None,
            bindings,
            debug_flags,
            params_id,
            views_id,
            debug_flags_id,
            cycle_center: None,
            published: None,
            next_face: 0,
        };
        (pass, outputs)
    }

    fn create_capture(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Capture {
        let module = self.shader.compile(device).expect("lighting shader compile failed");
        let both = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
        let scene_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reflection_probe_scene_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { visibility: both, ..GeometryArena::instance_layout_entry(7) },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: both,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<CaptureView>() as u64),
                    },
                    count: None,
                },
            ],
        });
        let materials = MaterialTextures::new(device, queue, "reflection_probe_materials");
        let [cluster, shadow, probe, debug_flags] = self.bindings.layouts(device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("reflection_probe_pipeline_layout"),
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
        let color_target = [Some(FORMAT.into())];
        // (label, vertex entry, fragment entry, color targets?, vertex buffers, cull, depth write, depth compare)
        let pipeline = |label,
                        vs,
                        fs: Option<&str>,
                        color: bool,
                        buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
                        cull_mode,
                        write_depth,
                        compare| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(vs),
                    buffers,
                    compilation_options: Default::default(),
                },
                fragment: fs.map(|fs| wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(fs),
                    targets: if color { &color_target } else { &[] },
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState { front_face: wgpu::FrontFace::Cw, cull_mode, ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(write_depth),
                    depth_compare: Some(compare),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        use wgpu::CompareFunction::{Equal, LessEqual};
        let vertex = [Some(Vertex::buffer_descriptor())];
        let back = Some(wgpu::Face::Back);
        let prepass_opaque = pipeline("reflection_probe_prepass", "vs_probe", None, false, &vertex, back, true, LessEqual);
        let prepass_masked =
            pipeline("reflection_probe_prepass_masked", "vs_probe", Some("fs_probe_depth"), false, &vertex, None, true, LessEqual);
        let shade_opaque = pipeline("reflection_probe_shade", "vs_probe", Some("fs_probe"), true, &vertex, back, false, Equal);
        let shade_masked = pipeline("reflection_probe_shade_masked", "vs_probe", Some("fs_probe"), true, &vertex, None, false, Equal);
        let sky = pipeline("reflection_probe_sky", "vs_probe_sky", Some("fs_probe_sky"), true, &[], None, false, LessEqual);

        let depth = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("reflection_probe_depth"),
                size: wgpu::Extent3d { width: CAPTURE_SIZE, height: CAPTURE_SIZE, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        let filter = Filter::new(device, probe_textures(&self.textures, device));
        Capture {
            scene_layout,
            scene_bg: None,
            prepass_opaque,
            prepass_masked,
            shade_opaque,
            shade_masked,
            sky,
            depth,
            materials,
            filter,
        }
    }
}

fn face_view_proj(center: Vec3, (forward, right, up): (Vec3, Vec3, Vec3)) -> Mat4 {
    let view = Mat4::new(
        Vec4::new(right.x, up.x, -forward.x, 0.0),
        Vec4::new(right.y, up.y, -forward.y, 0.0),
        Vec4::new(right.z, up.z, -forward.z, 0.0),
        Vec4::new(-right.dot(center), -up.dot(center), forward.dot(center), 1.0),
    );
    ultraviolet::projection::perspective_wgpu_dx(std::f32::consts::FRAC_PI_2, 1.0, NEAR, FAR) * view
}

impl PassDesc for ReflectionProbePass {
    fn name(&self) -> &'static str {
        "reflection_probe"
    }

    fn reads(&self) -> Vec<ResourceId> {
        // The bindings include the params this pass writes; listing them would make it depend on itself.
        self.bindings.reads().into_iter().filter(|id| *id != self.params_id).collect()
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![self.params_id, self.views_id, self.debug_flags_id]
    }
}

impl Pass for ReflectionProbePass {
    fn bind_node_id(&mut self, node_id: NodeId) {
        self.node_id = Some(node_id);
    }

    fn execute(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &PassContext<'_>,
        world: &World,
    ) {
        let (Some(&params_buf), Some(&views_buf), Some(&flags_buf)) =
            (ctx.buffers.get(&self.params_id), ctx.buffers.get(&self.views_id), ctx.buffers.get(&self.debug_flags_id))
        else {
            return;
        };
        let (enabled, flags_gpu) = {
            let flags = self.debug_flags.read().expect("debug_flags poisoned");
            (flags.reflection_probe && flags.reflections && flags.specular && flags.indirect, flags.to_gpu())
        };
        let camera = world.get_entities::<Camera>().first().map(|c| c.eye);
        let (true, Some(eye)) = (enabled, camera) else {
            self.cycle_center = None;
            self.published = None;
            self.next_face = 0;
            params_buf.write(queue, bytemuck::bytes_of(&ReflectionProbeParams::zeroed()));
            return;
        };
        flags_buf.write(queue, bytemuck::bytes_of(&flags_gpu));

        if self.capture.is_none() {
            self.capture = Some(self.create_capture(device, queue));
        }
        let Some(lighting_bgs) = self.bindings.bind_groups(device, ctx) else { return };
        let Some(capture) = self.capture.as_mut() else { return };

        let meshes: Vec<_> =
            world.get_entity_handles::<Mesh>().into_iter().filter(|h| !h.entity.is_blended()).collect();
        capture.materials.prepare(device, queue, encoder, meshes.iter().map(|h| h.entity));

        let face = self.next_face;
        if face == 0 && self.cycle_center.is_none_or(|c| (eye - c).mag() > RECAPTURE_DISTANCE) {
            self.cycle_center = Some(eye);
        }
        let center = self.cycle_center.unwrap_or(eye);

        let mut views = [CaptureView::zeroed(); 6];
        for (face, view) in views.iter_mut().enumerate() {
            let (forward, right, up) = FACES[face];
            let cols = face_view_proj(center, FACES[face]).cols;
            *view = CaptureView {
                view_proj: [cols[0].into(), cols[1].into(), cols[2].into(), cols[3].into()],
                center: [center.x, center.y, center.z, 1.0],
                forward: [forward.x, forward.y, forward.z, 0.0],
                right: [right.x, right.y, right.z, 0.0],
                up: [up.x, up.y, up.z, 0.0],
            };
            views_buf.write_at(queue, STRIDE * face as u64, bytemuck::bytes_of(view));
        }

        let generation = ctx.geometry.generation();
        if capture.scene_bg.as_ref().is_none_or(|(g, _)| *g != generation) {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("reflection_probe_scene_bg"),
                layout: &capture.scene_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 7, resource: ctx.geometry.instance_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            size: wgpu::BufferSize::new(std::mem::size_of::<CaptureView>() as u64),
                            ..views_buf.buffer_binding()
                        }),
                    },
                ],
            });
            capture.scene_bg = Some((generation, bind_group));
        }
        let Some((_, scene_bg)) = capture.scene_bg.as_ref() else { return };

        let offset = (STRIDE * face as u64) as u32;
        let view_proj = face_view_proj(center, FACES[face]);
        // Front to back, so hidden fragments fail the depth test before they are shaded.
        let mut draws: Vec<_> = meshes
            .iter()
            .filter_map(|handle| {
                let draw = ctx.geometry.get(handle.id)?;
                if draw.bounds.outside_clip(view_proj * handle.entity.transform) {
                    return None;
                }
                let bounds = draw.bounds.transformed(handle.entity.transform);
                Some((((bounds.min + bounds.max) * 0.5 - center).mag_sq(), handle.entity, draw))
            })
            .collect();
        draws.sort_by(|a, b| a.0.total_cmp(&b.0));

        let bind_common = |pass: &mut wgpu::RenderPass<'_>| {
            pass.set_bind_group(0, scene_bg, &[offset]);
            for (i, bind_group) in lighting_bgs.into_iter().enumerate() {
                pass.set_bind_group(i as u32 + 1, bind_group, &[]);
            }
            pass.set_vertex_buffer(0, ctx.geometry.vertex_buffer().slice(..));
            pass.set_index_buffer(ctx.geometry.index_buffer().slice(..), wgpu::IndexFormat::Uint32);
        };
        let draw_meshes = |pass: &mut wgpu::RenderPass<'_>, opaque, masked| {
            for (_, mesh, draw) in &draws {
                pass.set_pipeline(if mesh.is_masked() { masked } else { opaque });
                pass.set_bind_group(5, capture.materials.bind_group(material_key(mesh)), &[]);
                pass.draw_indexed(draw.indices.clone(), draw.base_vertex, draw.instances());
            }
        };

        let mut prepass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("reflection_probe_prepass"),
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &capture.depth,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        bind_common(&mut prepass);
        draw_meshes(&mut prepass, &capture.prepass_opaque, &capture.prepass_masked);
        drop(prepass);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("reflection_probe_face"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &capture.filter.capture_targets[face][0],
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &capture.depth,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Discard }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        bind_common(&mut pass);
        draw_meshes(&mut pass, &capture.shade_opaque, &capture.shade_masked);
        pass.set_pipeline(&capture.sky);
        pass.set_bind_group(5, capture.materials.bind_group(Default::default()), &[]);
        pass.draw(0..3, 0..1);
        drop(pass);

        // The prefiltered cube only changes when a cycle completes, so `lighting` always samples six
        // faces from one center. GGX lobes cross faces, so prefiltering reads all six.
        capture.filter.downsample(encoder, &[face]);
        self.next_face = (face + 1) % 6;
        if self.next_face == 0 {
            capture.filter.prefilter(encoder);
            self.published = Some(center);
        }

        let Some(published) = self.published else {
            params_buf.write(queue, bytemuck::bytes_of(&ReflectionProbeParams::zeroed()));
            return;
        };
        let params = ReflectionProbeParams {
            center: [published.x, published.y, published.z, 1.0],
            max_lod: (PREFILTER_MIPS - 1) as f32,
            _pad: [0.0; 3],
        };
        params_buf.write(queue, bytemuck::bytes_of(&params));
    }
}
