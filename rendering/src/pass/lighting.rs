use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicUsize, Ordering};

use bytemuck;
use engine_core::world::World;

use crate::{
    debug_flags::{DebugFlagsGpu, RenderDebugFlags},
    graph::{NodeId, RenderGraph},
    pass::cluster_assignment::ClusterOutputs,
    pass::gbuffer::GBufferOutputs,
    pass::light_upload::LightBuffers,
    pass::probe_atlas::ProbeAtlasOutputs,
    pass::reflection_probe::{ReflectionProbeOutputs, SharedReflectionProbe, probe_textures},
    pass::shadow::ShadowOutputs,
    pass::{Pass, PassContext, PassDesc},
    resource::{ResourceDescriptor, ResourceId},
    shader::ShaderAsset,
};

pub struct LightingOutputs {
    /// Diffuse + direct specular. Indirect specular is left to `composite`.
    pub lit_id: ResourceId,
    /// Rgba8Unorm. RGB: split-sum env BRDF weight for reflections, 0 where nothing should reflect
    /// (sky, debug views, specular or GI disabled). A: roughness.
    pub specular_weight_id: ResourceId,
    /// Rgba16Float. Probe irradiance around the reflection vector × weight, exposed: the
    /// reflection where SSR has nothing.
    pub specular_fallback_id: ResourceId,
    /// Uniform `DebugFlagsGpu`, uploaded by this pass every frame.
    pub debug_flags_id: ResourceId,
}

pub const SPECULAR_WEIGHT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_entry(
    binding: u32,
    view_dimension: wgpu::TextureViewDimension,
    sample_type: wgpu::TextureSampleType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture { multisampled: false, view_dimension, sample_type },
        count: None,
    }
}

const FILTERABLE: wgpu::TextureSampleType = wgpu::TextureSampleType::Float { filterable: true };

struct LightingLayouts {
    cluster: wgpu::BindGroupLayout,
    shadow: wgpu::BindGroupLayout,
    probe: wgpu::BindGroupLayout,
    debug_flags: wgpu::BindGroupLayout,
    probe_sampler: wgpu::Sampler,
    shadow_sampler: wgpu::Sampler,
}

/// Groups 1–4 of `lighting.wgsl`: lights and clusters, shadow maps, probes and camera, debug flags
/// and shadowed lights. Shared by the deferred `lighting` pass and the forward `transparent` pass.
pub(crate) struct LightingBindings {
    cluster_params_id: ResourceId,
    point_buffer_id: ResourceId,
    light_grid_id: ResourceId,
    light_indices_id: ResourceId,
    sky_buffer_id: ResourceId,
    spot_buffer_id: ResourceId,
    area_buffer_id: ResourceId,
    counts_buffer_id: ResourceId,

    shadow_map_id: ResourceId,
    shadow_params_id: ResourceId,
    local_shadow_map_id: ResourceId,
    local_shadow_params_id: ResourceId,
    glass_tint_id: ResourceId,
    local_glass_tint_id: ResourceId,

    irradiance_atlas_ids: [ResourceId; 2],
    visibility_atlas_ids: [ResourceId; 2],
    probe_params_id: ResourceId,
    camera_buffer_id: ResourceId,
    current_atlas: Arc<AtomicUsize>,
    reflection_params_id: ResourceId,
    reflection_textures: SharedReflectionProbe,

    debug_flags_id: ResourceId,
    shadowed_buffer_id: ResourceId,

    layouts: Option<LightingLayouts>,
    // Buffer-only bind groups never change. The shadow and probe groups are rebuilt when the
    // shadow map or the active irradiance atlas view changes.
    cluster_bg: Option<wgpu::BindGroup>,
    shadow_bg: Option<(usize, wgpu::BindGroup)>,
    probe_bg: Option<(usize, wgpu::BindGroup)>,
    debug_flags_bg: Option<wgpu::BindGroup>,
}

impl LightingBindings {
    pub(crate) fn new(
        cluster: &ClusterOutputs,
        light_buffers: &LightBuffers,
        shadow: &ShadowOutputs,
        probe_atlas: &ProbeAtlasOutputs,
        camera_buffer_id: ResourceId,
        debug_flags_id: ResourceId,
        reflection: &ReflectionProbeOutputs,
    ) -> Self {
        Self {
            cluster_params_id: cluster.cluster_params_id,
            point_buffer_id: light_buffers.point_buffer_id,
            light_grid_id: cluster.light_grid_id,
            light_indices_id: cluster.light_indices_id,
            sky_buffer_id: light_buffers.sky_buffer_id,
            spot_buffer_id: light_buffers.spot_buffer_id,
            area_buffer_id: light_buffers.area_buffer_id,
            counts_buffer_id: light_buffers.counts_buffer_id,
            shadow_map_id: shadow.shadow_map_id,
            shadow_params_id: shadow.shadow_params_id,
            local_shadow_map_id: shadow.local_shadow_map_id,
            local_shadow_params_id: shadow.local_shadow_params_id,
            glass_tint_id: shadow.glass_tint_id,
            local_glass_tint_id: shadow.local_glass_tint_id,
            irradiance_atlas_ids: probe_atlas.irradiance_atlas_ids,
            visibility_atlas_ids: probe_atlas.visibility_atlas_ids,
            probe_params_id: probe_atlas.probe_params_id,
            camera_buffer_id,
            current_atlas: Arc::clone(&probe_atlas.current_atlas),
            reflection_params_id: reflection.params_id,
            reflection_textures: Arc::clone(&reflection.textures),
            debug_flags_id,
            shadowed_buffer_id: light_buffers.shadowed_buffer_id,
            layouts: None,
            cluster_bg: None,
            shadow_bg: None,
            probe_bg: None,
            debug_flags_bg: None,
        }
    }

    /// Everything bound except the debug flags uniform, which `lighting` writes.
    pub(crate) fn reads(&self) -> Vec<ResourceId> {
        vec![
            self.cluster_params_id,
            self.point_buffer_id,
            self.light_grid_id,
            self.light_indices_id,
            self.sky_buffer_id,
            self.spot_buffer_id,
            self.area_buffer_id,
            self.counts_buffer_id,
            self.shadow_map_id,
            self.shadow_params_id,
            self.local_shadow_map_id,
            self.local_shadow_params_id,
            self.glass_tint_id,
            self.local_glass_tint_id,
            self.irradiance_atlas_ids[0],
            self.irradiance_atlas_ids[1],
            self.visibility_atlas_ids[0],
            self.visibility_atlas_ids[1],
            self.probe_params_id,
            self.camera_buffer_id,
            self.reflection_params_id,
            self.shadowed_buffer_id,
        ]
    }

    /// Layouts of groups 1–4, in order. The camera (group 3 binding 4) is visible to vertex
    /// shaders too, for forward passes.
    pub(crate) fn layouts(&mut self, device: &wgpu::Device) -> [&wgpu::BindGroupLayout; 4] {
        let layouts = self.layouts.get_or_insert_with(|| {
            let fragment = wgpu::ShaderStages::FRAGMENT;
            let cluster = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("lighting_cluster_bgl"),
                entries: &[
                    uniform_entry(0, fragment),
                    storage_entry(1),
                    storage_entry(2),
                    storage_entry(3),
                    storage_entry(4),
                    storage_entry(5),
                    storage_entry(6),
                    uniform_entry(7, fragment),
                ],
            });
            let shadow = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("lighting_shadow_bgl"),
                entries: &[
                    uniform_entry(0, fragment),
                    texture_entry(1, wgpu::TextureViewDimension::D2Array, wgpu::TextureSampleType::Depth),
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: fragment,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
                    texture_entry(3, wgpu::TextureViewDimension::D2Array, wgpu::TextureSampleType::Depth),
                    uniform_entry(4, fragment),
                    texture_entry(5, wgpu::TextureViewDimension::D2Array, FILTERABLE),
                    texture_entry(6, wgpu::TextureViewDimension::D2Array, FILTERABLE),
                ],
            });
            let probe = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("lighting_probe_bgl"),
                entries: &[
                    texture_entry(0, wgpu::TextureViewDimension::D2, FILTERABLE),
                    texture_entry(1, wgpu::TextureViewDimension::D2, FILTERABLE),
                    uniform_entry(2, fragment),
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: fragment,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    uniform_entry(4, fragment | wgpu::ShaderStages::VERTEX),
                    texture_entry(5, wgpu::TextureViewDimension::Cube, FILTERABLE),
                    uniform_entry(6, fragment),
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: fragment,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
            let debug_flags = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("lighting_debug_flags_bgl"),
                entries: &[uniform_entry(0, fragment), uniform_entry(1, fragment)],
            });
            let probe_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("probe_atlas_sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
            let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("shadow_comparison_sampler"),
                compare: Some(wgpu::CompareFunction::LessEqual),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
            LightingLayouts { cluster, shadow, probe, debug_flags, probe_sampler, shadow_sampler }
        });
        [&layouts.cluster, &layouts.shadow, &layouts.probe, &layouts.debug_flags]
    }

    /// Bind groups 1–4, built or rebuilt as needed. `None` if a resource is missing from `ctx`.
    pub(crate) fn bind_groups(&mut self, device: &wgpu::Device, ctx: &PassContext<'_>) -> Option<[&wgpu::BindGroup; 4]> {
        self.layouts(device);
        let layouts = self.layouts.as_ref()?;
        let buffer = |id: &ResourceId| ctx.buffers.get(id).copied();
        let view = |id: &ResourceId| ctx.views.get(id).copied();

        if self.cluster_bg.is_none() {
            let buffers = [
                self.cluster_params_id,
                self.point_buffer_id,
                self.light_grid_id,
                self.light_indices_id,
                self.sky_buffer_id,
                self.spot_buffer_id,
                self.area_buffer_id,
                self.counts_buffer_id,
            ]
            .iter()
            .map(buffer)
            .collect::<Option<Vec<_>>>()?;
            let entries: Vec<_> = buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.binding() })
                .collect();
            self.cluster_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lighting_cluster_bg"),
                layout: &layouts.cluster,
                entries: &entries,
            }));
        }

        let shadow_map = view(&self.shadow_map_id)?;
        let shadow_ptr = shadow_map as *const wgpu::TextureView as usize;
        if self.shadow_bg.as_ref().is_none_or(|(ptr, _)| *ptr != shadow_ptr) {
            let params = buffer(&self.shadow_params_id)?;
            let local_map = view(&self.local_shadow_map_id)?;
            let local_params = buffer(&self.local_shadow_params_id)?;
            let glass_tint = view(&self.glass_tint_id)?;
            let local_glass_tint = view(&self.local_glass_tint_id)?;
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lighting_shadow_bg"),
                layout: &layouts.shadow,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params.binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(shadow_map) },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&layouts.shadow_sampler),
                    },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(local_map) },
                    wgpu::BindGroupEntry { binding: 4, resource: local_params.binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(glass_tint) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(local_glass_tint) },
                ],
            });
            self.shadow_bg = Some((shadow_ptr, bind_group));
        }

        let atlas = self.current_atlas.load(Ordering::Relaxed);
        let irradiance = view(&self.irradiance_atlas_ids[atlas])?;
        let irradiance_ptr = irradiance as *const wgpu::TextureView as usize;
        if self.probe_bg.as_ref().is_none_or(|(ptr, _)| *ptr != irradiance_ptr) {
            let visibility = view(&self.visibility_atlas_ids[atlas])?;
            let params = buffer(&self.probe_params_id)?;
            let camera = buffer(&self.camera_buffer_id)?;
            let reflection_params = buffer(&self.reflection_params_id)?;
            let reflection = probe_textures(&self.reflection_textures, device);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lighting_probe_bg"),
                layout: &layouts.probe,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(irradiance) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(visibility) },
                    wgpu::BindGroupEntry { binding: 2, resource: params.binding() },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&layouts.probe_sampler),
                    },
                    wgpu::BindGroupEntry { binding: 4, resource: camera.binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&reflection.view) },
                    wgpu::BindGroupEntry { binding: 6, resource: reflection_params.binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(&reflection.sampler) },
                ],
            });
            self.probe_bg = Some((irradiance_ptr, bind_group));
        }

        if self.debug_flags_bg.is_none() {
            let flags = buffer(&self.debug_flags_id)?;
            let shadowed = buffer(&self.shadowed_buffer_id)?;
            self.debug_flags_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lighting_debug_flags_bg"),
                layout: &layouts.debug_flags,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: flags.binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: shadowed.binding() },
                ],
            }));
        }

        Some([
            self.cluster_bg.as_ref()?,
            &self.shadow_bg.as_ref()?.1,
            &self.probe_bg.as_ref()?.1,
            self.debug_flags_bg.as_ref()?,
        ])
    }
}

pub struct LightingPass {
    node_id: Option<NodeId>,
    shader: ShaderAsset,
    pipeline: Option<wgpu::RenderPipeline>,
    gbuffer_bgl: Option<wgpu::BindGroupLayout>,
    sampler: Option<wgpu::Sampler>,

    albedo_id: ResourceId,
    normal_id: ResourceId,
    material_id: ResourceId,
    emissive_id: ResourceId,
    depth_id: ResourceId,

    lit_id: ResourceId,
    specular_weight_id: ResourceId,
    specular_fallback_id: ResourceId,

    // Binds texture views — recreated when albedo is reallocated (resize).
    gbuffer_bg: Option<wgpu::BindGroup>,
    cached_albedo_ptr: usize,

    bindings: LightingBindings,

    debug_flags: Arc<RwLock<RenderDebugFlags>>,
    debug_flags_id: ResourceId,
}

impl LightingPass {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        graph: &mut RenderGraph,
        gbuffer: &GBufferOutputs,
        cluster: &ClusterOutputs,
        light_buffers: &LightBuffers,
        shadow: &ShadowOutputs,
        probe_atlas: &ProbeAtlasOutputs,
        camera_buffer_id: ResourceId,
        reflection: &ReflectionProbeOutputs,
        _surface_config: &wgpu::SurfaceConfiguration,
        shader: ShaderAsset,
        debug_flags: Arc<RwLock<RenderDebugFlags>>,
    ) -> (Self, LightingOutputs) {
        let lit_id = graph.alloc_resource_id(ResourceDescriptor::ScreenTexture {
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        let specular_weight_id = graph.alloc_resource_id(ResourceDescriptor::ScreenTexture {
            format: SPECULAR_WEIGHT_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        });
        let specular_fallback_id = graph.alloc_resource_id(ResourceDescriptor::ScreenTexture {
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        let debug_flags_id = graph.alloc_resource_id(ResourceDescriptor::UploadBuffer {
            size: std::mem::size_of::<DebugFlagsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let outputs = LightingOutputs { lit_id, specular_weight_id, specular_fallback_id, debug_flags_id };

        (
            Self {
                node_id: None,
                shader,
                pipeline: None,
                gbuffer_bgl: None,
                sampler: None,
                albedo_id: gbuffer.albedo_id,
                normal_id: gbuffer.normal_id,
                material_id: gbuffer.material_id,
                emissive_id: gbuffer.emissive_id,
                depth_id: gbuffer.depth_id,
                lit_id,
                specular_weight_id,
                specular_fallback_id,
                gbuffer_bg: None,
                cached_albedo_ptr: 0,
                bindings: LightingBindings::new(
                    cluster,
                    light_buffers,
                    shadow,
                    probe_atlas,
                    camera_buffer_id,
                    debug_flags_id,
                    reflection,
                ),
                debug_flags,
                debug_flags_id,
            },
            outputs,
        )
    }
}

impl PassDesc for LightingPass {
    fn name(&self) -> &'static str {
        "lighting"
    }

    fn reads(&self) -> Vec<ResourceId> {
        let mut reads = vec![self.albedo_id, self.normal_id, self.material_id, self.emissive_id, self.depth_id];
        reads.extend(self.bindings.reads());
        reads
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![self.lit_id, self.specular_weight_id, self.specular_fallback_id, self.debug_flags_id]
    }
}

impl Pass for LightingPass {
    fn bind_node_id(&mut self, node_id: NodeId) {
        self.node_id = Some(node_id);
    }

    fn on_resize(&mut self, _width: u32, _height: u32) {
        self.gbuffer_bg = None;
        self.cached_albedo_ptr = 0;
    }

    fn execute(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &PassContext<'_>,
        _world: &World,
    ) {
        let views = [
            self.lit_id,
            self.specular_weight_id,
            self.specular_fallback_id,
            self.albedo_id,
            self.normal_id,
            self.material_id,
            self.depth_id,
            self.emissive_id,
        ]
        .map(|id| ctx.views.get(&id).copied());
        let [
            Some(lit_view),
            Some(specular_weight_view),
            Some(specular_fallback_view),
            Some(albedo_view),
            Some(normal_view),
            Some(material_view),
            Some(depth_view),
            Some(emissive_view),
        ] = views
        else {
            return;
        };
        let Some(&debug_flags_buf) = ctx.buffers.get(&self.debug_flags_id) else { return };

        let gpu_flags = self
            .debug_flags
            .read()
            .expect("debug_flags RwLock poisoned")
            .to_gpu();
        debug_flags_buf.write(queue, bytemuck::bytes_of(&gpu_flags));

        if self.pipeline.is_none() {
            let module = self
                .shader
                .compile(device)
                .expect("lighting shader compile failed");

            let gbuffer_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("lighting_gbuffer_bgl"),
                entries: &[
                    texture_entry(0, wgpu::TextureViewDimension::D2, FILTERABLE),
                    texture_entry(1, wgpu::TextureViewDimension::D2, FILTERABLE),
                    texture_entry(2, wgpu::TextureViewDimension::D2, FILTERABLE),
                    texture_entry(3, wgpu::TextureViewDimension::D2, wgpu::TextureSampleType::Depth),
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    texture_entry(5, wgpu::TextureViewDimension::D2, FILTERABLE),
                ],
            });

            let [cluster_bgl, shadow_bgl, probe_bgl, debug_flags_bgl] = self.bindings.layouts(device);
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("lighting_pipeline_layout"),
                bind_group_layouts: &[
                    Some(&gbuffer_bgl),
                    Some(cluster_bgl),
                    Some(shadow_bgl),
                    Some(probe_bgl),
                    Some(debug_flags_bgl),
                ],
                immediate_size: 0,
            });

            let target = |format| {
                Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })
            };
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("lighting_pipeline"),
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
                    targets: &[
                        target(wgpu::TextureFormat::Rgba16Float),
                        target(SPECULAR_WEIGHT_FORMAT),
                        target(wgpu::TextureFormat::Rgba16Float),
                    ],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("lighting_sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });

            self.pipeline = Some(pipeline);
            self.gbuffer_bgl = Some(gbuffer_bgl);
            self.sampler = Some(sampler);
        }

        let (Some(pipeline), Some(gbuffer_bgl), Some(sampler)) = (&self.pipeline, &self.gbuffer_bgl, &self.sampler)
        else {
            return;
        };

        let albedo_ptr = albedo_view as *const wgpu::TextureView as usize;
        if self.gbuffer_bg.is_none() || albedo_ptr != self.cached_albedo_ptr {
            self.gbuffer_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lighting_gbuffer_bg"),
                layout: gbuffer_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(albedo_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(normal_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(material_view) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(depth_view) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(sampler) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(emissive_view) },
                ],
            }));
            self.cached_albedo_ptr = albedo_ptr;
        }
        let Some(gbuffer_bg) = self.gbuffer_bg.as_ref() else { return };
        let Some(lighting_bgs) = self.bindings.bind_groups(device, ctx) else { return };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("lighting"),
            color_attachments: &[lit_view, specular_weight_view, specular_fallback_view].map(|view| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        store: wgpu::StoreOp::Store,
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    },
                })
            }),
            depth_stencil_attachment: None,
            ..Default::default()
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, gbuffer_bg, &[]);
        for (i, bind_group) in lighting_bgs.into_iter().enumerate() {
            pass.set_bind_group(i as u32 + 1, bind_group, &[]);
        }
        pass.draw(0..3, 0..1);
    }
}
