use std::sync::{Arc, RwLock};

use crate::{
    debug_flags::RenderDebugFlags,
    graph::RenderGraph,
    light::{SharedShadowCasters, virtual_shadow::SharedPageRequests},
    pass::{
        ao::AoPass,
        shadow_pages::ShadowPagesPass,
        bvh_upload::BvhUploadPass,
        camera::CameraPass,
        cluster_assignment::ClusterAssignmentPass,
        debug_overlay::DebugOverlayPass,
        fullscreen::{FullscreenInput, FullscreenPass},
        gbuffer::GBufferPass,
        light_upload::LightUploadPass,
        lighting::LightingPass,
        particle_sim::ParticleSimPass,
        particles::ParticlesPass,
        present::PresentPass,
        probe_atlas::{ProbeAtlasPass, ProbeGridConfig},
        probe_trace::ProbeTracePass,
        probe_update::ProbeUpdatePass,
        reflection_probe::ReflectionProbePass,
        prev_frame::{FrameHistory, PrevFramePass},
        ssr::{SsrInputs, SsrPass},
        ssr_temporal::SsrTemporalPass,
        shadow::ShadowPass,
        transparent::TransparentPass,
    },
    renderer::RenderGraphBuilder,
    resource::ResourceId,
    shader::ShaderAsset,
};

pub struct RenderGraphPBRBuilder {
    gbuffer_shader:        ShaderAsset,
    cluster_shader:        ShaderAsset,
    lighting_shader:       ShaderAsset,
    present_shader:        ShaderAsset,
    shadow_shader:         ShaderAsset,
    probe_trace_shader:    ShaderAsset,
    probe_update_shader:   ShaderAsset,
    debug_overlay_shader:  ShaderAsset,
    ssr_shader:            ShaderAsset,
    composite_shader:      ShaderAsset,
    ssr_temporal_shader:   ShaderAsset,
    particle_sim_shader:   ShaderAsset,
    ao_shader:             ShaderAsset,
    debug_flags:           Option<Arc<RwLock<RenderDebugFlags>>>,
}

impl RenderGraphPBRBuilder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gbuffer_shader:       ShaderAsset,
        cluster_shader:       ShaderAsset,
        lighting_shader:      ShaderAsset,
        present_shader:       ShaderAsset,
        shadow_shader:        ShaderAsset,
        probe_trace_shader:   ShaderAsset,
        probe_update_shader:  ShaderAsset,
        debug_overlay_shader: ShaderAsset,
        ssr_shader:           ShaderAsset,
        composite_shader:     ShaderAsset,
        ssr_temporal_shader:  ShaderAsset,
        particle_sim_shader:  ShaderAsset,
        ao_shader:            ShaderAsset,
    ) -> Self {
        Self {
            gbuffer_shader,
            cluster_shader,
            lighting_shader,
            present_shader,
            shadow_shader,
            probe_trace_shader,
            probe_update_shader,
            debug_overlay_shader,
            ssr_shader,
            composite_shader,
            ssr_temporal_shader,
            particle_sim_shader,
            ao_shader,
            debug_flags: None,
        }
    }

    pub fn with_debug_flags(mut self, flags: Arc<RwLock<RenderDebugFlags>>) -> Self {
        self.debug_flags = Some(flags);
        self
    }
}

impl RenderGraphBuilder for RenderGraphPBRBuilder {
    fn build(
        self,
        device: &wgpu::Device,
        surface_config: &wgpu::SurfaceConfiguration,
    ) -> (RenderGraph, ResourceId) {
        let mut graph = RenderGraph::default();
        if let Some(flags) = self.debug_flags {
            graph.debug_flags = flags;
        }

        let surface_view_id = graph.create_external_resource();

        let (camera_pass, camera_buffer_id) = CameraPass::new(&mut graph);
        let camera_node_id = graph.add_pass(device, camera_pass);

        let debug_flags = graph.debug_flags_arc();

        let shadow_casters: SharedShadowCasters = Arc::default();
        let page_requests: SharedPageRequests = Arc::default();
        let (light_upload_pass, light_buffers) =
            LightUploadPass::new(&mut graph, Arc::clone(&debug_flags), Arc::clone(&shadow_casters));
        graph.add_pass(device, light_upload_pass);

        let (bvh_pass, bvh_outputs) = BvhUploadPass::new(&mut graph);
        graph.add_pass(device, bvh_pass);

        let (probe_atlas_pass, probe_atlas_outputs) =
            ProbeAtlasPass::new(&mut graph, ProbeGridConfig::default(), std::sync::Arc::clone(&debug_flags));
        graph.add_pass(device, probe_atlas_pass);

        let (probe_trace_pass, probe_trace_outputs) = ProbeTracePass::new(
            &mut graph,
            &bvh_outputs,
            &probe_atlas_outputs,
            &light_buffers,
            self.probe_trace_shader,
        );
        graph.add_pass(device, probe_trace_pass);

        let probe_update_pass = ProbeUpdatePass::new(
            &mut graph,
            &probe_atlas_outputs,
            &probe_trace_outputs,
            self.probe_update_shader,
        );
        graph.add_pass(device, probe_update_pass);

        let (shadow_pass, shadow_outputs) =
            ShadowPass::new(&mut graph, self.shadow_shader, light_buffers.shadowed_buffer_id, shadow_casters, Arc::clone(&page_requests));
        graph.add_pass(device, shadow_pass);

        let (gbuffer_pass, gbuffer_outputs) = GBufferPass::new(
            &mut graph,
            camera_node_id,
            camera_buffer_id,
            surface_config,
            self.gbuffer_shader,
        );
        graph.add_pass(device, gbuffer_pass);
        graph.add_pass(
            device,
            ShadowPagesPass::new(gbuffer_outputs.depth_id, camera_buffer_id, shadow_outputs.local_shadow_params_id, page_requests),
        );

        let (ao_pass, ao_outputs) = AoPass::new(
            &mut graph,
            gbuffer_outputs.depth_id,
            gbuffer_outputs.normal_id,
            camera_buffer_id,
            self.ao_shader,
            Arc::clone(&debug_flags),
        );
        graph.add_pass(device, ao_pass);

        let (cluster_pass, cluster_outputs) =
            ClusterAssignmentPass::new(&mut graph, &light_buffers, self.cluster_shader);
        graph.add_pass(device, cluster_pass);

        let (reflection_probe_pass, reflection_outputs) = ReflectionProbePass::new(
            &mut graph,
            &cluster_outputs,
            &light_buffers,
            &shadow_outputs,
            &probe_atlas_outputs,
            camera_buffer_id,
            self.lighting_shader.clone(),
            Arc::clone(&debug_flags),
        );
        graph.add_pass(device, reflection_probe_pass);

        let (lighting_pass, lighting_outputs) = LightingPass::new(
            &mut graph,
            &gbuffer_outputs,
            &ao_outputs,
            &cluster_outputs,
            &light_buffers,
            &shadow_outputs,
            &probe_atlas_outputs,
            camera_buffer_id,
            &reflection_outputs,
            surface_config,
            self.lighting_shader.clone(),
            std::sync::Arc::clone(&debug_flags),
        );
        graph.add_pass(device, lighting_pass);

        let prev_frame = FrameHistory::default();
        let prev_glass_frame = FrameHistory::default();
        let (ssr_pass, ssr_id, ssr_ray_length_id) = SsrPass::new(
            &mut graph,
            SsrInputs {
                depth_id: gbuffer_outputs.depth_id,
                normal_id: gbuffer_outputs.normal_id,
                lit_id: lighting_outputs.lit_id,
                specular_weight_id: lighting_outputs.specular_weight_id,
                camera_buffer_id,
                debug_flags_id: lighting_outputs.debug_flags_id,
                prev_frame: prev_frame.clone(),
                glass_depth_id: ao_outputs.glass_depth_id,
                prev_glass_frame: prev_glass_frame.clone(),
            },
            self.ssr_shader,
        );
        graph.add_pass(device, ssr_pass);

        let (ssr_temporal_pass, ssr_resolved_id) = SsrTemporalPass::new(
            &mut graph,
            ssr_id,
            ssr_ray_length_id,
            gbuffer_outputs.depth_id,
            lighting_outputs.specular_weight_id,
            camera_buffer_id,
            self.ssr_temporal_shader,
        );
        graph.add_pass(device, ssr_temporal_pass);

        let (composite_pass, composite_id) = FullscreenPass::new(
            &mut graph,
            "composite",
            vec![
                FullscreenInput::Color(lighting_outputs.lit_id),
                FullscreenInput::Color(lighting_outputs.specular_weight_id),
                FullscreenInput::Color(lighting_outputs.specular_fallback_id),
                FullscreenInput::Color(ssr_resolved_id),
                FullscreenInput::Depth(gbuffer_outputs.depth_id),
            ],
            vec![],
            wgpu::TextureFormat::Rgba16Float,
            self.composite_shader,
        );
        graph.add_pass(device, composite_pass);
        graph.add_pass(device, PrevFramePass::new(composite_id, wgpu::TextureFormat::Rgba16Float, prev_frame));

        let (transparent_pass, transparent_id) = TransparentPass::new(
            &mut graph,
            composite_id,
            &gbuffer_outputs,
            &ao_outputs,
            &lighting_outputs,
            &cluster_outputs,
            &light_buffers,
            &shadow_outputs,
            &probe_atlas_outputs,
            camera_buffer_id,
            &reflection_outputs,
            self.lighting_shader.clone(),
            Arc::clone(&debug_flags),
        );
        graph.add_pass(device, transparent_pass);
        graph.add_pass(device, PrevFramePass::new(transparent_id, wgpu::TextureFormat::Rgba16Float, prev_glass_frame));

        let (particle_sim_pass, particle_outputs) =
            ParticleSimPass::new(
                &mut graph,
                gbuffer_outputs.depth_id,
                gbuffer_outputs.normal_id,
                camera_buffer_id,
                self.particle_sim_shader,
                Arc::clone(&debug_flags),
            );
        graph.add_pass(device, particle_sim_pass);

        let (particles_pass, particles_id) = ParticlesPass::new(
            &mut graph,
            transparent_id,
            &particle_outputs,
            &gbuffer_outputs,
            &lighting_outputs,
            &cluster_outputs,
            &light_buffers,
            &shadow_outputs,
            &probe_atlas_outputs,
            camera_buffer_id,
            &reflection_outputs,
            self.lighting_shader,
            Arc::clone(&debug_flags),
        );
        graph.add_pass(device, particles_pass);

        let (debug_overlay_pass, debug_overlay_outputs) = DebugOverlayPass::new(
            &mut graph,
            bvh_outputs.nodes_id,
            bvh_outputs.info_id,
            camera_buffer_id,
            probe_atlas_outputs.probe_params_id,
            bvh_outputs.node_count,
            debug_flags,
            self.debug_overlay_shader,
        );
        graph.add_pass(device, debug_overlay_pass);

        let present_pass = PresentPass::new(
            particles_id,
            debug_overlay_outputs.debug_overlay_id,
            surface_view_id,
            surface_config.format,
            self.present_shader,
        );
        graph.add_pass(device, present_pass);

        (graph, surface_view_id)
    }
}
