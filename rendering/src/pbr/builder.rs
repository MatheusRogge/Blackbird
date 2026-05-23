use crate::{
    graph::RenderGraph,
    pass::{
        bvh_upload::BvhUploadPass,
        camera::CameraPass,
        cluster_assignment::ClusterAssignmentPass,
        gbuffer::GBufferPass,
        light_upload::LightUploadPass,
        lighting::LightingPass,
        present::PresentPass,
        probe_atlas::{ProbeAtlasPass, ProbeGridConfig},
        probe_trace::ProbeTracePass,
        probe_update::ProbeUpdatePass,
        shadow::ShadowPass,
    },
    renderer::RenderGraphBuilder,
    resource::ResourceId,
    shader::ShaderAsset,
};

pub struct RenderGraphPBRBuilder {
    gbuffer_shader:     ShaderAsset,
    cluster_shader:     ShaderAsset,
    lighting_shader:    ShaderAsset,
    present_shader:     ShaderAsset,
    shadow_shader:      ShaderAsset,
    probe_trace_shader: ShaderAsset,
    probe_update_shader: ShaderAsset,
}

impl RenderGraphPBRBuilder {
    pub fn new(
        gbuffer_shader:      ShaderAsset,
        cluster_shader:      ShaderAsset,
        lighting_shader:     ShaderAsset,
        present_shader:      ShaderAsset,
        shadow_shader:       ShaderAsset,
        probe_trace_shader:  ShaderAsset,
        probe_update_shader: ShaderAsset,
    ) -> Self {
        Self {
            gbuffer_shader,
            cluster_shader,
            lighting_shader,
            present_shader,
            shadow_shader,
            probe_trace_shader,
            probe_update_shader,
        }
    }
}

impl RenderGraphBuilder for RenderGraphPBRBuilder {
    fn build(
        self,
        device: &wgpu::Device,
        surface_config: &wgpu::SurfaceConfiguration,
    ) -> (RenderGraph, ResourceId) {
        let mut graph = RenderGraph::default();

        let surface_view_id = graph.create_external_resource();

        let (camera_pass, camera_buffer_id) = CameraPass::new(&mut graph);
        let camera_node_id = graph.add_pass(device, camera_pass);

        let (light_upload_pass, light_buffers) = LightUploadPass::new(&mut graph);
        graph.add_pass(device, light_upload_pass);

        let (bvh_pass, bvh_outputs) = BvhUploadPass::new(&mut graph);
        graph.add_pass(device, bvh_pass);

        let (probe_atlas_pass, probe_atlas_outputs) =
            ProbeAtlasPass::new(&mut graph, ProbeGridConfig::default());
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

        let (shadow_pass, shadow_outputs) = ShadowPass::new(&mut graph, self.shadow_shader);
        graph.add_pass(device, shadow_pass);

        let (gbuffer_pass, gbuffer_outputs) = GBufferPass::new(
            &mut graph,
            camera_node_id,
            camera_buffer_id,
            surface_config,
            self.gbuffer_shader,
        );
        graph.add_pass(device, gbuffer_pass);

        let (cluster_pass, cluster_outputs) =
            ClusterAssignmentPass::new(&mut graph, &light_buffers, self.cluster_shader);
        graph.add_pass(device, cluster_pass);

        let (lighting_pass, lighting_outputs) = LightingPass::new(
            &mut graph,
            &gbuffer_outputs,
            &cluster_outputs,
            &light_buffers,
            &shadow_outputs,
            &probe_atlas_outputs,
            camera_buffer_id,
            surface_config,
            self.lighting_shader,
        );
        graph.add_pass(device, lighting_pass);

        let present_pass = PresentPass::new(
            lighting_outputs.lit_id,
            surface_view_id,
            surface_config.format,
            self.present_shader,
        );
        graph.add_pass(device, present_pass);

        (graph, surface_view_id)
    }
}
