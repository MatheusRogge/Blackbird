use std::sync::Arc;

use crate::{
    graph::RenderGraph,
    pass::{
        camera::CameraPass,
        gbuffer::GBufferPass,
        present::PresentPass,
    },
    renderer::RenderGraphBuilder,
    resource::ResourceId,
    shader::ShaderAsset,
};

pub struct RenderGraphPBRBuilder {
    gbuffer_shader: Arc<ShaderAsset>,
    present_shader: Arc<ShaderAsset>,
}

impl RenderGraphPBRBuilder {
    pub fn new(gbuffer_shader: Arc<ShaderAsset>, present_shader: Arc<ShaderAsset>) -> Self {
        Self {
            gbuffer_shader,
            present_shader,
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

        let (gbuffer_pass, gbuffer_outputs) = GBufferPass::new(
            &mut graph,
            camera_node_id,
            camera_buffer_id,
            surface_config,
            self.gbuffer_shader,
        );
        graph.add_pass(device, gbuffer_pass);

        let present_pass = PresentPass::new(
            gbuffer_outputs.albedo_id,
            surface_view_id,
            surface_config.format,
            self.present_shader,
        );
        graph.add_pass(device, present_pass);

        (graph, surface_view_id)
    }
}
