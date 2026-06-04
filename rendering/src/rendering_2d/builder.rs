use crate::{graph::RenderGraph, renderer::RenderGraphBuilder, resource::ResourceId};

use crate::rendering_2d::pass::SpritePass;

pub struct SpriteRenderGraphBuilder;

impl RenderGraphBuilder for SpriteRenderGraphBuilder {
    fn build(
        self,
        device: &wgpu::Device,
        surface_config: &wgpu::SurfaceConfiguration,
    ) -> (RenderGraph, ResourceId) {
        let mut graph = RenderGraph::default();
        let surface_id = graph.create_external_resource();

        let sprite_pass = SpritePass::new(
            surface_id,
            surface_config.format,
            Some(wgpu::Color::BLACK),
        );
        graph.add_pass(device, sprite_pass);

        (graph, surface_id)
    }
}
