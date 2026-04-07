pub mod camera;
pub mod gbuffer;
pub mod present;

use crate::{
    graph::NodeId,
    resource::{BindingResource, ResourceId},
};
use engine::world::World;
use std::collections::HashMap;

pub struct PassContext<'a> {
    pub views: HashMap<ResourceId, &'a wgpu::TextureView>,
    pub buffers: HashMap<ResourceId, &'a wgpu::Buffer>,
    pub bind_group: Option<&'a wgpu::BindGroup>,
    pub upstream: HashMap<NodeId, &'a wgpu::BindGroup>,
}

pub trait RenderPassDesc {
    fn name(&self) -> &'static str;

    fn reads(&self) -> Vec<ResourceId>;
    fn writes(&self) -> Vec<ResourceId>;

    fn layout_entries(&self) -> Vec<wgpu::BindGroupLayoutEntry> {
        vec![]
    }

    fn binding_resources(&self) -> Vec<BindingResource> {
        vec![]
    }

    fn samplers(&self) -> Vec<(u32, wgpu::SamplerDescriptor<'static>)> {
        vec![]
    }
}

pub trait RenderPass: RenderPassDesc + Send + Sync + 'static {
    fn bind_node_id(&mut self, node_id: NodeId);

    fn execute(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &PassContext<'_>,
        world: &World,
    );
}
