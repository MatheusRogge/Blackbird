pub mod bvh_upload;
pub mod camera;
pub mod cluster_assignment;
pub mod gbuffer;
pub mod light_upload;
pub mod lighting;
pub mod present;
pub mod probe_atlas;
pub mod probe_trace;
pub mod probe_update;
pub mod shadow;

use crate::{
    graph::NodeId,
    resource::{BindingResource, ResourceId},
};
use engine_core::world::World;
use std::collections::HashMap;

pub struct PassContext<'a> {
    pub views: HashMap<ResourceId, &'a wgpu::TextureView>,
    pub textures: HashMap<ResourceId, &'a wgpu::Texture>,
    pub buffers: HashMap<ResourceId, &'a wgpu::Buffer>,
    pub bind_group: Option<&'a wgpu::BindGroup>,
    pub upstream: HashMap<NodeId, &'a wgpu::BindGroup>,
    pub surface_size: (u32, u32),
}

pub trait PassDesc {
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

pub trait Pass: PassDesc + Send + Sync + 'static {
    fn bind_node_id(&mut self, node_id: NodeId);

    fn on_resize(&mut self, _width: u32, _height: u32) {}

    fn execute(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &PassContext<'_>,
        world: &World,
    );
}
