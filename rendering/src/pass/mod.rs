pub mod ao;
pub mod bvh_upload;
pub mod camera;
pub mod debug_overlay;
pub mod fullscreen;
pub mod cluster_assignment;
pub mod gbuffer;
pub mod light_upload;
pub mod lighting;
pub mod material;
pub mod mipmap;
pub mod particle_sim;
pub mod particles;
pub mod present;
pub mod probe_atlas;
pub mod probe_trace;
pub mod probe_update;
pub mod reflection_probe;
pub mod reflection_trace;
pub mod prev_frame;
pub mod ssr;
pub mod ssr_temporal;
pub mod shadow;
mod shadow_cull;
pub mod shadow_pages;
pub mod transparent;

use crate::{
    arena::{BufferRange, FrameArena, GeometryArena},
    graph::NodeId,
    resource::{BindingResource, ResourceId},
};
use engine_core::world::World;
use std::collections::HashMap;

pub struct PassContext<'a> {
    pub views: HashMap<ResourceId, &'a wgpu::TextureView>,
    pub textures: HashMap<ResourceId, &'a wgpu::Texture>,
    pub buffers: HashMap<ResourceId, BufferRange<'a>>,
    pub bind_group: Option<&'a wgpu::BindGroup>,
    pub upstream: HashMap<NodeId, &'a wgpu::BindGroup>,
    /// Size the graph renders at (screen textures, aspect ratio).
    pub surface_size: (u32, u32),
    /// Top-left pixel of the region of the output texture the graph draws into: passes writing
    /// the output set their viewport to it (`surface_size` from here). (0, 0) unless a host shows
    /// the scene in part of the window.
    pub output_origin: (u32, u32),
    pub geometry: &'a GeometryArena,
    pub frame: &'a FrameArena,
}

pub trait PassDesc {
    fn name(&self) -> &'static str;

    fn reads(&self) -> Vec<ResourceId>;
    fn writes(&self) -> Vec<ResourceId>;

    /// Resources read as an earlier frame left them, e.g. the output of a pass this one feeds.
    /// They reach `PassContext` but add no ordering edge (which would be a cycle) and don't extend
    /// lifetimes, so they must not be screen textures, which are aliased within the frame.
    fn history_reads(&self) -> Vec<ResourceId> {
        vec![]
    }

    fn layout_entries(&self) -> Vec<wgpu::BindGroupLayoutEntry> {
        vec![]
    }

    fn binding_resources(&self) -> Vec<BindingResource> {
        vec![]
    }

    fn samplers(&self) -> Vec<(u32, wgpu::SamplerDescriptor<'static>)> {
        vec![]
    }

    /// Run even though the frame output doesn't depend on this pass's writes (e.g. it only fills
    /// a texture a pass reads next frame). Otherwise the graph culls it.
    fn has_side_effects(&self) -> bool {
        false
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

    /// The mesh draws of the last `execute`, for `FrameStats`. `None` for passes that don't count.
    fn draw_counts(&self) -> Option<crate::profiler::DrawCounts> {
        None
    }
}
