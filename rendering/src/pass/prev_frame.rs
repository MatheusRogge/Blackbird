//! Keeps a copy of a screen texture for the next frame. `ssr` reads last frame's `composite` output
//! through it, so reflected surfaces show their own reflections (one more bounce per frame).

use std::sync::{Arc, Mutex};

use engine_core::world::World;

use crate::{
    graph::NodeId,
    pass::{Pass, PassContext, PassDesc},
    resource::ResourceId,
};

/// Shared handle to the copy. Readers that run before `PrevFramePass` in a frame see the
/// previous frame's content; `None` until the first copy. The generation changes whenever the
/// texture is recreated, so readers can key their bind groups on it.
#[derive(Clone, Default)]
pub struct FrameHistory(Arc<Mutex<Option<(u64, wgpu::TextureView)>>>);

impl FrameHistory {
    pub fn get(&self) -> Option<(u64, wgpu::TextureView)> {
        self.0.lock().expect("frame history poisoned").clone()
    }
}

struct Target {
    size: (u32, u32),
    texture: wgpu::Texture,
}

pub struct PrevFramePass {
    node_id: Option<NodeId>,
    source_id: ResourceId,
    format: wgpu::TextureFormat,
    history: FrameHistory,
    target: Option<Target>,
    generation: u64,
}

impl PrevFramePass {
    /// `source_id` must be a screen texture of `format` with `COPY_SRC` usage. Readers get a clone
    /// of `history`, created before them since they usually run earlier in the frame.
    pub fn new(source_id: ResourceId, format: wgpu::TextureFormat, history: FrameHistory) -> Self {
        Self { node_id: None, source_id, format, history, target: None, generation: 0 }
    }
}

impl PassDesc for PrevFramePass {
    fn name(&self) -> &'static str {
        "prev_frame"
    }

    fn reads(&self) -> Vec<ResourceId> {
        vec![self.source_id]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![]
    }

    fn has_side_effects(&self) -> bool {
        true
    }
}

impl Pass for PrevFramePass {
    fn bind_node_id(&mut self, node_id: NodeId) {
        self.node_id = Some(node_id);
    }

    fn execute(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &PassContext<'_>,
        _world: &World,
    ) {
        let Some(&source) = ctx.textures.get(&self.source_id) else { return };
        if self.target.as_ref().is_none_or(|t| t.size != ctx.surface_size) {
            let (width, height) = ctx.surface_size;
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("prev_frame"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.generation += 1;
            *self.history.0.lock().expect("frame history poisoned") =
                Some((self.generation, texture.create_view(&wgpu::TextureViewDescriptor::default())));
            self.target = Some(Target { size: ctx.surface_size, texture });
        }
        let Some(target) = &self.target else { return };
        let (width, height) = target.size;
        encoder.copy_texture_to_texture(
            source.as_image_copy(),
            target.texture.as_image_copy(),
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
    }
}
