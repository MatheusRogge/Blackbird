use std::sync::Arc;

use texture_pool::AliasingReport;

mod frame;
mod geometry;
mod upload;
pub mod texture_pool;

pub use frame::{FrameArena, FrameSlice};
pub use geometry::{GeometryArena, MeshDraw};
pub use upload::UploadArena;

/// Occupancy of the graph's arenas, published with `FrameStats`.
#[derive(Debug, Clone, Default)]
pub struct ArenaStats {
    pub upload_bytes: u64,
    pub upload_pages: usize,
    /// (used, capacity) in vertices.
    pub geometry_vertices: (u32, u32),
    /// (used, capacity) in indices.
    pub geometry_indices: (u32, u32),
    pub geometry_generation: u64,
    /// (used this frame, capacity) in bytes.
    pub frame_bytes: (u64, u64),
    pub frame_generation: u64,
    pub screen_textures: Arc<AliasingReport>,
}

/// A byte range inside a GPU buffer. Packed resources share their backing
/// buffer with others, so always bind, write and slice through the range.
#[derive(Clone, Copy)]
pub struct BufferRange<'a> {
    pub buffer: &'a wgpu::Buffer,
    pub offset: u64,
    pub size: u64,
}

impl<'a> BufferRange<'a> {
    pub fn binding(&self) -> wgpu::BindingResource<'a> {
        wgpu::BindingResource::Buffer(self.buffer_binding())
    }

    pub fn buffer_binding(&self) -> wgpu::BufferBinding<'a> {
        wgpu::BufferBinding {
            buffer: self.buffer,
            offset: self.offset,
            size: wgpu::BufferSize::new(self.size),
        }
    }

    pub fn write(&self, queue: &wgpu::Queue, data: &[u8]) {
        self.write_at(queue, 0, data);
    }

    pub fn write_at(&self, queue: &wgpu::Queue, offset: u64, data: &[u8]) {
        debug_assert!(offset + data.len() as u64 <= self.size, "write past end of buffer range");
        queue.write_buffer(self.buffer, self.offset + offset, data);
    }

    pub fn slice(&self) -> wgpu::BufferSlice<'a> {
        self.buffer.slice(self.offset..self.offset + self.size)
    }
}
