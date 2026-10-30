mod frame;
mod geometry;
mod upload;

pub use frame::{FrameArena, FrameSlice};
pub use geometry::{GeometryArena, MeshDraw};
pub use upload::UploadArena;

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
