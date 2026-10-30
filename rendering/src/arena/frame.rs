use std::cell::Cell;

use wgpu::util::DeviceExt;

use super::BufferRange;

const INITIAL_CAPACITY: u64 = 1024 * 1024;
const USAGE: wgpu::BufferUsages = wgpu::BufferUsages::VERTEX
    .union(wgpu::BufferUsages::INDEX)
    .union(wgpu::BufferUsages::UNIFORM)
    .union(wgpu::BufferUsages::STORAGE)
    .union(wgpu::BufferUsages::COPY_DST);

/// Data uploaded through the frame arena. Valid until the next frame.
pub struct FrameSlice {
    pub buffer: wgpu::Buffer,
    pub offset: u64,
    pub size: u64,
}

impl FrameSlice {
    pub fn range(&self) -> BufferRange<'_> {
        BufferRange {
            buffer: &self.buffer,
            offset: self.offset,
            size: self.size,
        }
    }
}

/// Linear allocator for data that changes every frame. One buffer is reused
/// across frames: queue writes are ordered after earlier submissions, and the
/// renderer keeps at most one frame in flight.
pub struct FrameArena {
    buffer: wgpu::Buffer,
    capacity: u64,
    cursor: Cell<u64>,
    high_water: u64,
    generation: u64,
}

impl FrameArena {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            buffer: Self::create(device, INITIAL_CAPACITY),
            capacity: INITIAL_CAPACITY,
            cursor: Cell::new(0),
            high_water: 0,
            generation: 0,
        }
    }

    fn create(device: &wgpu::Device, capacity: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame_arena"),
            size: capacity,
            usage: USAGE,
            mapped_at_creation: false,
        })
    }

    /// Resets the cursor. If the last frame overflowed, the buffer is replaced
    /// with one big enough for it and `generation` is bumped.
    pub fn begin_frame(&mut self, device: &wgpu::Device) {
        let used = self.cursor.replace(0);
        self.high_water = used;
        if used > self.capacity {
            self.capacity = used.next_power_of_two();
            self.buffer = Self::create(device, self.capacity);
            self.generation += 1;
        }
    }

    /// Uploads `data` at an offset aligned to `align`. When the arena is full
    /// the data goes to a one-off buffer and the arena grows next frame.
    pub fn alloc(&self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8], align: u64) -> FrameSlice {
        debug_assert!((data.len() as u64).is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT));
        let size = data.len() as u64;
        let offset = self.cursor.get().next_multiple_of(align.max(wgpu::COPY_BUFFER_ALIGNMENT));
        self.cursor.set(offset + size);

        if offset + size > self.capacity {
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("frame_arena_overflow"),
                contents: data,
                usage: USAGE,
            });
            return FrameSlice { buffer, offset: 0, size };
        }

        queue.write_buffer(&self.buffer, offset, data);
        FrameSlice {
            buffer: self.buffer.clone(),
            offset,
            size,
        }
    }

    /// Bytes allocated last frame and the current capacity.
    pub fn usage(&self) -> (u64, u64) {
        (self.high_water, self.capacity)
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }
}
