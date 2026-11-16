/// A GPU buffer replaced by a bigger one (contents lost) when it doesn't fit what a frame needs.
pub struct GrowableBuffer {
    label: &'static str,
    usage: wgpu::BufferUsages,
    buffer: Option<wgpu::Buffer>,
}

impl GrowableBuffer {
    pub fn new(label: &'static str, usage: wgpu::BufferUsages) -> Self {
        Self { label, usage, buffer: None }
    }

    /// The buffer, at least `size` bytes (rounded up to a power of two when replaced).
    pub fn fit(&mut self, device: &wgpu::Device, size: u64) -> &wgpu::Buffer {
        let size = size.max(16);
        if self.buffer.as_ref().is_none_or(|b| b.size() < size) {
            self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.label),
                size: size.next_power_of_two(),
                usage: self.usage,
                mapped_at_creation: false,
            }));
        }
        self.buffer.as_ref().expect("just created")
    }

    /// `None` before the first `fit`.
    pub fn get(&self) -> Option<&wgpu::Buffer> {
        self.buffer.as_ref()
    }

    /// The whole buffer at `binding`. Panics before the first `fit`.
    pub fn entry(&self, binding: u32) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry { binding, resource: self.get().expect("GrowableBuffer::fit first").as_entire_binding() }
    }
}
