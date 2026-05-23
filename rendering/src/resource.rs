pub type ResourceId = u32;

#[derive(Clone, Hash, PartialEq, Eq)]
pub enum ResourceDescriptor {
    Buffer {
        size: u64,
        usage: wgpu::BufferUsages,
    },
    BufferInit {
        data: Vec<u8>,
        usage: wgpu::BufferUsages,
    },
    Texture {
        size: wgpu::Extent3d,
        usage: wgpu::TextureUsages,
        format: wgpu::TextureFormat,
    },
    ScreenTexture {
        usage: wgpu::TextureUsages,
        format: wgpu::TextureFormat,
    },
    /// Fixed-size texture that is never resized with the screen.
    FixedTexture {
        size: wgpu::Extent3d,
        usage: wgpu::TextureUsages,
        format: wgpu::TextureFormat,
    },
    /// Fixed-size 3D texture that is never resized.
    Fixed3DTexture {
        size: [u32; 3],
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    },
    ExternalView,
}

pub enum AllocatedResource {
    Buffer(wgpu::Buffer),
    Texture(wgpu::Texture, wgpu::TextureView),
    ExternalView(wgpu::TextureView),
}

pub struct GraphResource {
    pub version: u64,
    pub desc: ResourceDescriptor,
    pub resource: Option<AllocatedResource>,
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct BindingResource {
    pub slot: u32,
    pub resource_id: ResourceId,
    pub descriptor: ResourceDescriptor,
}
