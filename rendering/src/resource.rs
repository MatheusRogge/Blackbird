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
