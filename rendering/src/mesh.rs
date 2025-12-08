use engine::entity::Entity;

pub use ultraviolet::Vec3;
use wgpu::{
    Buffer, BufferUsages, Device,
    util::{BufferInitDescriptor, DeviceExt},
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

    pub fn buffer_descriptor() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            attributes: &Self::ATTRIBS,
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
        }
    }
}

#[derive(Debug, Default)]
pub struct Mesh {
    indices: Vec<u16>,
    vertices: Vec<Vertex>,
}

impl Mesh {
    pub fn new(vertices: Vec<Vertex>) -> Self {
        Self {
            vertices,
            ..Default::default()
        }
    }

    pub fn push_indices(&mut self, indices: Vec<u16>) {
        self.indices = indices;
    }

    pub fn get_indices_count(&self) -> usize {
        self.indices.len()
    }

    pub fn get_vertices_count(&self) -> usize {
        self.vertices.len()
    }

    pub fn get_vertex_buffer(&self, device: &Device) -> Buffer {
        device.create_buffer_init(&BufferInitDescriptor {
            label: Some("vertex_buffer"),
            usage: BufferUsages::VERTEX,
            contents: bytemuck::cast_slice(self.vertices.as_slice()),
        })
    }

    pub fn get_index_buffer(&self, device: &Device) -> Buffer {
        device.create_buffer_init(&BufferInitDescriptor {
            label: Some("index_buffer"),
            usage: BufferUsages::INDEX,
            contents: bytemuck::cast_slice(self.indices.as_slice()),
        })
    }
}

impl Entity for Mesh {}
