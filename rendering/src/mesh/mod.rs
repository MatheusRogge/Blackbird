use std::sync::Arc;

use engine_core::entity::Entity;

pub use ultraviolet::Vec3;
use wgpu::{
    Buffer, BufferUsages, Device,
    util::{BufferInitDescriptor, DeviceExt},
};

use crate::texture::TextureAsset;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
    pub uv: [f32; 2],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 4] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x3, 3 => Float32x2];

    pub fn buffer_descriptor() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            attributes: &Self::ATTRIBS,
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct Mesh {
    pub indices: Vec<u32>,
    pub vertices: Vec<Vertex>,
    pub albedo_texture: Option<Arc<TextureAsset>>,
}

impl Mesh {
    pub fn new(
        vertices: Vec<Vertex>,
        indices: Vec<u32>,
        albedo_texture: Option<Arc<TextureAsset>>,
    ) -> Self {
        Self {
            indices,
            vertices,
            albedo_texture,
        }
    }

    pub fn get_indices_count(&self) -> usize {
        self.indices.len()
    }

    pub fn get_vertices_count(&self) -> usize {
        self.vertices.len()
    }

    pub fn get_vertex_buffer_content(&self) -> &[u8] {
        bytemuck::cast_slice(self.vertices.as_slice())
    }

    pub fn get_indices_buffer_content(&self) -> &[u8] {
        bytemuck::cast_slice(self.indices.as_slice())
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

/// A snapshot of all meshes packed into a single flat batch for upload to the GPU.
///
/// The application populates this each frame (or whenever meshes change) and the
/// `MeshPass` uploads it to a single vertex + index buffer, issuing one draw call.
#[derive(Clone, Default)]
pub struct MeshBatch {
    pub meshes: Vec<Mesh>,
    pub viewport_width: u32,
    pub viewport_height: u32,
}

impl MeshBatch {
    pub fn is_empty(&self) -> bool {
        self.meshes.is_empty()
    }

    /// Returns all vertices from all meshes concatenated in order.
    pub fn packed_vertices(&self) -> Vec<Vertex> {
        self.meshes
            .iter()
            .flat_map(|m| m.vertices.iter().copied())
            .collect()
    }

    /// Returns all indices with per-mesh base-vertex offsets applied, so the
    /// entire batch can be drawn with a single `draw_indexed` call.
    pub fn packed_indices(&self) -> Vec<u32> {
        let mut out = Vec::new();
        let mut base: u32 = 0;
        for mesh in &self.meshes {
            for &idx in &mesh.indices {
                out.push(idx + base);
            }
            base += mesh.vertices.len() as u32;
        }
        out
    }

    pub fn total_vertex_count(&self) -> usize {
        self.meshes.iter().map(|m| m.vertices.len()).sum()
    }

    pub fn total_index_count(&self) -> usize {
        self.meshes.iter().map(|m| m.indices.len()).sum()
    }
}
