use std::collections::{HashMap, HashSet};
use std::ops::Range;

use engine_core::{
    entity::EntityKey,
    world::{EntityHandle, World},
};
use offset_allocator::{Allocation, Allocator};

use crate::mesh::{Mesh, Vertex};

const INITIAL_VERTICES: u32 = 256 * 1024;
const INITIAL_INDICES: u32 = 1024 * 1024;

/// Where a mesh lives inside the shared geometry buffers.
#[derive(Clone, Debug)]
pub struct MeshDraw {
    pub base_vertex: i32,
    pub indices: Range<u32>,
}

struct MeshAlloc {
    vertex: Allocation,
    index: Allocation,
    draw: MeshDraw,
}

struct Pool {
    buffer: wgpu::Buffer,
    allocator: Allocator,
    capacity: u32,
    used: u32,
    stride: u64,
}

impl Pool {
    fn new(device: &wgpu::Device, label: &str, usage: wgpu::BufferUsages, stride: u64, capacity: u32) -> Self {
        Self {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: capacity as u64 * stride,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            allocator: Allocator::new(capacity),
            capacity,
            used: 0,
            stride,
        }
    }

    fn alloc(&mut self, queue: &wgpu::Queue, data: &[u8]) -> Option<Allocation> {
        let count = (data.len() as u64 / self.stride) as u32;
        let allocation = self.allocator.allocate(count)?;
        self.used += self.allocator.allocation_size(allocation);
        queue.write_buffer(&self.buffer, allocation.offset as u64 * self.stride, data);
        Some(allocation)
    }

    fn free(&mut self, allocation: Allocation) {
        self.used -= self.allocator.allocation_size(allocation);
        self.allocator.free(allocation);
    }
}

/// Vertex and index data for every `Mesh` in the world, sub-allocated from one
/// vertex buffer and one index buffer that grow on demand. Draw with
/// `draw_indexed(draw.indices, draw.base_vertex, ..)` after binding both buffers.
pub struct GeometryArena {
    vertices: Pool,
    indices: Pool,
    meshes: HashMap<EntityKey, MeshAlloc>,
    generation: u64,
}

impl GeometryArena {
    pub fn new(device: &wgpu::Device) -> Self {
        Self::with_capacity(device, INITIAL_VERTICES, INITIAL_INDICES)
    }

    fn with_capacity(device: &wgpu::Device, vertices: u32, indices: u32) -> Self {
        Self {
            vertices: Pool::new(
                device,
                "geometry_vertices",
                wgpu::BufferUsages::VERTEX,
                size_of::<Vertex>() as u64,
                vertices,
            ),
            indices: Pool::new(
                device,
                "geometry_indices",
                wgpu::BufferUsages::INDEX,
                size_of::<u32>() as u64,
                indices,
            ),
            meshes: HashMap::new(),
            generation: 0,
        }
    }

    /// Uploads meshes that are new since the last call and frees the ones whose
    /// entity is gone. Meshes edited in place keep their old GPU data.
    pub fn sync(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, world: &World) {
        let meshes = world.get_entity_handles::<Mesh>();
        let live: HashSet<EntityKey> = meshes.iter().map(|h| h.id).collect();

        let (vertices, indices) = (&mut self.vertices, &mut self.indices);
        self.meshes.retain(|key, alloc| {
            let keep = live.contains(key);
            if !keep {
                vertices.free(alloc.vertex);
                indices.free(alloc.index);
            }
            keep
        });

        for handle in &meshes {
            if self.meshes.contains_key(&handle.id) || handle.entity.indices.is_empty() {
                continue;
            }
            if !self.insert(queue, handle.id, handle.entity) {
                self.grow(device, queue, &meshes);
                return;
            }
        }
    }

    fn insert(&mut self, queue: &wgpu::Queue, key: EntityKey, mesh: &Mesh) -> bool {
        let Some(vertex) = self.vertices.alloc(queue, mesh.get_vertex_buffer_content()) else {
            return false;
        };
        let Some(index) = self.indices.alloc(queue, mesh.get_indices_buffer_content()) else {
            self.vertices.free(vertex);
            return false;
        };

        let draw = MeshDraw {
            base_vertex: vertex.offset as i32,
            indices: index.offset..index.offset + mesh.indices.len() as u32,
        };
        self.meshes.insert(key, MeshAlloc { vertex, index, draw });
        true
    }

    /// Rebuilds both buffers bigger and re-uploads every live mesh, which also
    /// compacts away fragmentation.
    fn grow(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, meshes: &[EntityHandle<'_, Mesh>]) {
        let needed_vertices: u64 = meshes.iter().map(|h| h.entity.vertices.len() as u64).sum();
        let needed_indices: u64 = meshes.iter().map(|h| h.entity.indices.len() as u64).sum();

        let mut vertex_cap = grown_capacity(self.vertices.capacity, needed_vertices);
        let mut index_cap = grown_capacity(self.indices.capacity, needed_indices);
        let generation = self.generation + 1;

        loop {
            *self = Self::with_capacity(device, vertex_cap, index_cap);
            self.generation = generation;

            let fits = meshes
                .iter()
                .filter(|h| !h.entity.indices.is_empty())
                .all(|h| self.insert(queue, h.id, h.entity));
            if fits {
                return;
            }

            if vertex_cap == u32::MAX && index_cap == u32::MAX {
                log::error!("geometry arena cannot fit the scene; some meshes are not drawn");
                return;
            }
            vertex_cap = vertex_cap.saturating_mul(2);
            index_cap = index_cap.saturating_mul(2);
        }
    }

    pub fn get(&self, key: EntityKey) -> Option<&MeshDraw> {
        self.meshes.get(&key).map(|alloc| &alloc.draw)
    }

    pub fn vertex_buffer(&self) -> &wgpu::Buffer {
        &self.vertices.buffer
    }

    pub fn index_buffer(&self) -> &wgpu::Buffer {
        &self.indices.buffer
    }

    /// Bumped every time the backing buffers are replaced.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn vertex_usage(&self) -> (u32, u32) {
        (self.vertices.used, self.vertices.capacity)
    }

    pub fn index_usage(&self) -> (u32, u32) {
        (self.indices.used, self.indices.capacity)
    }
}

/// Doubles, or jumps straight to `needed` plus allocator rounding slack.
fn grown_capacity(current: u32, needed: u64) -> u32 {
    let with_slack = needed + needed / 8;
    (current as u64 * 2).max(with_slack).min(u32::MAX as u64) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grows_at_least_double() {
        assert_eq!(grown_capacity(1024, 100), 2048);
    }

    #[test]
    fn grows_to_fit_large_request() {
        assert_eq!(grown_capacity(1024, 8000), 9000);
    }

    #[test]
    fn clamps_to_u32() {
        assert_eq!(grown_capacity(u32::MAX, 1), u32::MAX);
    }
}
