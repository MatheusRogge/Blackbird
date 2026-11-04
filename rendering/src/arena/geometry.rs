use std::collections::{HashMap, HashSet};
use std::ops::Range;

use engine_core::{
    entity::EntityKey,
    world::{EntityHandle, World},
};
use bytemuck::Zeroable;
use offset_allocator::{Allocation, Allocator};

use crate::mesh::{Mesh, Vertex};

const INITIAL_VERTICES: u32 = 256 * 1024;
const INITIAL_INDICES: u32 = 1024 * 1024;
const INITIAL_INSTANCES: u32 = 1024;

/// Per-mesh entry of the instance buffer. Mirrors `Model` in `gbuffer.wgsl`
/// and `shadow.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuModel {
    pub model: [[f32; 4]; 4],
    /// Inverse-transpose of `model`, for normals.
    pub normal: [[f32; 4]; 4],
    /// x: alpha cutoff (0 = never discard), y: 1 if double-sided, zw: unused.
    pub material: [f32; 4],
}

/// Where a mesh lives inside the shared geometry buffers. `instance` indexes
/// the instance buffer and is passed as the draw's first instance.
#[derive(Clone, Debug)]
pub struct MeshDraw {
    pub base_vertex: i32,
    pub indices: Range<u32>,
    pub instance: u32,
}

impl MeshDraw {
    pub fn instances(&self) -> Range<u32> {
        self.instance..self.instance + 1
    }
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
/// vertex buffer and one index buffer that grow on demand, plus a storage
/// buffer of per-mesh model matrices rewritten every frame. Draw with
/// `draw_indexed(draw.indices, draw.base_vertex, draw.instances())` after binding
/// both buffers and the instance buffer.
pub struct GeometryArena {
    vertices: Pool,
    indices: Pool,
    instances: wgpu::Buffer,
    instance_capacity: u32,
    free_instances: Vec<u32>,
    next_instance: u32,
    meshes: HashMap<EntityKey, MeshAlloc>,
    generation: u64,
}

impl GeometryArena {
    pub fn new(device: &wgpu::Device) -> Self {
        Self::with_capacity(device, INITIAL_VERTICES, INITIAL_INDICES, INITIAL_INSTANCES)
    }

    pub fn instance_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(size_of::<GpuModel>() as u64),
            },
            count: None,
        }
    }

    fn create_instances(device: &wgpu::Device, capacity: u32) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("geometry_instances"),
            size: capacity as u64 * size_of::<GpuModel>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn with_capacity(device: &wgpu::Device, vertices: u32, indices: u32, instances: u32) -> Self {
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
            instances: Self::create_instances(device, instances),
            instance_capacity: instances,
            free_instances: Vec::new(),
            next_instance: 0,
            meshes: HashMap::new(),
            generation: 0,
        }
    }

    /// Uploads meshes that are new since the last call, frees the ones whose
    /// entity is gone and rewrites every mesh's transform. Vertices edited in
    /// place keep their old GPU data.
    pub fn sync(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, world: &World) {
        let meshes = world.get_entity_handles::<Mesh>();
        let live: HashSet<EntityKey> = meshes.iter().map(|h| h.id).collect();

        let (vertices, indices, free_instances) =
            (&mut self.vertices, &mut self.indices, &mut self.free_instances);
        self.meshes.retain(|key, alloc| {
            let keep = live.contains(key);
            if !keep {
                vertices.free(alloc.vertex);
                indices.free(alloc.index);
                free_instances.push(alloc.draw.instance);
            }
            keep
        });

        for handle in &meshes {
            if self.meshes.contains_key(&handle.id) || handle.entity.indices.is_empty() {
                continue;
            }
            if !self.insert(queue, handle.id, handle.entity) {
                self.grow(device, queue, &meshes);
                break;
            }
        }

        self.write_instances(device, queue, &meshes);
    }

    fn write_instances(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, meshes: &[EntityHandle<'_, Mesh>]) {
        if self.next_instance > self.instance_capacity {
            self.instance_capacity = self.next_instance.next_power_of_two();
            self.instances = Self::create_instances(device, self.instance_capacity);
            self.generation += 1;
        }

        let mut models = vec![GpuModel::zeroed(); self.next_instance as usize];
        for handle in meshes {
            if let Some(alloc) = self.meshes.get(&handle.id) {
                let mesh = handle.entity;
                let transform = mesh.transform;
                models[alloc.draw.instance as usize] = GpuModel {
                    model: transform.into(),
                    normal: transform.inversed().transposed().into(),
                    material: [
                        mesh.alpha_cutoff.unwrap_or(0.0),
                        if mesh.double_sided { 1.0 } else { 0.0 },
                        0.0,
                        0.0,
                    ],
                };
            }
        }

        if !models.is_empty() {
            queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(&models));
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

        let instance = self.free_instances.pop().unwrap_or_else(|| {
            self.next_instance += 1;
            self.next_instance - 1
        });
        let draw = MeshDraw {
            base_vertex: vertex.offset as i32,
            indices: index.offset..index.offset + mesh.indices.len() as u32,
            instance,
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
        let instance_cap = self.instance_capacity;

        loop {
            *self = Self::with_capacity(device, vertex_cap, index_cap, instance_cap);
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

    pub fn instance_buffer(&self) -> &wgpu::Buffer {
        &self.instances
    }

    /// Bumped every time a backing buffer is replaced. Rebuild bind groups that
    /// reference `instance_buffer()` when it changes.
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
