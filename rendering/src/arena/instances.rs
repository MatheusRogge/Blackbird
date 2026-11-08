use std::ops::Range;

use super::geometry::MeshDraw;
use crate::profiler::DrawCounts;

const MIN_CAPACITY: usize = 1024;

/// One instanced draw: meshes sharing geometry and a pass-specific `key` (pipeline, material).
#[derive(Clone, Debug)]
pub struct Batch<K> {
    pub key: K,
    pub indices: Range<u32>,
    pub base_vertex: i32,
    /// Range of the pass's `InstanceList`, passed as the draw's instances.
    pub instances: Range<u32>,
    /// The smallest `order` of its meshes (e.g. distance, for front-to-back drawing).
    pub order: f32,
}

/// A pass's instance slots for instanced draws, written to a storage buffer the shaders index
/// with `instance_index`: `models[instance_ids[instance_index]]`. Instance slots of meshes sharing
/// geometry aren't contiguous in `GeometryArena`'s instance buffer, so each batch gets a
/// contiguous run here instead.
///
/// Per frame: `begin` (reserves room for every id the pass can push, so the buffer bound while
/// encoding is the one written), `batch` for each view, then `finish` to upload. Queue writes land
/// before the frame's command buffer runs, so batches can be encoded before `finish`.
#[derive(Default)]
pub struct InstanceList {
    ids: Vec<u32>,
    draws: u32,
    buffer: Option<wgpu::Buffer>,
    capacity: usize,
    generation: u64,
}

impl InstanceList {
    pub fn layout_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(4),
            },
            count: None,
        }
    }

    /// Clears the list and makes room for `max_ids`, replacing the buffer (and bumping
    /// `generation`) if it's too small.
    pub fn begin(&mut self, device: &wgpu::Device, max_ids: usize) {
        self.ids.clear();
        self.draws = 0;
        if self.buffer.is_none() || max_ids > self.capacity {
            self.capacity = max_ids.next_power_of_two().max(MIN_CAPACITY);
            self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("instance_ids"),
                size: (self.capacity * size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.generation += 1;
        }
    }

    /// Groups `draws` by key and geometry into instanced draws, appending their instance slots.
    /// Batches come out sorted by key; `order` only sets each batch's `Batch::order`.
    pub fn batch<K: Ord + Copy>(&mut self, mut draws: Vec<(K, f32, &MeshDraw)>) -> Vec<Batch<K>> {
        draws.sort_by_key(|(key, _, draw)| (*key, draw.geometry));
        let mut batches: Vec<Batch<K>> = Vec::new();
        for (key, order, draw) in draws {
            let next = self.ids.len() as u32;
            match batches.last_mut() {
                Some(b) if b.key == key && b.indices == draw.indices && b.base_vertex == draw.base_vertex => {
                    b.instances.end = next + 1;
                    b.order = b.order.min(order);
                }
                _ => batches.push(Batch {
                    key,
                    indices: draw.indices.clone(),
                    base_vertex: draw.base_vertex,
                    instances: next..next + 1,
                    order,
                }),
            }
            self.ids.push(draw.instance);
        }
        debug_assert!(self.ids.len() <= self.capacity, "InstanceList::begin reserved too little");
        self.draws += batches.len() as u32;
        batches
    }

    pub fn finish(&self, queue: &wgpu::Queue) {
        if let Some(buffer) = &self.buffer
            && !self.ids.is_empty()
        {
            let len = self.ids.len().min(self.capacity);
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&self.ids[..len]));
        }
    }

    /// What was batched since `begin`. Counts batches the pass may still skip (e.g. a missing
    /// texture).
    pub fn counts(&self) -> DrawCounts {
        DrawCounts { draws: self.draws, instances: self.ids.len() as u32 }
    }

    /// The id buffer; `None` before the first `begin`.
    pub fn buffer(&self) -> Option<&wgpu::Buffer> {
        self.buffer.as_ref()
    }

    /// Bumped whenever the buffer is replaced: rebuild bind groups on it.
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bounds::Aabb;

    fn draw(geometry: u32, instance: u32) -> MeshDraw {
        MeshDraw {
            base_vertex: geometry as i32 * 100,
            indices: geometry * 10..geometry * 10 + 6,
            instance,
            bounds: Aabb::empty(),
            geometry,
        }
    }

    #[test]
    fn groups_meshes_by_key_and_geometry() {
        let draws = [draw(1, 0), draw(2, 1), draw(1, 2), draw(1, 3)];
        let mut list = InstanceList { capacity: 16, ..Default::default() };
        let batches = list.batch(vec![(0, 5.0, &draws[0]), (0, 1.0, &draws[1]), (0, 3.0, &draws[2]), (7, 0.5, &draws[3])]);
        let summary: Vec<(i32, u32, Range<u32>, f32)> =
            batches.iter().map(|b| (b.key, b.indices.start, b.instances.clone(), b.order)).collect();
        assert_eq!(summary, [(0, 10, 0..2, 3.0), (0, 20, 2..3, 1.0), (7, 10, 3..4, 0.5)]);
        // The slots, in batch order: geometry 1's two meshes first.
        assert_eq!(list.ids, [0, 2, 1, 3]);
    }
}
