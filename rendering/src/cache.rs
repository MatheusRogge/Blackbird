use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use crate::resource::ResourceId;

#[derive(Hash, PartialEq, Eq)]
pub struct PipelineLayoutKey {
    /// Stable pointer address of each `BindGroupLayout` slot (`None` = empty slot).
    /// Two passes holding the same `Arc<BindGroupLayout>` will share a cache entry.
    pub bind_group_layouts: Vec<Option<u64>>,
    /// Mirrors `PipelineLayoutDescriptor::immediate_size`.
    pub immediate_size: u32,
}

/// Build a [`PipelineLayoutKey`] from the given slices without requiring `Hash`
/// impls on the wgpu types.
pub fn pipeline_layout_key(
    bind_group_layouts: &[Option<&wgpu::BindGroupLayout>],
    immediate_size: u32,
) -> PipelineLayoutKey {
    PipelineLayoutKey {
        immediate_size,
        bind_group_layouts: bind_group_layouts
            .iter()
            .map(|opt| opt.map(|l| l as *const wgpu::BindGroupLayout as u64))
            .collect(),
    }
}

#[derive(Default)]
pub struct PipelineLayoutCache {
    cache: HashMap<PipelineLayoutKey, Arc<wgpu::PipelineLayout>>,
}

impl PipelineLayoutCache {
    pub fn get_or_create(
        &mut self,
        key: PipelineLayoutKey,
        device: &wgpu::Device,
        bind_group_layouts: &[Option<&wgpu::BindGroupLayout>],
        immediate_size: u32,
    ) -> Arc<wgpu::PipelineLayout> {
        self.cache
            .entry(key)
            .or_insert_with(|| {
                Arc::new(
                    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts,
                        immediate_size,
                    }),
                )
            })
            .clone()
    }
}

#[derive(Hash, PartialEq, Eq)]
pub struct PipelineKey {
    pub vertex_shader: u64,
    pub fragment_shader: u64,
    pub color_formats: Vec<wgpu::TextureFormat>,
    pub depth_format: Option<wgpu::TextureFormat>,
    pub topology: wgpu::PrimitiveTopology,
}

#[derive(Default)]
pub struct PipelineCache {
    cache: HashMap<PipelineKey, Arc<wgpu::RenderPipeline>>,
}

impl PipelineCache {
    pub fn get_or_create(
        &mut self,
        key: PipelineKey,
        device: &wgpu::Device,
        desc: &wgpu::RenderPipelineDescriptor,
    ) -> Arc<wgpu::RenderPipeline> {
        self.cache
            .entry(key)
            .or_insert_with(|| Arc::new(device.create_render_pipeline(desc)))
            .clone()
    }
}

#[derive(Hash, PartialEq, Eq, Clone)]
pub enum BindingKey {
    Texture(ResourceId),
    Buffer {
        id: ResourceId,
        offset: u64,
        size: u64,
    },
    Sampler(u64),
}

#[derive(Hash, PartialEq, Eq)]
pub struct BindGroupKey {
    pub layout_id: ResourceId,
    pub entries: Vec<BindingKey>,
}

#[derive(Default)]
pub struct BindGroupCache {
    cache: HashMap<u64, Arc<wgpu::BindGroup>>,
}

impl BindGroupCache {
    pub fn get_or_create(
        &mut self,
        key: &BindGroupKey,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        entries: &[wgpu::BindGroupEntry],
    ) -> Arc<wgpu::BindGroup> {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);

        self.cache
            .entry(hasher.finish())
            .or_insert_with(|| {
                Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout,
                    entries,
                }))
            })
            .clone()
    }

    pub fn hash_key(key: &BindGroupKey) -> u64 {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    pub fn get_by_hash(&self, hash: u64) -> Option<Arc<wgpu::BindGroup>> {
        self.cache.get(&hash).cloned()
    }

    pub fn store(&mut self, hash: u64, group: Arc<wgpu::BindGroup>) {
        self.cache.entry(hash).or_insert(group);
    }
}
