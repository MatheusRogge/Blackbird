//! GPU culling of the opaque shadow casters for the local shadow pages `shadow` renders in a frame
//! (`shadow_cull.wgsl`, embedded). Each page tests every caster's world bounds against its frustum
//! in a compute pass, which writes the visible instance slots and one compacted indirect command
//! per group of casters sharing geometry; the page is then one `multi_draw_indexed_indirect_count`.
//! Culling, batching and recording ~35 000 draws a frame on the CPU was most of animated Osaka's
//! `shadow` time.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};

use crate::{
    arena::{GrowableBuffer, MeshDraw},
    bounds::{Aabb, Frustum},
};

const SHADER: &str = include_str!("shadow_cull.wgsl");
/// Mirrors the workgroup size of both entry points.
const WORKGROUP: u32 = 64;
const DRAW_ARGS_SIZE: u64 = std::mem::size_of::<wgpu::util::DrawIndexedIndirectArgs>() as u64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuCaster {
    min: [f32; 3],
    instance: u32,
    max: [f32; 3],
    group: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuGroup {
    index_count: u32,
    first_index: u32,
    base_vertex: i32,
    offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    casters: u32,
    groups: u32,
    pages: u32,
    _pad: u32,
}

pub(super) struct PageCuller {
    cull: wgpu::ComputePipeline,
    commands: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    casters: GrowableBuffer,
    groups: GrowableBuffer,
    planes: GrowableBuffer,
    counts: GrowableBuffer,
    instance_ids: GrowableBuffer,
    draws: GrowableBuffer,
    draw_counts: GrowableBuffer,
    /// Groups (draw commands) per page this frame; `None` when nothing was culled.
    page_groups: Option<u32>,
}

impl PageCuller {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow_cull"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_cull"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1, true),
                storage(2, true),
                storage(3, true),
                storage(4, false),
                storage(5, false),
                storage(6, false),
                storage(7, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_cull"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        use wgpu::BufferUsages as U;
        Self {
            cull: pipeline("cs_cull"),
            commands: pipeline("cs_commands"),
            layout,
            params: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("shadow_cull_params"),
                size: std::mem::size_of::<Params>() as u64,
                usage: U::UNIFORM | U::COPY_DST,
                mapped_at_creation: false,
            }),
            casters: GrowableBuffer::new("shadow_cull_casters", U::STORAGE | U::COPY_DST),
            groups: GrowableBuffer::new("shadow_cull_groups", U::STORAGE | U::COPY_DST),
            planes: GrowableBuffer::new("shadow_cull_planes", U::STORAGE | U::COPY_DST),
            counts: GrowableBuffer::new("shadow_cull_counts", U::STORAGE | U::COPY_DST),
            instance_ids: GrowableBuffer::new("shadow_cull_instance_ids", U::STORAGE),
            draws: GrowableBuffer::new("shadow_cull_draws", U::STORAGE | U::INDIRECT),
            draw_counts: GrowableBuffer::new("shadow_cull_draw_counts", U::STORAGE | U::INDIRECT | U::COPY_DST),
            page_groups: None,
        }
    }

    /// Records the culling of `casters` (world bounds and draw) against every frustum in `pages`,
    /// for `draw`. Run before the render pass that draws the pages.
    pub(super) fn cull<'d>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        casters: impl Iterator<Item = (Aabb, &'d MeshDraw)>,
        pages: &[Frustum],
    ) {
        self.page_groups = None;
        let mut group_of: HashMap<u32, u32> = HashMap::new();
        let mut groups: Vec<GpuGroup> = Vec::new();
        let mut gpu_casters: Vec<GpuCaster> = Vec::new();
        for (bounds, draw) in casters.filter(|(b, _)| !b.is_empty()) {
            let group = *group_of.entry(draw.geometry).or_insert_with(|| {
                groups.push(GpuGroup {
                    index_count: draw.indices.len() as u32,
                    first_index: draw.indices.start,
                    base_vertex: draw.base_vertex,
                    offset: 0,
                });
                groups.len() as u32 - 1
            });
            groups[group as usize].offset += 1;
            gpu_casters.push(GpuCaster { min: bounds.min.into(), instance: draw.instance, max: bounds.max.into(), group });
        }
        if gpu_casters.is_empty() || pages.is_empty() {
            return;
        }
        // Group sizes into where each group's slots start.
        let mut next = 0;
        for group in &mut groups {
            next += std::mem::replace(&mut group.offset, next);
        }
        let planes: Vec<[f32; 4]> = pages.iter().flat_map(|f| *f.planes()).collect();
        let (casters_n, groups_n, pages_n) = (gpu_casters.len() as u64, groups.len() as u64, pages.len() as u64);
        let params = Params { casters: casters_n as u32, groups: groups_n as u32, pages: pages_n as u32, _pad: 0 };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        queue.write_buffer(self.casters.fit(device, casters_n * 32), 0, bytemuck::cast_slice(&gpu_casters));
        queue.write_buffer(self.groups.fit(device, groups_n * 16), 0, bytemuck::cast_slice(&groups));
        queue.write_buffer(self.planes.fit(device, pages_n * 96), 0, bytemuck::cast_slice(&planes));
        let counts = self.counts.fit(device, pages_n * groups_n * 4);
        let draw_counts = self.draw_counts.fit(device, pages_n * 4);
        encoder.clear_buffer(counts, 0, Some(pages_n * groups_n * 4));
        encoder.clear_buffer(draw_counts, 0, Some(pages_n * 4));
        self.instance_ids.fit(device, pages_n * casters_n * 4);
        self.draws.fit(device, pages_n * groups_n * DRAW_ARGS_SIZE);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_cull"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params.as_entire_binding() },
                self.casters.entry(1),
                self.groups.entry(2),
                self.planes.entry(3),
                self.counts.entry(4),
                self.instance_ids.entry(5),
                self.draws.entry(6),
                self.draw_counts.entry(7),
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("shadow_cull"), timestamp_writes: None });
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_pipeline(&self.cull);
        pass.dispatch_workgroups((casters_n as u32).div_ceil(WORKGROUP), pages_n as u32, 1);
        pass.set_pipeline(&self.commands);
        pass.dispatch_workgroups((groups_n as u32).div_ceil(WORKGROUP), pages_n as u32, 1);
        drop(pass);
        self.page_groups = Some(groups_n as u32);
    }

    /// The instance slots `cull` wrote, for the draws' `instance_ids` binding. `None` when nothing
    /// was culled this frame.
    pub(super) fn instance_ids(&self) -> Option<&wgpu::Buffer> {
        self.page_groups.and(self.instance_ids.get())
    }

    /// Draws the casters visible from page `index` of `cull`'s `pages`, with the pipeline and
    /// bind groups set (group 1's instance ids being `instance_ids`).
    pub(super) fn draw(&self, pass: &mut wgpu::RenderPass<'_>, index: usize) {
        let (Some(groups), Some(draws), Some(counts)) = (self.page_groups, self.draws.get(), self.draw_counts.get()) else {
            return;
        };
        let offset = index as u64 * u64::from(groups) * DRAW_ARGS_SIZE;
        pass.multi_draw_indexed_indirect_count(draws, offset, counts, index as u64 * 4, groups);
    }
}
