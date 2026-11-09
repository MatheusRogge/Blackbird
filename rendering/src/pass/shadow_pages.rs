//! Page requests for local light virtual shadow maps (`light::virtual_shadow`): a compute pass
//! (`shadow_pages.wgsl`, embedded) over the G-buffer depth sets a bit per virtual page that visible
//! pixels need, then copies the mask into a mappable buffer. Buffers cycle through a small ring:
//! one frame records the copy, the next maps it (a buffer can't be mapped before the copy is
//! submitted), and a later one reads it once mapped, so the render thread never waits. The shadow
//! pass applies the latest mask, a few frames old, through `SharedPageRequests`.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use engine_core::world::World;

use crate::{
    graph::NodeId,
    light::virtual_shadow::{REQUEST_WORDS, SharedPageRequests},
    pass::{Pass, PassContext, PassDesc},
    resource::ResourceId,
};

const SHADER: &str = include_str!("shadow_pages.wgsl");
const MASK_BYTES: u64 = (REQUEST_WORDS * std::mem::size_of::<u32>()) as u64;
/// Mirrors the workgroup size of `cs_main`, over half-resolution pixels.
const WORKGROUP: u32 = 8;
/// Enough that one buffer is always free while others wait to be mapped and read.
const RING: usize = 3;

#[derive(Default)]
enum State {
    #[default]
    Free,
    /// The copy is recorded in this frame's commands, not yet submitted.
    Copied,
    /// `map_async` called; the flag turns true once it can be read.
    Mapping(Arc<AtomicBool>),
}

struct Gpu {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    mask: wgpu::Buffer,
    ring: Vec<(wgpu::Buffer, State)>,
    /// Keyed by the depth view address.
    bind_group: Option<(usize, wgpu::BindGroup)>,
}

pub struct ShadowPagesPass {
    node_id: Option<NodeId>,
    depth_id: ResourceId,
    camera_buffer_id: ResourceId,
    local_params_id: ResourceId,
    published: SharedPageRequests,
    gpu: Option<Gpu>,
}

impl ShadowPagesPass {
    pub fn new(
        depth_id: ResourceId,
        camera_buffer_id: ResourceId,
        local_params_id: ResourceId,
        published: SharedPageRequests,
    ) -> Self {
        Self { node_id: None, depth_id, camera_buffer_id, local_params_id, published, gpu: None }
    }

    fn create_gpu(device: &wgpu::Device) -> Gpu {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow_pages"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let compute = wgpu::ShaderStages::COMPUTE;
        let uniform = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_pages"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: compute,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Depth,
                    },
                    count: None,
                },
                uniform(1),
                uniform(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: compute,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_pages"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shadow_pages"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffer = |label, usage| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: MASK_BYTES, usage, mapped_at_creation: false })
        };
        Gpu {
            pipeline,
            layout,
            mask: buffer(
                "shadow_page_requests",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            ),
            ring: (0..RING)
                .map(|_| {
                    let usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ;
                    (buffer("shadow_page_requests_readback", usage), State::Free)
                })
                .collect(),
            bind_group: None,
        }
    }
}

impl PassDesc for ShadowPagesPass {
    fn name(&self) -> &'static str {
        "shadow_pages"
    }

    fn reads(&self) -> Vec<ResourceId> {
        vec![self.depth_id, self.camera_buffer_id, self.local_params_id]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![]
    }

    fn has_side_effects(&self) -> bool {
        true
    }
}

impl Pass for ShadowPagesPass {
    fn bind_node_id(&mut self, node_id: NodeId) {
        self.node_id = Some(node_id);
    }

    fn execute(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &PassContext<'_>,
        _world: &World,
    ) {
        let Some(&depth) = ctx.views.get(&self.depth_id) else { return };
        let (Some(camera), Some(params)) = (ctx.buffers.get(&self.camera_buffer_id), ctx.buffers.get(&self.local_params_id)) else {
            return;
        };
        let gpu = self.gpu.get_or_insert_with(|| Self::create_gpu(device));

        for (buffer, state) in &mut gpu.ring {
            match std::mem::take(state) {
                State::Mapping(ready) if ready.load(Ordering::Acquire) => {
                    let mask = {
                        let data = buffer.slice(..).get_mapped_range().expect("page requests not mapped");
                        bytemuck::cast_slice::<u8, u32>(&data).to_vec()
                    };
                    buffer.unmap();
                    *self.published.lock().expect("page requests poisoned") = Some(mask);
                }
                State::Copied => {
                    let ready = Arc::new(AtomicBool::new(false));
                    let flag = Arc::clone(&ready);
                    buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                        if result.is_ok() {
                            flag.store(true, Ordering::Release);
                        }
                    });
                    *state = State::Mapping(ready);
                }
                other => *state = other,
            }
        }
        let _ = device.poll(wgpu::PollType::Poll);

        let Some((buffer, state)) = gpu.ring.iter_mut().find(|(_, s)| matches!(s, State::Free)) else { return };
        let key = std::ptr::from_ref(depth) as usize;
        if gpu.bind_group.as_ref().is_none_or(|(k, _)| *k != key) {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shadow_pages"),
                layout: &gpu.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth) },
                    wgpu::BindGroupEntry { binding: 1, resource: camera.binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: params.binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: gpu.mask.as_entire_binding() },
                ],
            });
            gpu.bind_group = Some((key, bind_group));
        }
        let Some((_, bind_group)) = &gpu.bind_group else { return };

        encoder.clear_buffer(&gpu.mask, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("shadow_pages"), timestamp_writes: None });
            pass.set_pipeline(&gpu.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            let (width, height) = ctx.surface_size;
            pass.dispatch_workgroups(width.div_ceil(2 * WORKGROUP), height.div_ceil(2 * WORKGROUP), 1);
        }
        encoder.copy_buffer_to_buffer(&gpu.mask, 0, buffer, 0, MASK_BYTES);
        *state = State::Copied;
    }
}
