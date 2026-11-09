//! Reads the visible surfaces back to the CPU for `light::shadow_lod`: reduces the G-buffer depth
//! to the nearest and farthest depth per 16 × 16 pixel tile (`depth_tiles.wgsl`, embedded), copies
//! that into a mappable buffer, and later turns it into world points with the camera it was seen
//! from. Buffers cycle through a small ring: one frame records the copy, the next maps it (a
//! buffer can't be mapped before the copy is submitted), and a later one reads it once mapped, so
//! the render thread never waits. The samples it publishes are a few frames old.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use engine_core::world::World;
use ultraviolet::{Mat4, Vec3, Vec4};

use crate::{
    camera::Camera,
    graph::NodeId,
    light::shadow_lod::{SharedSurfaceSamples, SurfaceSamples, ViewSamples},
    pass::{Pass, PassContext, PassDesc},
    resource::ResourceId,
};

const SHADER: &str = include_str!("depth_tiles.wgsl");
/// Mirrors `TILE` in `depth_tiles.wgsl`.
const TILE: u32 = 16;
const TILE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Float;
const TEXEL_BYTES: u32 = 8;
/// Enough that one buffer is always free while others wait to be mapped and read.
const RING: usize = 3;

/// The camera a readback was recorded with.
#[derive(Copy, Clone)]
struct View {
    inv_view_proj: Mat4,
    eye: Vec3,
    forward: Vec3,
    /// World size of a pixel per metre of depth.
    pixel_per_depth: f32,
    surface: (u32, u32),
}

#[derive(Default)]
enum State {
    #[default]
    Free,
    /// The copy is recorded in this frame's commands, not yet submitted.
    Copied(View),
    /// `map_async` called; `ready` turns true once it can be read.
    Mapping(View, Arc<AtomicBool>),
}

struct Slot {
    buffer: wgpu::Buffer,
    state: State,
}

struct Target {
    tiles: (u32, u32),
    view: wgpu::TextureView,
    texture: wgpu::Texture,
    /// Keyed by the depth view address.
    bind_group: Option<(usize, wgpu::BindGroup)>,
    ring: Vec<Slot>,
}

pub struct DepthReadbackPass {
    node_id: Option<NodeId>,
    depth_id: ResourceId,
    published: SharedSurfaceSamples,
    pipeline: Option<(wgpu::RenderPipeline, wgpu::BindGroupLayout)>,
    target: Option<Target>,
}

impl DepthReadbackPass {
    pub fn new(depth_id: ResourceId, published: SharedSurfaceSamples) -> Self {
        Self { node_id: None, depth_id, published, pipeline: None, target: None }
    }

    fn create_pipeline(device: &wgpu::Device) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("depth_tiles"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("depth_tiles"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Depth,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("depth_tiles"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("depth_tiles"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_tiles"),
                targets: &[Some(TILE_FORMAT.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        (pipeline, layout)
    }

    fn create_target(device: &wgpu::Device, (width, height): (u32, u32)) -> Target {
        let tiles = (width.div_ceil(TILE).max(1), height.div_ceil(TILE).max(1));
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_tiles"),
            size: wgpu::Extent3d { width: tiles.0, height: tiles.1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TILE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let ring = (0..RING)
            .map(|_| Slot {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("depth_tiles_readback"),
                    size: u64::from(padded_row(tiles.0) * tiles.1),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                state: State::Free,
            })
            .collect();
        Target {
            tiles,
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
            texture,
            bind_group: None,
            ring,
        }
    }
}

fn padded_row(tiles_x: u32) -> u32 {
    (tiles_x * TEXEL_BYTES).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
}

/// World points of a mapped readback: per tile, its nearest and (when distinct) farthest surface
/// at the tile's center, with the world size of a pixel there.
fn surface_points(data: &[u8], tiles: (u32, u32), view: &View) -> Vec<(Vec3, f32)> {
    let row = padded_row(tiles.0) as usize;
    let (width, height) = (view.surface.0 as f32, view.surface.1 as f32);
    let mut points = Vec::with_capacity((tiles.0 * tiles.1 * 2) as usize);
    for y in 0..tiles.1 {
        let texels: &[[f32; 2]] = bytemuck::cast_slice(&data[y as usize * row..][..(tiles.0 * TEXEL_BYTES) as usize]);
        let py = ((y * TILE) as f32 + TILE as f32 * 0.5).min(height - 0.5);
        for (x, &[nearest, farthest]) in texels.iter().enumerate() {
            let px = ((x as u32 * TILE) as f32 + TILE as f32 * 0.5).min(width - 0.5);
            let ndc = (px / width * 2.0 - 1.0, 1.0 - py / height * 2.0);
            for depth in [nearest, farthest] {
                if depth <= 0.0 || (depth == farthest && farthest == nearest) {
                    continue;
                }
                let h = view.inv_view_proj * Vec4::new(ndc.0, ndc.1, depth, 1.0);
                let p = h.xyz() / h.w;
                let along = (p - view.eye).dot(view.forward);
                points.push((p, along * view.pixel_per_depth));
            }
        }
    }
    points
}

impl PassDesc for DepthReadbackPass {
    fn name(&self) -> &'static str {
        "depth_readback"
    }

    fn reads(&self) -> Vec<ResourceId> {
        vec![self.depth_id]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![]
    }

    fn has_side_effects(&self) -> bool {
        true
    }
}

impl Pass for DepthReadbackPass {
    fn bind_node_id(&mut self, node_id: NodeId) {
        self.node_id = Some(node_id);
    }

    fn execute(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &PassContext<'_>,
        world: &World,
    ) {
        let Some(&depth) = ctx.views.get(&self.depth_id) else { return };
        let Some(camera) = world.get_entities::<Camera>().first().copied() else { return };
        let (pipeline, layout) = self.pipeline.get_or_insert_with(|| Self::create_pipeline(device));
        let tiles = (ctx.surface_size.0.div_ceil(TILE).max(1), ctx.surface_size.1.div_ceil(TILE).max(1));
        // A resize drops buffers still in flight; their callbacks find nothing to report.
        if self.target.as_ref().is_none_or(|t| t.tiles != tiles) {
            self.target = Some(Self::create_target(device, ctx.surface_size));
        }
        let Some(target) = self.target.as_mut() else { return };

        for slot in &mut target.ring {
            match std::mem::take(&mut slot.state) {
                State::Mapping(view, ready) if ready.load(Ordering::Acquire) => {
                    let points = {
                        let data = slot.buffer.slice(..).get_mapped_range().expect("depth readback not mapped");
                        surface_points(&data, target.tiles, &view)
                    };
                    slot.buffer.unmap();
                    *self.published.lock().expect("surface samples poisoned") = Some(SurfaceSamples {
                        samples: Arc::new(ViewSamples::from_points(points)),
                        eye: view.eye,
                        forward: view.forward,
                    });
                }
                State::Copied(view) => {
                    let ready = Arc::new(AtomicBool::new(false));
                    let flag = Arc::clone(&ready);
                    slot.buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                        if result.is_ok() {
                            flag.store(true, Ordering::Release);
                        }
                    });
                    slot.state = State::Mapping(view, ready);
                }
                state => slot.state = state,
            }
        }
        let _ = device.poll(wgpu::PollType::Poll);

        let Some(slot) = target.ring.iter_mut().find(|s| matches!(s.state, State::Free)) else { return };
        let depth_key = std::ptr::from_ref(depth) as usize;
        if target.bind_group.as_ref().is_none_or(|(k, _)| *k != depth_key) {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("depth_tiles"),
                layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth) }],
            });
            target.bind_group = Some((depth_key, bind_group));
        }
        let Some((_, bind_group)) = &target.bind_group else { return };
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("depth_tiles"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            target.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &slot.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row(target.tiles.0)),
                    rows_per_image: Some(target.tiles.1),
                },
            },
            wgpu::Extent3d { width: target.tiles.0, height: target.tiles.1, depth_or_array_layers: 1 },
        );

        let (width, height) = ctx.surface_size;
        let forward = (camera.target - camera.eye).normalized();
        slot.state = State::Copied(View {
            inv_view_proj: camera.view_proj_matrix(width as f32 / height.max(1) as f32).inversed(),
            eye: camera.eye,
            forward,
            pixel_per_depth: 2.0 * (camera.fovy * 0.5).tan() / height.max(1) as f32,
            surface: ctx.surface_size,
        });
    }
}
