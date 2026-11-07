use std::sync::{
    Arc, RwLock,
    atomic::{AtomicU32, Ordering},
};

use bytemuck::{Pod, Zeroable};
use engine_core::{entity::EntityKey, world::World};

use crate::{
    camera::Camera,
    debug_flags::RenderDebugFlags,
    graph::{NodeId, RenderGraph},
    particles::{EmitterShape, ParticleEmitter, ParticleLighting},
    pass::{Pass, PassContext, PassDesc},
    resource::{ResourceDescriptor, ResourceId},
    shader::ShaderAsset,
};

/// Mirrors `MAX_EMITTERS` in `particle_sim.wgsl` and `lighting.wgsl`.
pub const MAX_EMITTERS: usize = 16;
/// Particle slots shared by every emitter.
pub const MAX_PARTICLES: u32 = 1 << 17;
/// Size of one `Particle` in `particle_sim.wgsl`.
const PARTICLE_SIZE: u64 = 32;
const WORKGROUP_SIZE: u32 = 64;
/// Longest step simulated at once, so a stall doesn't fling particles out of their volume.
const MAX_STEP: f64 = 0.1;
/// `GpuEmitter::info.y` flags. Mirrored in the WGSL.
const FLAG_PREWARM: u32 = 1;
const FLAG_LIGHT_ONLY: u32 = 2;
const FLAG_UNLIT: u32 = 4;
/// The time sent to the GPU wraps here (noise animation only), keeping it precise in f32.
const TIME_PERIOD: f64 = 4096.0;

/// Mirrors `GpuEmitter` in `particle_sim.wgsl` / `lighting.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuEmitter {
    transform: [[f32; 4]; 4],
    /// Box half extents (xyz) or sphere radius (x); w: 0 box, 1 sphere.
    extent: [f32; 4],
    /// Box shear (xz per unit y) in xy; z: spawn rate.
    shear: [f32; 4],
    /// World-space velocity; w: turbulence.
    velocity: [f32; 4],
    /// rgb, opacity.
    color: [f32; 4],
    /// Size min / max, lifetime min / max.
    ranges: [f32; 4],
    /// First slot, slot count, first slot spawned this frame, slots spawned this frame.
    slots: [u32; 4],
    /// Step (s), time (s, wrapped), 0, 0.
    step: [f32; 4],
    /// Seed, flags, spawn generation, 0.
    info: [u32; 4],
}

/// Mirrors `ParticleParams` in `particle_sim.wgsl` / `lighting.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParticleParams {
    /// Emitter count, total slots in use, 0, 0.
    counts: [u32; 4],
    /// Surface width, height (px), projection y scale (1 / tan(fovy / 2)), 0.
    screen: [f32; 4],
    emitters: [GpuEmitter; MAX_EMITTERS],
}

pub struct ParticleOutputs {
    pub particles_id: ResourceId,
    pub params_id: ResourceId,
    /// Slots the `particles` pass draws this frame (0 = nothing to draw).
    pub draw_count: Arc<AtomicU32>,
}

/// CPU side of one emitter's slot range.
struct EmitterState {
    key: EntityKey,
    first: u32,
    count: u32,
    /// Next slot to spawn into (ring order).
    cursor: u32,
    /// Fraction of a particle owed to the next frame.
    carry: f64,
    last_time: f64,
    generation: u32,
    prewarm: bool,
}

/// Simulates every `ParticleEmitter` on the GPU (`particle_sim.wgsl`). Each emitter owns a range
/// of slots in one storage buffer and spawns into it in ring order, `spawn_rate` per second of its
/// own `time`. A particle lives until its lifetime runs out or its slot comes round again. When
/// the set of emitters changes, slots are reassigned and every emitter starts in its steady state
/// (each slot aged as if the ring had always been running).
pub struct ParticleSimPass {
    node_id: Option<NodeId>,
    shader: ShaderAsset,
    pipeline: Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    bind_group: Option<wgpu::BindGroup>,
    debug_flags: Arc<RwLock<RenderDebugFlags>>,
    states: Vec<EmitterState>,
    warned_full: bool,

    particles_id: ResourceId,
    params_id: ResourceId,
    draw_count: Arc<AtomicU32>,
}

impl ParticleSimPass {
    pub fn new(
        graph: &mut RenderGraph,
        shader: ShaderAsset,
        debug_flags: Arc<RwLock<RenderDebugFlags>>,
    ) -> (Self, ParticleOutputs) {
        let particles_id = graph.alloc_resource_id(ResourceDescriptor::Buffer {
            size: u64::from(MAX_PARTICLES) * PARTICLE_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let params_id = graph.alloc_resource_id(ResourceDescriptor::UploadBuffer {
            size: size_of::<ParticleParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let draw_count = Arc::new(AtomicU32::new(0));
        let pass = Self {
            node_id: None,
            shader,
            pipeline: None,
            bind_group: None,
            debug_flags,
            states: Vec::new(),
            warned_full: false,
            particles_id,
            params_id,
            draw_count: Arc::clone(&draw_count),
        };
        (pass, ParticleOutputs { particles_id, params_id, draw_count })
    }

    /// Reassigns slot ranges if the emitters changed (keys or counts), in iteration order.
    fn sync_layout(&mut self, emitters: &[(EntityKey, &ParticleEmitter)]) {
        let unchanged = self.states.len() == emitters.len()
            && self.states.iter().zip(emitters).all(|(s, (key, e))| s.key == *key && s.count == e.count);
        if unchanged {
            return;
        }
        self.states.clear();
        let mut first = 0;
        for &(key, emitter) in emitters {
            if self.states.len() == MAX_EMITTERS || first + emitter.count > MAX_PARTICLES {
                if !self.warned_full {
                    log::warn!(
                        "particles: more than {MAX_EMITTERS} emitters or {MAX_PARTICLES} particles, skipping the rest"
                    );
                    self.warned_full = true;
                }
                break;
            }
            self.states.push(EmitterState {
                key,
                first,
                count: emitter.count,
                cursor: 0,
                carry: 0.0,
                last_time: emitter.time,
                generation: 0,
                prewarm: true,
            });
            first += emitter.count;
        }
    }

    fn create_pipeline(&self, device: &wgpu::Device) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let module = self.shader.compile(device).expect("particle_sim shader compile failed");
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle_sim_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particle_sim_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("particle_sim"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cs_simulate"),
            compilation_options: Default::default(),
            cache: None,
        });
        (pipeline, layout)
    }
}

fn gpu_emitter(emitter: &ParticleEmitter, state: &mut EmitterState) -> GpuEmitter {
    let step = (emitter.time - state.last_time).clamp(0.0, MAX_STEP);
    state.last_time = emitter.time;
    let mut flags = 0;
    let (mut spawn_start, mut spawn_count) = (state.cursor, 0);
    if state.prewarm {
        flags |= FLAG_PREWARM;
        state.prewarm = false;
    } else if state.count > 0 {
        state.carry += f64::from(emitter.spawn_rate.max(0.0)) * step;
        let spawned = (state.carry.floor() as u32).min(state.count);
        state.carry = (state.carry - f64::from(spawned)).min(f64::from(state.count));
        spawn_start = state.cursor;
        spawn_count = spawned;
        state.cursor = (state.cursor + spawned) % state.count;
        state.generation = state.generation.wrapping_add(1);
    }
    if emitter.light_only {
        flags |= FLAG_LIGHT_ONLY;
    }
    if emitter.lighting == ParticleLighting::Unlit {
        flags |= FLAG_UNLIT;
    }
    let (extent, shear) = match emitter.shape {
        EmitterShape::Box { half_extents, shear } => {
            ([half_extents.x, half_extents.y, half_extents.z, 0.0], [shear.x, shear.y])
        }
        EmitterShape::Sphere { radius } => ([radius, radius, radius, 1.0], [0.0, 0.0]),
    };
    let v = emitter.velocity;
    let c = emitter.color;
    GpuEmitter {
        transform: emitter.transform.into(),
        extent,
        shear: [shear[0], shear[1], emitter.spawn_rate.max(0.0), 0.0],
        velocity: [v.x, v.y, v.z, emitter.turbulence],
        color: [c.x, c.y, c.z, emitter.opacity],
        ranges: [emitter.size[0], emitter.size[1], emitter.lifetime[0], emitter.lifetime[1]],
        slots: [state.first, state.count, spawn_start, spawn_count],
        step: [step as f32, emitter.time.rem_euclid(TIME_PERIOD) as f32, 0.0, 0.0],
        info: [emitter.seed, flags, state.generation, 0],
    }
}

impl PassDesc for ParticleSimPass {
    fn name(&self) -> &'static str {
        "particle_sim"
    }

    fn reads(&self) -> Vec<ResourceId> {
        vec![]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![self.particles_id, self.params_id]
    }
}

impl Pass for ParticleSimPass {
    fn bind_node_id(&mut self, node_id: NodeId) {
        self.node_id = Some(node_id);
    }

    fn execute(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &PassContext<'_>,
        world: &World,
    ) {
        let (Some(&particles), Some(&params_buf)) = (ctx.buffers.get(&self.particles_id), ctx.buffers.get(&self.params_id))
        else {
            return;
        };

        let enabled = self.debug_flags.read().expect("debug_flags poisoned").particles;
        let handles = world.get_entity_handles::<ParticleEmitter>();
        let emitters: Vec<(EntityKey, &ParticleEmitter)> = handles
            .iter()
            .filter(|h| enabled && h.entity.enabled && h.entity.count > 0)
            .map(|h| (h.id, h.entity))
            .collect();
        self.sync_layout(&emitters);

        let mut params = ParticleParams::zeroed();
        for (gpu, state) in params.emitters.iter_mut().zip(&mut self.states) {
            let Some(emitter) = world.get_entity::<ParticleEmitter>(state.key) else { continue };
            *gpu = gpu_emitter(emitter, state);
        }
        let total = self.states.last().map_or(0, |s| s.first + s.count);
        let (w, h) = ctx.surface_size;
        let fovy = world.get_entities::<Camera>().first().map_or(1.0, |c| c.fovy);
        params.counts = [self.states.len() as u32, total, 0, 0];
        params.screen = [w as f32, h as f32, 1.0 / (fovy * 0.5).tan(), 0.0];
        params_buf.write(queue, bytemuck::bytes_of(&params));
        self.draw_count.store(total, Ordering::Relaxed);
        if total == 0 {
            return;
        }

        if self.pipeline.is_none() {
            self.pipeline = Some(self.create_pipeline(device));
        }
        let Some((pipeline, layout)) = &self.pipeline else { return };
        let bind_group = self.bind_group.get_or_insert_with(|| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("particle_sim_bg"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: particles.binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: params_buf.binding() },
                ],
            })
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("particle_sim"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &*bind_group, &[]);
        pass.dispatch_workgroups(total.div_ceil(WORKGROUP_SIZE), 1, 1);
    }
}
