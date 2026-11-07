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
    particles::{EmitterShape, ParticleCollision, ParticleEmitter, ParticleLighting},
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
const FLAG_SETTLE: u32 = 8;
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
    /// First slot, slot count, first spawn thread, spawns this frame.
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
    /// Emitter count, total slots in use, total spawns this frame, 0.
    counts: [u32; 4],
    /// Surface width, height (px), projection y scale (1 / tan(fovy / 2)), 0.
    screen: [f32; 4],
    emitters: [GpuEmitter; MAX_EMITTERS],
}

/// Mirrors `ParticleState` in `particle_sim.wgsl`: `draw_indirect` arguments for the live
/// particles, then each emitter's free-list length.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParticleState {
    draw: [u32; 4],
    free_count: [i32; MAX_EMITTERS],
}

pub struct ParticleOutputs {
    pub particles_id: ResourceId,
    pub params_id: ResourceId,
    /// `ParticleState`: the `particles` pass draws indirectly from its first 16 bytes.
    pub state_id: ResourceId,
    /// Slot of each live particle, `state.draw.instance_count` of them.
    pub draw_list_id: ResourceId,
    /// Slots in use this frame (0 = no emitters, nothing to draw).
    pub slots_in_use: Arc<AtomicU32>,
}

/// CPU side of one emitter's slot range.
struct EmitterState {
    key: EntityKey,
    first: u32,
    count: u32,
    /// Fraction of a particle owed to the next frame.
    carry: f64,
    last_time: f64,
    bursts_seen: u64,
    generation: u32,
    prewarm: bool,
}

/// Simulates every `ParticleEmitter` on the GPU (`particle_sim.wgsl`). Each emitter owns a range
/// of slots in one storage buffer and a free list of its expired slots. Each frame `cs_simulate`
/// moves and ages every live particle (settling colliding ones on the depth buffer), frees the
/// ones that expire and lists the rest for drawing; then `cs_spawn` starts `spawn_rate` per second
/// of the emitter's own `time` (plus new `burst`s) in free slots, dropping spawns past capacity.
/// The CPU only counts how many to spawn. When the set of emitters changes, slots are reassigned
/// and every emitter starts in its steady state.
pub struct ParticleSimPass {
    node_id: Option<NodeId>,
    shader: ShaderAsset,
    pipelines: Option<Pipelines>,
    /// Keyed by the depth and normal view addresses.
    bind_group: Option<((usize, usize), wgpu::BindGroup)>,
    debug_flags: Arc<RwLock<RenderDebugFlags>>,
    states: Vec<EmitterState>,
    warned_full: bool,
    /// Free-list lengths must be zeroed before the next prewarm.
    reset_free_lists: bool,

    particles_id: ResourceId,
    params_id: ResourceId,
    state_id: ResourceId,
    free_list_id: ResourceId,
    draw_list_id: ResourceId,
    depth_id: ResourceId,
    normal_id: ResourceId,
    camera_buffer_id: ResourceId,
    slots_in_use: Arc<AtomicU32>,
}

struct Pipelines {
    simulate: wgpu::ComputePipeline,
    spawn: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

impl ParticleSimPass {
    /// `depth_id` / `normal_id` are the G-buffer's, for collisions.
    pub fn new(
        graph: &mut RenderGraph,
        depth_id: ResourceId,
        normal_id: ResourceId,
        camera_buffer_id: ResourceId,
        shader: ShaderAsset,
        debug_flags: Arc<RwLock<RenderDebugFlags>>,
    ) -> (Self, ParticleOutputs) {
        let mut storage = |size: u64, extra: wgpu::BufferUsages| {
            graph.alloc_resource_id(ResourceDescriptor::Buffer {
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
            })
        };
        let particles_id = storage(u64::from(MAX_PARTICLES) * PARTICLE_SIZE, wgpu::BufferUsages::empty());
        let state_id = storage(size_of::<ParticleState>() as u64, wgpu::BufferUsages::INDIRECT);
        let free_list_id = storage(u64::from(MAX_PARTICLES) * 4, wgpu::BufferUsages::empty());
        let draw_list_id = storage(u64::from(MAX_PARTICLES) * 4, wgpu::BufferUsages::empty());
        let params_id = graph.alloc_resource_id(ResourceDescriptor::UploadBuffer {
            size: size_of::<ParticleParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let slots_in_use = Arc::new(AtomicU32::new(0));
        let pass = Self {
            node_id: None,
            shader,
            pipelines: None,
            bind_group: None,
            debug_flags,
            states: Vec::new(),
            warned_full: false,
            reset_free_lists: true,
            particles_id,
            params_id,
            state_id,
            free_list_id,
            draw_list_id,
            depth_id,
            normal_id,
            camera_buffer_id,
            slots_in_use: Arc::clone(&slots_in_use),
        };
        (pass, ParticleOutputs { particles_id, params_id, state_id, draw_list_id, slots_in_use })
    }

    /// Reassigns slot ranges if the emitters changed (keys or counts), in iteration order.
    fn sync_layout(&mut self, emitters: &[(EntityKey, &ParticleEmitter)]) {
        let unchanged = self.states.len() == emitters.len()
            && self.states.iter().zip(emitters).all(|(s, (key, e))| s.key == *key && s.count == e.count);
        if unchanged {
            return;
        }
        self.states.clear();
        self.reset_free_lists = true;
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
                carry: 0.0,
                last_time: emitter.time,
                bursts_seen: emitter.bursts,
                generation: 0,
                prewarm: true,
            });
            first += emitter.count;
        }
    }

    fn create_pipelines(&self, device: &wgpu::Device) -> Pipelines {
        let module = self.shader.compile(device).expect("particle_sim shader compile failed");
        let compute = wgpu::ShaderStages::COMPUTE;
        let buffer = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let storage = |binding| buffer(binding, wgpu::BufferBindingType::Storage { read_only: false });
        let texture = |binding, sample_type| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                view_dimension: wgpu::TextureViewDimension::D2,
                sample_type,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle_sim_bgl"),
            entries: &[
                storage(0),
                buffer(1, wgpu::BufferBindingType::Uniform),
                storage(2),
                storage(3),
                storage(4),
                texture(5, wgpu::TextureSampleType::Depth),
                texture(6, wgpu::TextureSampleType::Float { filterable: false }),
                buffer(7, wgpu::BufferBindingType::Uniform),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particle_sim_pipeline_layout"),
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
        Pipelines { simulate: pipeline("cs_simulate"), spawn: pipeline("cs_spawn"), layout }
    }
}

/// The emitter's GPU parameters for this frame, with `spawn_start` as its first spawn thread.
fn gpu_emitter(emitter: &ParticleEmitter, state: &mut EmitterState, spawn_start: u32) -> GpuEmitter {
    let step = (emitter.time - state.last_time).clamp(0.0, MAX_STEP);
    state.last_time = emitter.time;
    let bursts = emitter.bursts.saturating_sub(state.bursts_seen);
    state.bursts_seen = emitter.bursts;
    let mut flags = 0;
    let mut spawn_count = 0;
    if state.prewarm {
        flags |= FLAG_PREWARM;
        state.prewarm = false;
    } else {
        state.carry += f64::from(emitter.spawn_rate.max(0.0)) * step;
        let spawned = state.carry.floor();
        state.carry -= spawned;
        // More than `count` can never find a free slot.
        spawn_count = (spawned as u64).saturating_add(bursts).min(u64::from(state.count)) as u32;
        state.generation = state.generation.wrapping_add(1);
    }
    if emitter.collision == ParticleCollision::Settle {
        flags |= FLAG_SETTLE;
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
        vec![self.depth_id, self.normal_id, self.camera_buffer_id]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![self.particles_id, self.params_id, self.state_id, self.free_list_id, self.draw_list_id]
    }
}

impl Pass for ParticleSimPass {
    fn bind_node_id(&mut self, node_id: NodeId) {
        self.node_id = Some(node_id);
    }

    fn on_resize(&mut self, _width: u32, _height: u32) {
        self.bind_group = None;
    }

    fn execute(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &PassContext<'_>,
        world: &World,
    ) {
        let buffer = |id| ctx.buffers.get(id).copied();
        let (Some(particles), Some(params_buf), Some(state_buf), Some(free_list), Some(draw_list), Some(camera)) = (
            buffer(&self.particles_id),
            buffer(&self.params_id),
            buffer(&self.state_id),
            buffer(&self.free_list_id),
            buffer(&self.draw_list_id),
            buffer(&self.camera_buffer_id),
        ) else {
            return;
        };
        let (Some(&depth), Some(&normal)) = (ctx.views.get(&self.depth_id), ctx.views.get(&self.normal_id)) else {
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
        let mut spawns = 0;
        for (gpu, state) in params.emitters.iter_mut().zip(&mut self.states) {
            let Some(emitter) = world.get_entity::<ParticleEmitter>(state.key) else { continue };
            *gpu = gpu_emitter(emitter, state, spawns);
            spawns += gpu.slots[3];
        }
        let total = self.states.last().map_or(0, |s| s.first + s.count);
        let (w, h) = ctx.surface_size;
        let fovy = world.get_entities::<Camera>().first().map_or(1.0, |c| c.fovy);
        params.counts = [self.states.len() as u32, total, spawns, 0];
        params.screen = [w as f32, h as f32, 1.0 / (fovy * 0.5).tan(), 0.0];
        params_buf.write(queue, bytemuck::bytes_of(&params));
        self.slots_in_use.store(total, Ordering::Relaxed);
        if total == 0 {
            return;
        }

        // The draw count is rebuilt every frame; free lists only on a new layout (prewarm).
        if std::mem::take(&mut self.reset_free_lists) {
            let reset = ParticleState { draw: [6, 0, 0, 0], free_count: [0; MAX_EMITTERS] };
            state_buf.write(queue, bytemuck::bytes_of(&reset));
        } else {
            state_buf.write(queue, bytemuck::cast_slice(&[6u32, 0, 0, 0]));
        }

        if self.pipelines.is_none() {
            self.pipelines = Some(self.create_pipelines(device));
        }
        let Some(pipelines) = &self.pipelines else { return };
        let key = (depth as *const wgpu::TextureView as usize, normal as *const wgpu::TextureView as usize);
        if self.bind_group.as_ref().is_none_or(|(k, _)| *k != key) {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("particle_sim_bg"),
                layout: &pipelines.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: particles.binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: params_buf.binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: state_buf.binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: free_list.binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: draw_list.binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(depth) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(normal) },
                    wgpu::BindGroupEntry { binding: 7, resource: camera.binding() },
                ],
            });
            self.bind_group = Some((key, bind_group));
        }
        let Some((_, bind_group)) = &self.bind_group else { return };

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("particle_sim"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_pipeline(&pipelines.simulate);
        pass.dispatch_workgroups(total.div_ceil(WORKGROUP_SIZE), 1, 1);
        if spawns > 0 {
            pass.set_pipeline(&pipelines.spawn);
            pass.dispatch_workgroups(spawns.div_ceil(WORKGROUP_SIZE), 1, 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(count: u32) -> EmitterState {
        EmitterState {
            key: EntityKey::default(),
            first: 0,
            count,
            carry: 0.0,
            last_time: 0.0,
            bursts_seen: 0,
            generation: 0,
            prewarm: false,
        }
    }

    fn spawns(emitter: &ParticleEmitter, state: &mut EmitterState) -> u32 {
        gpu_emitter(emitter, state, 0).slots[3]
    }

    #[test]
    fn spawn_rate_carries_fractions_across_frames() {
        let mut emitter = ParticleEmitter { spawn_rate: 30.0, count: 100, ..Default::default() };
        let mut s = state(100);
        let mut total = 0;
        for frame in 1..=600 {
            emitter.time = f64::from(frame) / 60.0;
            total += spawns(&emitter, &mut s);
        }
        // Rounding can leave the last particle's fraction a hair short of whole until next frame.
        assert!((299..=300).contains(&total), "{total}");
    }

    #[test]
    fn bursts_spawn_once_and_are_capped_at_capacity() {
        let mut emitter = ParticleEmitter { spawn_rate: 0.0, count: 50, ..Default::default() };
        let mut s = state(50);
        emitter.burst(20);
        assert_eq!(spawns(&emitter, &mut s), 20);
        assert_eq!(spawns(&emitter, &mut s), 0);
        emitter.burst(500);
        assert_eq!(spawns(&emitter, &mut s), 50);
    }

    #[test]
    fn prewarm_spawns_nothing_and_a_stall_is_clamped() {
        let mut emitter = ParticleEmitter { spawn_rate: 100.0, count: 1000, ..Default::default() };
        let mut s = EmitterState { prewarm: true, ..state(1000) };
        let first = gpu_emitter(&emitter, &mut s, 0);
        assert_eq!(first.info[1] & FLAG_PREWARM, FLAG_PREWARM);
        assert_eq!(first.slots[3], 0);
        emitter.time = 10.0;
        assert_eq!(spawns(&emitter, &mut s), (100.0 * MAX_STEP) as u32);
    }
}
