use engine_core::entity::Entity;
use ultraviolet::{Mat4, Vec2, Vec3};

/// Where an emitter spawns particles, in its local space (`ParticleEmitter::transform`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EmitterShape {
    /// A box of `half_extents`, centered. `shear` slides each horizontal slice sideways by `shear × y`
    /// (local XZ per unit of local Y), e.g. to fill a slanted light shaft.
    Box { half_extents: Vec3, shear: Vec2 },
    Sphere { radius: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticleLighting {
    /// Particles scatter the scene's lights (with shadows) and, unless `light_only`, DDGI.
    Lit,
    /// Particles emit `color` as radiance (nits).
    Unlit,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ParticleCollision {
    /// Particles pass through geometry.
    #[default]
    None,
    /// Particles that cross a visible surface (G-buffer depth) come to rest on it until they expire.
    /// Screen-space: surfaces off screen or hidden behind others don't stop them.
    Settle,
}

/// A GPU-simulated particle system, drawn additively as camera-facing discs (`particle_sim` +
/// `particles` passes). Up to `count` particles are alive at once: `spawn_rate` per second (plus
/// `burst`s), each living a random time in `lifetime`. Spawns only take expired particles' slots,
/// so at capacity they are dropped. The system starts in its steady state, so it is already full
/// on its first frame.
#[derive(Clone, Debug, PartialEq)]
pub struct ParticleEmitter {
    pub transform: Mat4,
    pub shape: EmitterShape,
    pub count: u32,
    pub spawn_rate: f32,
    /// Seconds, min and max.
    pub lifetime: [f32; 2],
    /// Diameter in meters, min and max.
    pub size: [f32; 2],
    /// World-space drift, m/s.
    pub velocity: Vec3,
    /// Speed (m/s) of the curl-noise turbulence added to `velocity`.
    pub turbulence: f32,
    /// Scattering albedo for `Lit`, radiance for `Unlit`.
    pub color: Vec3,
    /// Coverage of one particle.
    pub opacity: f32,
    pub lighting: ParticleLighting,
    /// Lit by direct light only (no DDGI), so particles show only where a light reaches them.
    pub light_only: bool,
    pub collision: ParticleCollision,
    pub seed: u32,
    /// Hidden and not simulated (e.g. a deleted node in the editor).
    pub enabled: bool,
    /// Simulated seconds, advanced by `tick`. The `particle_sim` pass steps by its change.
    pub time: f64,
    /// Particles requested through `burst` so far. The `particle_sim` pass spawns the increase.
    pub bursts: u64,
}

impl ParticleEmitter {
    /// Spawns `count` particles at once on the next frame (as many as there are free slots).
    pub fn burst(&mut self, count: u32) {
        self.bursts += u64::from(count);
    }
}

impl Default for ParticleEmitter {
    fn default() -> Self {
        Self {
            transform: Mat4::identity(),
            shape: EmitterShape::Sphere { radius: 1.0 },
            count: 1000,
            spawn_rate: 100.0,
            lifetime: [5.0, 10.0],
            size: [0.005, 0.01],
            velocity: Vec3::zero(),
            turbulence: 0.0,
            color: Vec3::one(),
            opacity: 1.0,
            lighting: ParticleLighting::Lit,
            light_only: false,
            collision: ParticleCollision::None,
            seed: 0,
            enabled: true,
            time: 0.0,
            bursts: 0,
        }
    }
}

impl Entity for ParticleEmitter {
    fn tick(&mut self, delta: f32) {
        self.time += f64::from(delta);
    }
}
