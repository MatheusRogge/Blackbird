// Particle simulation (rendering::pass::particle_sim). One thread per slot: spawn into the slots
// the CPU picked this frame (ring order), otherwise drift and age; a particle is dead once its age
// reaches its lifetime. `lighting.wgsl` draws them (vs_particle / fs_particle).

// Mirrors rendering::pass::particle_sim::MAX_EMITTERS.
const MAX_EMITTERS: u32 = 16u;
const FLAG_PREWARM: u32 = 1u;

struct Particle {
    pos:  vec3<f32>,
    age:  f32,
    // Diameter in meters.
    size: f32,
    life: f32,
    // Per-particle random in [0, 1): speed variation, shading variation.
    rand: f32,
    _pad: f32,
}

// Mirrors rendering::pass::particle_sim::GpuEmitter.
struct GpuEmitter {
    transform: mat4x4<f32>,
    // Box half extents (xyz) or sphere radius (x); w: 0 box, 1 sphere.
    extent:   vec4<f32>,
    // Box shear (xz per unit y) in xy; z: spawn rate.
    shear:    vec4<f32>,
    // World-space velocity; w: turbulence speed.
    velocity: vec4<f32>,
    color:    vec4<f32>,
    // Size min / max, lifetime min / max.
    ranges:   vec4<f32>,
    // First slot, slot count, first slot spawned this frame, slots spawned this frame.
    slots:    vec4<u32>,
    // Step (s), time (s), 0, 0.
    step:     vec4<f32>,
    // Seed, flags, spawn generation, 0.
    info:     vec4<u32>,
}

struct ParticleParams {
    counts:   vec4<u32>,
    screen:   vec4<f32>,
    emitters: array<GpuEmitter, MAX_EMITTERS>,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: ParticleParams;

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

struct Rng { state: u32 }

fn next(rng: ptr<function, Rng>) -> f32 {
    (*rng).state = pcg((*rng).state);
    return f32((*rng).state >> 8u) / 16777216.0;
}

fn hash3(p: vec3<i32>) -> f32 {
    let h = pcg(u32(p.x) ^ pcg(u32(p.y) ^ pcg(u32(p.z))));
    return f32(h >> 8u) / 16777216.0 * 2.0 - 1.0;
}

// Smooth value noise in [-1, 1].
fn noise3(p: vec3<f32>) -> f32 {
    let i = vec3<i32>(floor(p));
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    var n = 0.0;
    for (var c = 0u; c < 8u; c++) {
        let o = vec3<i32>(i32(c & 1u), i32((c >> 1u) & 1u), i32(c >> 2u));
        let w = mix(1.0 - u, u, vec3f(o));
        n += hash3(i + o) * w.x * w.y * w.z;
    }
    return n;
}

fn potential(p: vec3<f32>) -> vec3<f32> {
    return vec3f(noise3(p), noise3(p + vec3f(31.4, 47.2, 12.9)), noise3(p + vec3f(-23.1, 17.7, 61.3)));
}

// Divergence-free flow (curl of a noise potential), roughly unit speed.
fn curl_noise(p: vec3<f32>) -> vec3<f32> {
    let e = 0.1;
    let dx = vec3f(e, 0.0, 0.0);
    let dy = vec3f(0.0, e, 0.0);
    let dz = vec3f(0.0, 0.0, e);
    let x = potential(p + dx) - potential(p - dx);
    let y = potential(p + dy) - potential(p - dy);
    let z = potential(p + dz) - potential(p - dz);
    return vec3f(y.z - z.y, z.x - x.z, x.y - y.x) / (2.0 * e) * 0.5;
}

// Noise features per meter, and how fast the flow pattern evolves.
const TURBULENCE_SCALE: f32 = 0.8;
const TURBULENCE_EVOLUTION: f32 = 0.05;

fn flow(e: GpuEmitter, pos: vec3<f32>, rand: f32) -> vec3<f32> {
    let t = e.step.y * TURBULENCE_EVOLUTION;
    let turbulence = curl_noise(pos * TURBULENCE_SCALE + vec3f(0.0, t, t * 0.7)) * e.velocity.w;
    return e.velocity.xyz * (0.6 + 0.8 * rand) + turbulence;
}

fn spawn_position(e: GpuEmitter, rng: ptr<function, Rng>) -> vec3<f32> {
    var local: vec3<f32>;
    if e.extent.w > 0.5 {
        // Uniform in the ball: a uniform direction, radius ∝ cbrt.
        let z = next(rng) * 2.0 - 1.0;
        let phi = next(rng) * 6.28318531;
        let s = sqrt(max(1.0 - z * z, 0.0));
        let r = e.extent.x * pow(next(rng), 1.0 / 3.0);
        local = vec3f(s * cos(phi), s * sin(phi), z) * r;
    } else {
        let u = vec3f(next(rng), next(rng), next(rng)) * 2.0 - 1.0;
        local = u * e.extent.xyz;
        local.x += e.shear.x * local.y;
        local.z += e.shear.y * local.y;
    }
    return (e.transform * vec4f(local, 1.0)).xyz;
}

fn spawn(e: GpuEmitter, slot: u32, generation: u32, age: f32) -> Particle {
    var rng = Rng(pcg(slot ^ pcg(e.info.x ^ pcg(generation))));
    var p: Particle;
    p.rand = next(&rng);
    p.size = mix(e.ranges.x, e.ranges.y, next(&rng));
    p.life = mix(e.ranges.z, e.ranges.w, next(&rng));
    p.age = age;
    // Turbulence is left out of the catch-up drift: prewarmed particles would all need integrating.
    p.pos = spawn_position(e, &rng) + e.velocity.xyz * (0.6 + 0.8 * p.rand) * age;
    return p;
}

@compute @workgroup_size(64)
fn cs_simulate(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.counts.y { return; }

    var ei = 0u;
    for (var k = 0u; k < params.counts.x; k++) {
        let s = params.emitters[k].slots;
        if i >= s.x && i < s.x + s.y { ei = k; break; }
    }
    let e = params.emitters[ei];
    let local = i - e.slots.x;
    let count = e.slots.y;
    let dt = e.step.x;

    if (e.info.y & FLAG_PREWARM) != 0u {
        // As if the ring had always run: the slot just before the cursor is the newest.
        let order = (e.slots.z + count - 1u - local) % count;
        var age: f32;
        if e.shear.z > 0.0 {
            age = (f32(order) + 0.5) / e.shear.z;
        } else {
            var rng = Rng(pcg(local ^ e.info.x));
            age = next(&rng) * e.ranges.w;
        }
        particles[i] = spawn(e, local, 0xffffffffu - order, age);
        return;
    }

    let order = (local + count - e.slots.z) % count;
    if order < e.slots.w {
        // Spread spawns over the step, oldest first.
        let age = dt * (1.0 - (f32(order) + 0.5) / f32(e.slots.w));
        particles[i] = spawn(e, local, e.info.z, age);
        return;
    }

    var p = particles[i];
    if p.age >= p.life { return; }
    p.pos += flow(e, p.pos, p.rand) * dt;
    p.age += dt;
    particles[i] = p;
}
