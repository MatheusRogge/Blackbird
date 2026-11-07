// Particle simulation (rendering::pass::particle_sim), entirely on the GPU. Each emitter owns a
// range of slots and a free list of the expired ones in it. Every frame:
//   cs_simulate (one thread per slot): moves and ages live particles, settles colliding ones, pushes
//     particles that just expired onto their emitter's free list and the rest onto the draw list.
//   cs_spawn (one thread per requested spawn): pops a free slot and starts a particle in it; with no
//     free slot left the spawn is dropped.
// The draw list and its count (the `particles` pass's indirect draw arguments) only hold live
// particles. `lighting.wgsl` draws them (vs_particle / fs_particle).

// Mirrors rendering::pass::particle_sim::MAX_EMITTERS.
const MAX_EMITTERS: u32 = 16u;
// Mirrors the FLAG_* constants in rendering::pass::particle_sim.
const FLAG_PREWARM: u32 = 1u;
const FLAG_SETTLE: u32 = 8u;
// Particle::flags.
const PARTICLE_RESTING: u32 = 1u;
// How far behind the depth buffer a particle may get in one step and still count as hitting the
// surface rather than having been hidden behind it.
const COLLISION_THICKNESS: f32 = 0.1;

struct Particle {
    pos:   vec3<f32>,
    age:   f32,
    // Diameter in meters.
    size:  f32,
    life:  f32,
    // Per-particle random in [0, 1): speed variation.
    rand:  f32,
    flags: u32,
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
    // First slot, slot count, first spawn thread, spawns this frame.
    slots:    vec4<u32>,
    // Step (s), time (s), 0, 0.
    step:     vec4<f32>,
    // Seed, flags, spawn generation, 0.
    info:     vec4<u32>,
}

struct ParticleParams {
    // Emitters, slots in use, spawns this frame, 0.
    counts:   vec4<u32>,
    screen:   vec4<f32>,
    emitters: array<GpuEmitter, MAX_EMITTERS>,
}

// Mirrors rendering::pass::particle_sim::ParticleState. `draw` is a `draw_indirect` argument
// block: 6 vertices × `instance_count` live particles.
struct DrawArgs {
    vertex_count:   u32,
    instance_count: atomic<u32>,
    first_vertex:   u32,
    first_instance: u32,
}

struct ParticleState {
    draw: DrawArgs,
    // Entries in each emitter's free list (its part of `free_slots`, starting at its first slot).
    free_count: array<atomic<i32>, MAX_EMITTERS>,
}

struct CameraUniform {
    view_proj:     mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    inv_view:      mat4x4<f32>,
    // near, far
    depth_params:  vec4<f32>,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: ParticleParams;
@group(0) @binding(2) var<storage, read_write> state: ParticleState;
@group(0) @binding(3) var<storage, read_write> free_slots: array<u32>;
@group(0) @binding(4) var<storage, read_write> draw_list: array<u32>;
@group(0) @binding(5) var t_depth: texture_depth_2d;
@group(0) @binding(6) var t_normal: texture_2d<f32>;
@group(0) @binding(7) var<uniform> camera: CameraUniform;

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
    p.flags = 0u;
    // Turbulence is left out of the catch-up drift: prewarmed particles would all need integrating.
    // Settling emitters get none, since it would carry particles through surfaces unchecked.
    let drift = select(age, 0.0, (e.info.y & FLAG_SETTLE) != 0u);
    p.pos = spawn_position(e, &rng) + e.velocity.xyz * (0.6 + 0.8 * p.rand) * drift;
    return p;
}

fn emitter_of_slot(slot: u32) -> u32 {
    for (var k = 0u; k < params.counts.x; k++) {
        let s = params.emitters[k].slots;
        if slot >= s.x && slot < s.x + s.y { return k; }
    }
    return 0u;
}

fn push_free(ei: u32, slot: u32) {
    let index = atomicAdd(&state.free_count[ei], 1);
    free_slots[params.emitters[ei].slots.x + u32(index)] = slot;
}

fn push_draw(slot: u32) {
    draw_list[atomicAdd(&state.draw.instance_count, 1u)] = slot;
}

// Reversed-Z depth to view-space distance along -Z (clip w).
fn linear_depth(d: f32) -> f32 {
    let n = camera.depth_params.x;
    let f = camera.depth_params.y;
    return n * f / (d * (f - n) + n);
}

struct Screen {
    on_screen: bool,
    texel: vec2<i32>,
    ndc: vec2<f32>,
    // Distance of the point along the view axis, and of the surface seen at its pixel.
    w: f32,
    surface_w: f32,
    depth: f32,
}

fn to_screen(pos: vec3<f32>) -> Screen {
    var s: Screen;
    let clip = camera.view_proj * vec4f(pos, 1.0);
    s.w = clip.w;
    if clip.w <= camera.depth_params.x { return s; }
    s.ndc = clip.xy / clip.w;
    if any(abs(s.ndc) >= vec2f(1.0)) { return s; }
    let dims = vec2f(textureDimensions(t_depth));
    s.texel = vec2<i32>((s.ndc * vec2f(0.5, -0.5) + 0.5) * dims);
    s.depth = textureLoad(t_depth, s.texel, 0);
    // Sky: nothing to hit.
    if s.depth <= 0.0 { return s; }
    s.surface_w = linear_depth(s.depth);
    s.on_screen = true;
    return s;
}

// `p` has just moved from `previous`; with settling on, a step that crosses from in
// front of the visible surface to just behind it ends on the surface, at rest.
fn collide(p: ptr<function, Particle>, previous: vec3<f32>) {
    let now = to_screen((*p).pos);
    if !now.on_screen || now.w <= now.surface_w || now.w > now.surface_w + COLLISION_THICKNESS { return; }
    let before = to_screen(previous);
    if before.on_screen && before.w > before.surface_w { return; }
    let surface = camera.inv_view_proj * vec4f(now.ndc, now.depth, 1.0);
    let normal = normalize(textureLoad(t_normal, now.texel, 0).rgb * 2.0 - 1.0);
    (*p).pos = surface.xyz / surface.w + normal * ((*p).size * 0.5);
    (*p).flags |= PARTICLE_RESTING;
}

@compute @workgroup_size(64)
fn cs_simulate(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.counts.y { return; }

    let ei = emitter_of_slot(i);
    let e = params.emitters[ei];
    let local = i - e.slots.x;
    let count = e.slots.y;
    let dt = e.step.x;

    if (e.info.y & FLAG_PREWARM) != 0u {
        // Steady state, as if the emitter had always run: one particle spawned every 1 / rate
        // seconds before now, oldest last, and those past their lifetime already free.
        var age: f32;
        if e.shear.z > 0.0 {
            age = (f32(local) + 0.5) / e.shear.z;
        } else {
            var rng = Rng(pcg(local ^ e.info.x));
            age = next(&rng) * e.ranges.w;
        }
        let p = spawn(e, local, 0xffffffffu - local, age);
        particles[i] = p;
        if p.age >= p.life { push_free(ei, i); } else { push_draw(i); }
        return;
    }

    var p = particles[i];
    if p.age >= p.life { return; }
    p.age += dt;
    if (p.flags & PARTICLE_RESTING) == 0u {
        let previous = p.pos;
        p.pos += flow(e, p.pos, p.rand) * dt;
        if (e.info.y & FLAG_SETTLE) != 0u {
            collide(&p, previous);
        }
    }
    particles[i] = p;
    if p.age >= p.life { push_free(ei, i); } else { push_draw(i); }
}

@compute @workgroup_size(64)
fn cs_spawn(@builtin(global_invocation_id) id: vec3<u32>) {
    let t = id.x;
    if t >= params.counts.z { return; }

    var ei = 0u;
    for (var k = 0u; k < params.counts.x; k++) {
        let s = params.emitters[k].slots;
        if t >= s.z && t < s.z + s.w { ei = k; break; }
    }
    let e = params.emitters[ei];
    let order = t - e.slots.z;

    let free = atomicSub(&state.free_count[ei], 1);
    if free <= 0 {
        // At capacity: undo and drop this spawn.
        atomicAdd(&state.free_count[ei], 1);
        return;
    }
    let slot = free_slots[e.slots.x + u32(free - 1)];
    // Spread spawns over the step, oldest first.
    let age = e.step.x * (1.0 - (f32(order) + 0.5) / f32(e.slots.w));
    particles[slot] = spawn(e, slot, e.info.z, age);
    push_draw(slot);
}
