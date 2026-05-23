const PI: f32 = 3.14159265358979323846;
const GOLDEN_RATIO: f32 = 1.6180339887498948;

// ── BVH structures ─────────────────────────────────────────────────────────────

struct BvhNode {
    aabb_min:            vec3<f32>,
    left_or_first_prim:  u32,      // internal→left child; leaf→first tri index
    aabb_max:            vec3<f32>,
    right_or_prim_count: u32,      // internal→right child; leaf→(0x80000000 | count)
}

struct BvhTriangle {
    v0: vec3<f32>, _pad0: u32,
    v1: vec3<f32>, _pad1: u32,
    v2: vec3<f32>, _pad2: u32,
}

struct BvhInfo {
    node_count: u32,
    tri_count:  u32,
    enabled:    u32,
    _pad:       u32,
}

// ── Other structures ──────────────────────────────────────────────────────────

struct ProbeGridParams {
    grid_dim:              vec3<u32>,
    probe_count:           u32,
    grid_origin:           vec3<f32>,
    max_ray_distance:      f32,
    probe_spacing:         f32,
    rays_per_probe:        u32,
    irradiance_texel_size: u32,
    visibility_texel_size: u32,
    hysteresis:            f32,
    normal_bias:           f32,
    view_bias:             f32,
    _pad:                  f32,
}

struct GpuSkyLight {
    direction_vs: vec3<f32>,
    intensity:    f32,
    color:        vec3<f32>,
    _pad:         f32,
}

struct LightCounts {
    num_point: u32,
    num_spot:  u32,
    num_area:  u32,
    num_sky:   u32,
}

struct ProbeTraceFrame {
    frame_rotation: mat3x3<f32>,
    inv_view:       mat4x4<f32>,
    frame_index:    u32,
}

struct TraceResult {
    hit:    bool,
    pos:    vec3<f32>,
    normal: vec3<f32>,
    dist:   f32,
}

// ── Bindings ──────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read> bvh_nodes:    array<BvhNode>;
@group(0) @binding(1) var<storage, read> bvh_tris:     array<BvhTriangle>;
@group(0) @binding(2) var<uniform>       bvh_info:     BvhInfo;
@group(0) @binding(3) var<uniform>       probe_params: ProbeGridParams;
@group(0) @binding(4) var<storage, read> sky_lights:   array<GpuSkyLight>;
@group(0) @binding(5) var<uniform>       light_counts: LightCounts;
@group(0) @binding(6) var<uniform>       frame:        ProbeTraceFrame;
@group(0) @binding(7) var<storage, read_write> ray_radiance:  array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> ray_direction: array<vec4<f32>>;

// ── Ray–AABB intersection (slab method) ───────────────────────────────────────

fn ray_aabb_hit(ro: vec3<f32>, inv_rd: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>, t_max: f32) -> f32 {
    let t1 = (lo - ro) * inv_rd;
    let t2 = (hi - ro) * inv_rd;
    let t_near = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let t_far  = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
    if t_near > t_far || t_far < 0.0 || t_near > t_max { return -1.0; }
    return t_near;
}

// ── Möller–Trumbore ray–triangle intersection ─────────────────────────────────

fn ray_tri_hit(ro: vec3<f32>, rd: vec3<f32>, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> f32 {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let h  = cross(rd, e2);
    let a  = dot(e1, h);
    if abs(a) < 1e-7 { return -1.0; }
    let f  = 1.0 / a;
    let s  = ro - v0;
    let u  = f * dot(s, h);
    if u < 0.0 || u > 1.0 { return -1.0; }
    let q  = cross(s, e1);
    let v  = f * dot(rd, q);
    if v < 0.0 || u + v > 1.0 { return -1.0; }
    let t  = f * dot(e2, q);
    if t < 1e-4 { return -1.0; }
    return t;
}

// ── Iterative BVH traversal ───────────────────────────────────────────────────

fn bvh_trace(ro: vec3<f32>, rd: vec3<f32>, max_dist: f32) -> TraceResult {
    var result: TraceResult;
    result.hit  = false;
    result.dist = max_dist;

    let inv_rd = 1.0 / rd;
    var stack: array<u32, 64>;
    var sp: i32 = 1;
    stack[0] = 0u;

    while sp > 0 {
        sp -= 1;
        let node = bvh_nodes[stack[sp]];

        if ray_aabb_hit(ro, inv_rd, node.aabb_min, node.aabb_max, result.dist) < 0.0 { continue; }

        if (node.right_or_prim_count & 0x80000000u) != 0u {
            // Leaf — test each triangle.
            let first = node.left_or_first_prim;
            let count = node.right_or_prim_count & 0x7FFFFFFFu;
            for (var i = 0u; i < count; i++) {
                let tri = bvh_tris[first + i];
                let t = ray_tri_hit(ro, rd, tri.v0, tri.v1, tri.v2);
                if t > 0.0 && t < result.dist {
                    result.hit    = true;
                    result.dist   = t;
                    result.pos    = ro + rd * t;
                    let e1 = tri.v1 - tri.v0;
                    let e2 = tri.v2 - tri.v0;
                    var n = normalize(cross(e1, e2));
                    if dot(n, rd) > 0.0 { n = -n; }
                    result.normal = n;
                }
            }
        } else {
            // Internal — push both children.
            if sp < 62 {
                stack[sp]     = node.left_or_first_prim;
                stack[sp + 1] = node.right_or_prim_count;
                sp += 2;
            }
        }
    }

    return result;
}

// ── Ray generation ────────────────────────────────────────────────────────────

fn spherical_fibonacci(idx: u32, total: u32) -> vec3<f32> {
    let i       = f32(idx);
    let n       = f32(total);
    let theta   = 2.0 * PI * i / GOLDEN_RATIO;
    let cos_phi = 1.0 - (2.0 * i + 1.0) / n;
    let sin_phi = sqrt(max(0.0, 1.0 - cos_phi * cos_phi));
    return frame.frame_rotation * vec3<f32>(cos(theta) * sin_phi, sin(theta) * sin_phi, cos_phi);
}

fn probe_world_position(probe_index: u32) -> vec3<f32> {
    let dim = probe_params.grid_dim;
    let gx  = probe_index % dim.x;
    let gy  = (probe_index / dim.x) % dim.y;
    let gz  = probe_index / (dim.x * dim.y);
    return probe_params.grid_origin + vec3<f32>(f32(gx), f32(gy), f32(gz)) * probe_params.probe_spacing;
}

// ── Shading helpers ───────────────────────────────────────────────────────────

fn shade_hit(pos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let albedo = vec3<f32>(0.5);
    var radiance = vec3<f32>(0.0);
    for (var i = 0u; i < light_counts.num_sky; i++) {
        let dir_ws  = (frame.inv_view * vec4<f32>(sky_lights[i].direction_vs, 0.0)).xyz;
        let n_dot_l = max(0.0, dot(normal, -dir_ws));
        radiance   += albedo * sky_lights[i].color * sky_lights[i].intensity * n_dot_l;
    }
    return radiance;
}

fn sample_sky(ray_dir: vec3<f32>) -> vec3<f32> {
    if light_counts.num_sky > 0u {
        let dir_ws     = (frame.inv_view * vec4<f32>(sky_lights[0].direction_vs, 0.0)).xyz;
        let sun_factor = max(0.0, dot(ray_dir, -dir_ws));
        let sky_base   = sky_lights[0].color * sky_lights[0].intensity;
        return sky_base * (0.3 + 0.7 * max(0.0, ray_dir.y) + pow(sun_factor, 32.0));
    }
    let t = max(0.0, ray_dir.y) * 0.5 + 0.5;
    return mix(vec3f(0.8, 0.9, 1.0), vec3f(0.2, 0.4, 0.8), t) * 1.5;
}

// ── Main compute entry ────────────────────────────────────────────────────────

@compute @workgroup_size(32, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ray_index   = gid.x;
    let probe_index = gid.y;

    if ray_index   >= probe_params.rays_per_probe { return; }
    if probe_index >= probe_params.probe_count    { return; }

    let out_idx = probe_index * probe_params.rays_per_probe + ray_index;

    if bvh_info.enabled == 0u {
        ray_radiance[out_idx]  = vec4f(0.0, 0.0, 0.0, probe_params.max_ray_distance);
        ray_direction[out_idx] = vec4f(0.0, 1.0, 0.0, 0.0);
        return;
    }

    let probe_pos = probe_world_position(probe_index);
    let ray_dir   = spherical_fibonacci(ray_index, probe_params.rays_per_probe);
    let result    = bvh_trace(probe_pos, ray_dir, probe_params.max_ray_distance);

    var radiance: vec3<f32>;
    var hit_dist: f32;

    if result.hit {
        hit_dist = result.dist;
        radiance = shade_hit(result.pos, result.normal);
    } else {
        hit_dist = probe_params.max_ray_distance;
        radiance = sample_sky(ray_dir);
    }

    ray_radiance[out_idx]  = vec4f(radiance, hit_dist);
    ray_direction[out_idx] = vec4f(ray_dir, 0.0);
}
