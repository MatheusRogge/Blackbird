// Ray-traced shadows for the local lights `light_upload` picks (ShadowedLights): one any-hit ray per
// light and pixel through the DDGI BVH, written as visibility into one channel each (1 = lit).
// Area lights are sampled at one jittered point per pixel; local_shadow_blur.wgsl smooths the noise.

@group(0) @binding(0)  var t_depth:  texture_depth_2d;
@group(0) @binding(1)  var t_normal: texture_2d<f32>;
@group(0) @binding(2)  var<uniform> camera:   CameraUniform;
@group(0) @binding(3)  var<uniform> shadowed: ShadowedLights;
@group(0) @binding(4)  var<uniform> bvh_info: BvhInfo;
@group(0) @binding(5)  var<storage, read> point_lights: array<GpuPointLight>;
@group(0) @binding(6)  var<storage, read> spot_lights:  array<GpuSpotLight>;
@group(0) @binding(7)  var<storage, read> area_lights:  array<GpuAreaLight>;
@group(0) @binding(8)  var<storage, read> bvh_nodes:    array<BvhNode>;
@group(0) @binding(9)  var<storage, read> bvh_tris:     array<BvhTriangle>;
@group(0) @binding(10) var s_linear: sampler;

struct CameraUniform {
    view_proj:     mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    inv_view:      mat4x4<f32>,
    depth_params:  vec4<f32>,
}

// Mirrors rendering::light::ShadowedLights. Entry = kind << 16 | index.
struct ShadowedLights {
    entries: vec4<u32>,
    count:   u32,
}

const KIND_POINT: u32 = 0u;
const KIND_SPOT:  u32 = 1u;
const KIND_AREA:  u32 = 2u;

struct BvhInfo {
    node_count: u32,
    tri_count:  u32,
    enabled:    u32,
    _pad:       u32,
}

struct BvhNode {
    aabb_min:            vec3<f32>,
    left_or_first_prim:  u32,
    aabb_max:            vec3<f32>,
    right_or_prim_count: u32,
}

// Mirrors rendering::bvh::GpuTriangle; only the vertices are used here.
struct BvhTriangle {
    v0: vec3<f32>, albedo_metallic: u32,
    v1: vec3<f32>, roughness: f32,
    v2: vec3<f32>, emissive: u32,
}

struct GpuPointLight {
    position_vs: vec3<f32>,
    radius:      f32,
    color:       vec3<f32>,
    intensity:   f32,
}

struct GpuSpotLight {
    position_vs:  vec3<f32>,
    radius:       f32,
    direction_vs: vec3<f32>,
    inner_cos:    f32,
    color:        vec3<f32>,
    outer_cos:    f32,
    intensity:    f32,
    _pad0:        f32,
    _pad1:        f32,
    _pad2:        f32,
}

struct GpuAreaLight {
    position_vs: vec3<f32>,
    intensity:   f32,
    right_vs:    vec3<f32>,
    two_sided:   f32,
    up_vs:       vec3<f32>,
    _pad1:       f32,
    color:       vec3<f32>,
    _pad2:       f32,
}

// Rays start this far off the surface and stop this far short of the light: a constant plus a
// fraction of the distance to the camera, where depth precision drops.
const BIAS:          f32 = 0.01;
const BIAS_PER_UNIT: f32 = 0.002;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    return VertexOutput(vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0));
}

fn ray_aabb_hit(ro: vec3<f32>, inv_rd: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>, t_max: f32) -> bool {
    let t1 = (lo - ro) * inv_rd;
    let t2 = (hi - ro) * inv_rd;
    let t_near = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let t_far  = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
    return t_near <= t_far && t_far >= 0.0 && t_near <= t_max;
}

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

// True if anything blocks the segment from `ro` along `rd` (normalized) for `max_dist`.
fn occluded(ro: vec3<f32>, rd: vec3<f32>, max_dist: f32) -> bool {
    let inv_rd = 1.0 / rd;
    var stack: array<u32, 64>;
    var sp: i32 = 1;
    stack[0] = 0u;
    while sp > 0 {
        sp -= 1;
        let node = bvh_nodes[stack[sp]];
        if !ray_aabb_hit(ro, inv_rd, node.aabb_min, node.aabb_max, max_dist) { continue; }
        if (node.right_or_prim_count & 0x80000000u) != 0u {
            let first = node.left_or_first_prim;
            let count = node.right_or_prim_count & 0x7FFFFFFFu;
            for (var i = 0u; i < count; i++) {
                let tri = bvh_tris[first + i];
                let t = ray_tri_hit(ro, rd, tri.v0, tri.v1, tri.v2);
                if t > 0.0 && t < max_dist { return true; }
            }
        } else if sp < 62 {
            stack[sp]     = node.left_or_first_prim;
            stack[sp + 1] = node.right_or_prim_count;
            sp += 2;
        }
    }
    return false;
}

fn view_to_world(p: vec3<f32>) -> vec3<f32> {
    return (camera.inv_view * vec4<f32>(p, 1.0)).xyz;
}

fn dir_to_world(d: vec3<f32>) -> vec3<f32> {
    return (camera.inv_view * vec4<f32>(d, 0.0)).xyz;
}

fn interleaved_gradient_noise(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715))));
}

// Visibility of shadowed light `slot` from `pos` (normal `N`). Lights that can't reach the
// surface count as visible: they add nothing there anyway, and it keeps the blur from darkening
// lit neighbors.
fn visibility(slot: u32, pos: vec3<f32>, N: vec3<f32>, bias: f32, pixel: vec2<f32>) -> f32 {
    let entry = shadowed.entries[slot];
    let kind  = entry >> 16u;
    let index = entry & 0xFFFFu;

    var light_pos: vec3<f32>;
    if kind == KIND_POINT {
        let light = point_lights[index];
        light_pos = view_to_world(light.position_vs);
        if distance(light_pos, pos) >= light.radius { return 1.0; }
    } else if kind == KIND_SPOT {
        let light = spot_lights[index];
        light_pos = view_to_world(light.position_vs);
        let L = normalize(light_pos - pos);
        if distance(light_pos, pos) >= light.radius || dot(-L, normalize(dir_to_world(light.direction_vs))) <= light.outer_cos {
            return 1.0;
        }
    } else {
        let light = area_lights[index];
        // Stratified per pixel (different per light), so the blur sees the whole rectangle.
        let u = interleaved_gradient_noise(pixel + f32(slot) * 17.0);
        let v = interleaved_gradient_noise(pixel.yx + f32(slot) * 31.0 + 7.0);
        light_pos = view_to_world(light.position_vs)
            + dir_to_world(light.right_vs) * (2.0 * u - 1.0)
            + dir_to_world(light.up_vs) * (2.0 * v - 1.0);
        let facing = dot(normalize(pos - light_pos), dir_to_world(cross(light.right_vs, light.up_vs)));
        if light.two_sided < 0.5 && facing <= 0.0 { return 1.0; }
    }

    let to_light = light_pos - pos;
    let dist = length(to_light);
    let L = to_light / max(dist, 1e-5);
    if dot(N, L) <= 0.0 || dist <= 2.0 * bias { return 1.0; }
    return select(1.0, 0.0, occluded(pos + N * bias, L, dist - 2.0 * bias));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let px = vec2<i32>(in.pos.xy);
    let depth = textureLoad(t_depth, px, 0);
    if depth == 0.0 || bvh_info.enabled == 0u || shadowed.count == 0u {
        return vec4f(1.0);
    }

    let dims = vec2<f32>(textureDimensions(t_depth));
    let uv = in.pos.xy / dims;
    let pos4 = camera.inv_view_proj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let pos = pos4.xyz / pos4.w;
    let N = normalize(textureLoad(t_normal, px, 0).rgb * 2.0 - 1.0);
    let bias = BIAS + BIAS_PER_UNIT * distance(pos, camera.inv_view[3].xyz);

    var out = vec4f(1.0);
    for (var slot = 0u; slot < min(shadowed.count, 4u); slot++) {
        out[slot] = visibility(slot, pos, N, bias, in.pos.xy);
    }
    return out;
}
