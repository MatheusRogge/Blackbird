// Ground-truth ambient occlusion (Jimenez et al. 2016, "Practical Real-Time Strategies for Accurate
// Indirect Occlusion") over the G-buffer depth and normals, in world space. `fs_ao` integrates the
// visible hemisphere over a few screen-space slices per pixel, each rotated by a 4×4 interleaved
// pattern; `fs_blur` averages that 4×4 neighbourhood, weighted by how well each tap lies on the
// centre pixel's surface, so the result sees 4 × DIRECTIONS slice angles without temporal history.
// Glass isn't in the G-buffer depth: `vs_glass` / `fs_glass` first draw the nearest visible glass
// surface into its own depth target, with how much light it blocks, and `fs_ao` counts it as a
// partial occluder.

@group(0) @binding(0) var t_depth:  texture_depth_2d;
@group(0) @binding(1) var t_normal: texture_2d<f32>;
@group(0) @binding(2) var<uniform> camera: CameraUniform;
// `fs_ao`'s output, read by `fs_blur`.
@group(0) @binding(3) var t_raw:    texture_2d<f32>;
@group(0) @binding(4) var<storage, read> models: array<Model>;
// Nearest glass in front of the opaque depth (reversed-Z, 0 = none) and its opacity.
@group(0) @binding(5) var t_glass_depth:   texture_depth_2d;
@group(0) @binding(6) var t_glass_opacity: texture_2d<f32>;

// Mirrors rendering::arena::GpuModel.
struct Model {
    model: mat4x4<f32>,
    normal: mat4x4<f32>,
    material: vec4<f32>,
    base_color: vec4<f32>,
    emissive: vec4<f32>,
    // x: transmission factor.
    transmission: vec4<f32>,
    attenuation: vec4<f32>,
    diffuse_transmission: vec4<f32>,
}

// Mirrors the prefix of rendering::pass::camera::CameraUniform.
struct CameraUniform {
    view_proj:     mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    inv_view:      mat4x4<f32>,
    depth_params:  vec4<f32>,
}

const PI: f32 = 3.14159265;
// World-space reach of the occlusion.
const RADIUS: f32 = 0.5;
// Up close the radius would cover most of the screen; cap it at this fraction of its height.
const MAX_RADIUS_FRACTION: f32 = 0.15;
// Occluders count fully up to this fraction of RADIUS, then fade out linearly (XeGTAO).
const FALLOFF_START: f32 = 0.6;
const DIRECTIONS: u32 = 2u;
const STEPS: u32 = 6u;
// A tap counts for the blur while it is within this fraction of the depth of the centre's plane.
const BLUR_PLANE_TOLERANCE: f32 = 0.02;


struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    return VertexOutput(vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0));
}

// 4×4 Bayer matrix entry (0–15): neighbouring cells get well-spread values.
fn bayer(c: vec2<u32>) -> f32 {
    let a = c.x ^ c.y;
    return f32(((a & 1u) << 3u) | ((c.y & 1u) << 2u) | (a & 2u) | ((c.y >> 1u) & 1u));
}

fn world_pos(pixel: vec2<f32>, depth: f32, dims: vec2<f32>) -> vec3<f32> {
    let ndc = vec2f(pixel.x / dims.x * 2.0 - 1.0, 1.0 - pixel.y / dims.y * 2.0);
    let h = camera.inv_view_proj * vec4f(ndc, depth, 1.0);
    return h.xyz / h.w;
}

fn load_normal(pixel: vec2<i32>) -> vec3<f32> {
    return normalize(textureLoad(t_normal, pixel, 0).xyz * 2.0 - 1.0);
}

// Visible fraction of one side of a slice, between the normal's angle `n` and horizon angle `h`
// (cosine-weighted, Jimenez's eq. 10).
fn slice_arc(h: f32, n: f32) -> f32 {
    return 0.25 * (-cos(2.0 * h - n) + cos(n) + 2.0 * h * sin(n));
}

struct GlassVertex {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) opacity: f32,
}

// The fraction of light a glass surface blocks: its coverage, minus what transmission lets through,
// tinted by the base color and absorbed through the volume's thickness (Beer–Lambert). Textures and
// Fresnel are ignored.
@vertex
fn vs_glass(@location(0) position: vec3<f32>, @builtin(instance_index) instance: u32) -> GlassVertex {
    let m = models[instance];
    var tint = m.base_color.rgb;
    if m.transmission.w > 0.0 {
        let scale = (length(m.model[0].xyz) + length(m.model[1].xyz) + length(m.model[2].xyz)) / 3.0;
        tint *= pow(m.attenuation.rgb, vec3f(m.transmission.z * scale / m.transmission.w));
    }
    let passed = m.transmission.x * dot(tint, vec3f(0.2126, 0.7152, 0.0722));
    return GlassVertex(camera.view_proj * m.model * vec4f(position, 1.0), m.base_color.a * saturate(1.0 - passed));
}

@fragment
fn fs_glass(in: GlassVertex) -> @location(0) vec4<f32> {
    if in.pos.z < textureLoad(t_depth, vec2<i32>(in.pos.xy), 0) {
        discard;
    }
    return vec4f(in.opacity);
}

// Horizon cosine of an occluder at `sample_pos`, pulled down to -1 (no occlusion) by the distance
// falloff.
fn horizon_cos(sample_pos: vec3<f32>, p: vec3<f32>, v: vec3<f32>) -> f32 {
    let delta = sample_pos - p;
    let dist = length(delta);
    let cos_h = dot(delta, v) / max(dist, 1e-4);
    let falloff = saturate((dist - FALLOFF_START * RADIUS) / ((1.0 - FALLOFF_START) * RADIUS));
    return mix(cos_h, -1.0, falloff);
}

@fragment
fn fs_ao(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2f(textureDimensions(t_depth));
    let pixel = vec2<i32>(in.pos.xy);
    let depth = textureLoad(t_depth, pixel, 0);
    if depth == 0.0 {
        return vec4f(1.0);
    }
    let p = world_pos(in.pos.xy, depth, dims);
    let n = load_normal(pixel);
    let v = normalize(camera.inv_view[3].xyz - p);

    let texel = length(world_pos(in.pos.xy + vec2f(1.0, 0.0), depth, dims) - p);
    let radius_px = min(RADIUS / max(texel, 1e-6), MAX_RADIUS_FRACTION * dims.y);
    if radius_px < 1.0 {
        return vec4f(1.0);
    }

    let cell = vec2<u32>(pixel) & vec2<u32>(3u);
    let rotation = (bayer(cell) + 0.5) / 16.0;
    let jitter = (bayer(cell.yx) + 0.5) / 16.0;

    var visible = 0.0;
    var total = 0.0;
    for (var i = 0u; i < DIRECTIONS; i++) {
        let phi = (f32(i) + rotation) * PI / f32(DIRECTIONS);
        let dir = vec2f(cos(phi), sin(phi));
        // The slice's world-space direction: toward the pixel next along `dir`, made orthogonal to v.
        let toward = world_pos(in.pos.xy + dir, depth, dims) - p;
        let ortho = normalize(toward - v * dot(toward, v));
        let axis = cross(ortho, v);
        let n_proj = n - axis * dot(n, axis);
        let n_len = length(n_proj);
        if n_len < 1e-4 {
            continue;
        }
        let angle_n = sign(dot(n_proj, ortho)) * acos(clamp(dot(n_proj / n_len, v), -1.0, 1.0));

        // Cosine of the highest horizon on the -dir and +dir sides.
        // Per side: the opaque horizon, the horizon counting glass as solid, and the opacity of the
        // glass that set it. A side's visibility blends the two by that opacity.
        var horizon = vec2f(-1.0);
        var glass_horizon = vec2f(-1.0);
        var glass_opacity = vec2f(0.0);
        for (var side = 0u; side < 2u; side++) {
            let s = f32(side) * 2.0 - 1.0;
            for (var j = 0u; j < STEPS; j++) {
                let f = (f32(j) + jitter) / f32(STEPS);
                let offset = max(f * f * radius_px, f32(j) + 1.0);
                let sample_px = floor(in.pos.xy + dir * (offset * s)) + 0.5;
                if any(sample_px < vec2f(0.0)) || any(sample_px >= dims) {
                    break;
                }
                let sample_depth = textureLoad(t_depth, vec2<i32>(sample_px), 0);
                if sample_depth != 0.0 {
                    horizon[side] = max(horizon[side], horizon_cos(world_pos(sample_px, sample_depth, dims), p, v));
                }
                let glass_depth = textureLoad(t_glass_depth, vec2<i32>(sample_px), 0);
                if glass_depth != 0.0 {
                    let cos_h = horizon_cos(world_pos(sample_px, glass_depth, dims), p, v);
                    if cos_h > glass_horizon[side] {
                        glass_horizon[side] = cos_h;
                        glass_opacity[side] = textureLoad(t_glass_opacity, vec2<i32>(sample_px), 0).r;
                    }
                }
            }
        }

        let with_glass = max(horizon, glass_horizon);
        let h0 = angle_n + max(-acos(horizon.x) - angle_n, -0.5 * PI);
        let h1 = angle_n + min(acos(horizon.y) - angle_n, 0.5 * PI);
        let g0 = angle_n + max(-acos(with_glass.x) - angle_n, -0.5 * PI);
        let g1 = angle_n + min(acos(with_glass.y) - angle_n, 0.5 * PI);
        visible += (mix(slice_arc(h0, angle_n), slice_arc(g0, angle_n), glass_opacity.x)
            + mix(slice_arc(h1, angle_n), slice_arc(g1, angle_n), glass_opacity.y)) * n_len;
        total += n_len;
    }
    return vec4f(select(1.0, visible / total, total > 0.0));
}

@fragment
fn fs_blur(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2f(textureDimensions(t_depth));
    let pixel = vec2<i32>(in.pos.xy);
    let depth = textureLoad(t_depth, pixel, 0);
    if depth == 0.0 {
        return vec4f(1.0);
    }
    let p = world_pos(in.pos.xy, depth, dims);
    let n = load_normal(pixel);
    let tolerance = BLUR_PLANE_TOLERANCE * length(camera.inv_view[3].xyz - p);

    var sum = 0.0;
    var weight = 0.0;
    for (var y = -2; y < 2; y++) {
        for (var x = -2; x < 2; x++) {
            let tap = clamp(pixel + vec2<i32>(x, y), vec2<i32>(0), vec2<i32>(dims) - 1);
            let tap_depth = textureLoad(t_depth, tap, 0);
            if tap_depth == 0.0 {
                continue;
            }
            let off_plane = abs(dot(world_pos(vec2f(tap) + 0.5, tap_depth, dims) - p, n));
            let w = saturate(1.0 - off_plane / tolerance) * pow(saturate(dot(load_normal(tap), n)), 8.0);
            sum += textureLoad(t_raw, tap, 0).r * w;
            weight += w;
        }
    }
    return vec4f(select(textureLoad(t_raw, pixel, 0).r, sum / weight, weight > 1e-4));
}
