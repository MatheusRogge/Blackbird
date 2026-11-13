// Ground-truth ambient occlusion (Jimenez et al. 2016, "Practical Real-Time Strategies for Accurate
// Indirect Occlusion") over the G-buffer depth and normals, in world space. `fs_ao` integrates the
// visible hemisphere over a few screen-space slices per pixel, each rotated by a 4×4 interleaved
// pattern, at half resolution; `fs_blur` upsamples by averaging the 4×4 nearest half-resolution
// texels, weighted by how well each lies on the pixel's surface, so the result sees 4 × DIRECTIONS
// slice angles without temporal history.
// Glass isn't in the G-buffer depth: `vs_glass` / `fs_glass` first draw the nearest visible glass
// surface into its own depth target, with the light it lets through, and `fs_ao` counts it as a
// partial occluder (`lighting`'s contact shadows march it too).

@group(0) @binding(0) var t_depth:  texture_depth_2d;
@group(0) @binding(1) var t_normal: texture_2d<f32>;
@group(0) @binding(2) var<uniform> camera: CameraUniform;
// `fs_ao`'s output, read by `fs_blur`.
@group(0) @binding(3) var t_raw:    texture_2d<f32>;
@group(0) @binding(4) var<storage, read> models: array<Model>;
// Transmittance (rgb) of the nearest glass in front of the opaque depth.
@group(0) @binding(6) var t_glass_transmittance: texture_2d<f32>;
// Linear depth of the opaque surface (r) and of that glass (g), 0 for none: `fs_contact_depth` and
// `fs_glass` fill it before `fs_ao`, which marches both with one load per step when glass is visible.
@group(0) @binding(7) var t_contact_depth: texture_2d<f32>;

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
// False for frames with no visible glass: the march then reads the 4-byte G-buffer depth instead of
// the 8-byte contact depths. It is bound by its cache footprint, so that is 1.5–2 ms at 2000×844.
override HAS_GLASS: bool = true;


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
    @location(0) @interpolate(flat) transmittance: vec3<f32>,
}

// The light a glass object lets through: the uncovered part, plus the covered part's transmission
// tinted by the base color and absorbed through the volume's whole thickness (Beer–Lambert).
// Textures and Fresnel are ignored.
@vertex
fn vs_glass(@location(0) position: vec3<f32>, @builtin(instance_index) instance: u32) -> GlassVertex {
    let m = models[instance];
    var tint = m.base_color.rgb;
    if m.transmission.w > 0.0 {
        let scale = (length(m.model[0].xyz) + length(m.model[1].xyz) + length(m.model[2].xyz)) / 3.0;
        tint *= pow(m.attenuation.rgb, vec3f(m.transmission.z * scale / m.transmission.w));
    }
    let through = 1.0 - m.base_color.a + m.base_color.a * m.transmission.x * saturate(tint);
    return GlassVertex(camera.view_proj * m.model * vec4f(position, 1.0), through);
}

// The glass pass's targets: transmittance, and linear depths for `lighting`'s contact shadows
// (r opaque, g glass, 0 for none).
struct GlassTargets {
    @location(0) transmittance: vec4<f32>,
    @location(1) contact_depth: vec4<f32>,
}

fn linear_depth(d: f32) -> f32 {
    let n = camera.depth_params.x;
    let f = camera.depth_params.y;
    return select(0.0, n * f / (d * (f - n) + n), d != 0.0);
}

fn device_depth(linear: f32) -> f32 {
    let n = camera.depth_params.x;
    let f = camera.depth_params.y;
    return select(0.0, (n * f / linear - n) / (f - n), linear != 0.0);
}

// Reversed-Z depths of the opaque surface (x) and the nearest glass (y) at `pixel`, 0 for none.
fn march_depths(pixel: vec2<i32>) -> vec2<f32> {
    if HAS_GLASS {
        let linear = textureLoad(t_contact_depth, pixel, 0).rg;
        return vec2f(device_depth(linear.r), device_depth(linear.g));
    }
    return vec2f(textureLoad(t_depth, pixel, 0), 0.0);
}

@fragment
fn fs_contact_depth(in: VertexOutput) -> GlassTargets {
    return GlassTargets(vec4f(1.0), vec4f(linear_depth(textureLoad(t_depth, vec2<i32>(in.pos.xy), 0)), 0.0, 0.0, 0.0));
}

@fragment
fn fs_glass(in: GlassVertex) -> GlassTargets {
    if in.pos.z < textureLoad(t_depth, vec2<i32>(in.pos.xy), 0) {
        discard;
    }
    return GlassTargets(vec4f(in.transmittance, 1.0), vec4f(0.0, linear_depth(in.pos.z), 0.0, 0.0));
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

// Half resolution: texel q integrates full-resolution pixel 2q; `fs_blur` upsamples.
@fragment
fn fs_ao(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2f(textureDimensions(t_depth));
    let half_texel = vec2<i32>(in.pos.xy);
    let pixel = min(half_texel * 2, vec2<i32>(dims) - 1);
    let center = vec2f(pixel) + 0.5;
    let depth = textureLoad(t_depth, pixel, 0);
    if depth == 0.0 {
        return vec4f(1.0);
    }
    let p = world_pos(center, depth, dims);
    let n = load_normal(pixel);
    let v = normalize(camera.inv_view[3].xyz - p);

    let texel = length(world_pos(center + vec2f(1.0, 0.0), depth, dims) - p);
    let radius_px = min(RADIUS / max(texel, 1e-6), MAX_RADIUS_FRACTION * dims.y);
    if radius_px < 1.0 {
        return vec4f(1.0);
    }

    let cell = vec2<u32>(half_texel) & vec2<u32>(3u);
    let rotation = (bayer(cell) + 0.5) / 16.0;
    let jitter = (bayer(cell.yx) + 0.5) / 16.0;

    var visible = 0.0;
    var total = 0.0;
    for (var i = 0u; i < DIRECTIONS; i++) {
        let phi = (f32(i) + rotation) * PI / f32(DIRECTIONS);
        let dir = vec2f(cos(phi), sin(phi));
        // The slice's world-space direction: toward the pixel next along `dir`, made orthogonal to v.
        let toward = world_pos(center + dir, depth, dims) - p;
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
                let sample_px = floor(center + dir * (offset * s)) + 0.5;
                if any(sample_px < vec2f(0.0)) || any(sample_px >= dims) {
                    break;
                }
                let depths = march_depths(vec2<i32>(sample_px));
                if depths.x != 0.0 {
                    horizon[side] = max(horizon[side], horizon_cos(world_pos(sample_px, depths.x, dims), p, v));
                }
                if depths.y != 0.0 {
                    let cos_h = horizon_cos(world_pos(sample_px, depths.y, dims), p, v);
                    if cos_h > glass_horizon[side] {
                        glass_horizon[side] = cos_h;
                        let through = textureLoad(t_glass_transmittance, vec2<i32>(sample_px), 0).rgb;
                        glass_opacity[side] = 1.0 - dot(through, vec3f(0.2126, 0.7152, 0.0722));
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

    // The 4×4 half-resolution texels nearest the pixel, each at the full-resolution pixel it integrated.
    let raw_max = vec2<i32>(textureDimensions(t_raw)) - 1;
    let base = vec2<i32>(floor((vec2f(pixel) - 3.0) * 0.5));
    var sum = 0.0;
    var weight = 0.0;
    for (var y = 0; y < 4; y++) {
        for (var x = 0; x < 4; x++) {
            let tap_texel = clamp(base + vec2<i32>(x, y), vec2<i32>(0), raw_max);
            let tap = min(tap_texel * 2, vec2<i32>(dims) - 1);
            let tap_depth = textureLoad(t_depth, tap, 0);
            if tap_depth == 0.0 {
                continue;
            }
            let off_plane = abs(dot(world_pos(vec2f(tap) + 0.5, tap_depth, dims) - p, n));
            // Tent over the distance to the pixel, so the window doesn't jump by a texel every 2 pixels.
            let spread = 1.0 - abs(vec2f(tap - pixel)) / 5.0;
            let w = saturate(1.0 - off_plane / tolerance) * pow(saturate(dot(load_normal(tap), n)), 8.0) * spread.x * spread.y;
            sum += textureLoad(t_raw, tap_texel, 0).r * w;
            weight += w;
        }
    }
    return vec4f(select(textureLoad(t_raw, min(pixel / 2, raw_max), 0).r, sum / weight, weight > 1e-4));
}
