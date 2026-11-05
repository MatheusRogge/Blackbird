// Screen-space reflections. Marches the reflection ray across the screen with perspective-correct
// depth and writes the lit color it hits, premultiplied by a confidence the composite pass blends with.

@group(0) @binding(0) var t_depth:           texture_depth_2d;
@group(0) @binding(1) var t_normal:          texture_2d<f32>;
@group(0) @binding(2) var t_lit:             texture_2d<f32>;
// rgb: reflection weight (0 = nothing to reflect), a: roughness. Written by lighting.wgsl.
@group(0) @binding(3) var t_specular_weight: texture_2d<f32>;
@group(0) @binding(4) var<uniform> camera:   CameraUniform;
@group(0) @binding(5) var<uniform> debug:    DebugFlags;
@group(0) @binding(6) var s_linear:          sampler;

// Mirrors rendering::debug_flags::DebugFlagsGpu.
struct DebugFlags {
    disable_albedo:      u32,
    disable_indirect:    u32,
    disable_direct:      u32,
    disable_shadows:     u32,
    view:                u32,
    disable_specular:    u32,
    disable_reflections: u32,
    disable_ssr:         u32,
}

struct CameraUniform {
    view_proj:     mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    inv_view:      mat4x4<f32>,
    // near, far
    depth_params:  vec4<f32>,
}

// Rougher surfaces get only the probe fallback; SSR fades out from FADE_ROUGHNESS.
const MAX_ROUGHNESS:  f32 = 0.4;
const FADE_ROUGHNESS: f32 = 0.25;
const MAX_DISTANCE:   f32 = 50.0;
const MAX_STEPS:      u32 = 64u;
const STEP_PIXELS:    f32 = 2.0;
const REFINE_STEPS:   u32 = 6u;
// A ray counts as hitting a surface when it is behind it by less than this fraction of its depth.
const THICKNESS:      f32 = 0.03;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    return VertexOutput(vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0));
}

// Reversed-Z depth to view-space distance along -Z (clip w).
fn linear_depth(d: f32) -> f32 {
    let n = camera.depth_params.x;
    let f = camera.depth_params.y;
    return n * f / (d * (f - n) + n);
}

fn interleaved_gradient_noise(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715))));
}

fn clip_to_pixel(h: vec4<f32>, dims: vec2<f32>) -> vec2<f32> {
    let ndc = h.xy / h.w;
    return vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * dims;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let px = vec2<i32>(in.pos.xy);
    let dims = vec2<f32>(textureDimensions(t_depth));

    let depth = textureLoad(t_depth, px, 0);
    let weight = textureLoad(t_specular_weight, px, 0);
    let roughness = weight.a;
    if debug.disable_ssr != 0u || depth == 0.0 || all(weight.rgb == vec3f(0.0)) || roughness > MAX_ROUGHNESS {
        return vec4f(0.0);
    }

    let uv = in.pos.xy / dims;
    let pos4 = camera.inv_view_proj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let pos = pos4.xyz / pos4.w;
    let N = normalize(textureLoad(t_normal, px, 0).rgb * 2.0 - 1.0);
    let cam_pos = camera.inv_view[3].xyz;
    let V = normalize(cam_pos - pos);
    let R = reflect(-V, N);

    // inv_view is rigid, so v * M == transpose(M) * v maps world -> view.
    let to_view = mat3x3<f32>(camera.inv_view[0].xyz, camera.inv_view[1].xyz, camera.inv_view[2].xyz);
    let R_vs = R * to_view;
    let pos_vs = (pos - cam_pos) * to_view;

    // Keep the ray in front of the near plane.
    var dist = MAX_DISTANCE;
    if R_vs.z > 0.0 {
        dist = min(dist, (-1.01 * camera.depth_params.x - pos_vs.z) / R_vs.z);
    }
    if dist <= 0.0 {
        return vec4f(0.0);
    }

    // March in screen space, interpolating 1/w so depth stays perspective-correct.
    let h0 = camera.view_proj * vec4f(pos, 1.0);
    let h1 = camera.view_proj * vec4f(pos + R * dist, 1.0);
    let k0 = 1.0 / h0.w;
    let k1 = 1.0 / h1.w;
    let p0 = clip_to_pixel(h0, dims);
    let delta = clip_to_pixel(h1, dims) - p0;
    let steps = u32(clamp(length(delta) / STEP_PIXELS, 1.0, f32(MAX_STEPS)));
    let jitter = interleaved_gradient_noise(in.pos.xy);

    var lo = 0.0;
    var hi = -1.0;
    for (var i = 1u; i <= steps; i++) {
        let s = (f32(i) - jitter) / f32(steps);
        let p = p0 + delta * s;
        if any(p < vec2f(0.0)) || any(p >= dims) {
            break;
        }
        let scene_d = textureLoad(t_depth, vec2<i32>(p), 0);
        if scene_d > 0.0 {
            let scene_w = linear_depth(scene_d);
            let behind = 1.0 / mix(k0, k1, s) - scene_w;
            if behind > 0.0 && behind < THICKNESS * scene_w {
                hi = s;
                break;
            }
        }
        lo = s;
    }
    if hi < 0.0 {
        return vec4f(0.0);
    }

    for (var i = 0u; i < REFINE_STEPS; i++) {
        let mid = 0.5 * (lo + hi);
        let scene_d = textureLoad(t_depth, vec2<i32>(p0 + delta * mid), 0);
        if scene_d > 0.0 && 1.0 / mix(k0, k1, mid) > linear_depth(scene_d) {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    let hit_px = p0 + delta * hi;
    let hit_N = normalize(textureLoad(t_normal, vec2<i32>(hit_px), 0).rgb * 2.0 - 1.0);
    if dot(hit_N, R) > 0.0 {
        return vec4f(0.0);
    }
    let hit_uv = hit_px / dims;
    let color = textureSampleLevel(t_lit, s_linear, hit_uv, 0.0).rgb;

    let edge = min(hit_uv, 1.0 - hit_uv);
    let confidence = smoothstep(0.0, 0.08, min(edge.x, edge.y))
        * (1.0 - smoothstep(FADE_ROUGHNESS, MAX_ROUGHNESS, roughness))
        * (1.0 - smoothstep(0.1, 0.5, R_vs.z))
        * (1.0 - smoothstep(0.7, 1.0, hi));
    return vec4f(color * confidence, confidence);
}
