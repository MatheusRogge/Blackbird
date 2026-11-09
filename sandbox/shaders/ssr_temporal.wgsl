// Temporal accumulation for SSR (rendering::pass::ssr_temporal). The ray start is jittered per frame
// in ssr.wgsl; this blends each frame into a reprojected history, so the jitter averages out
// instead of showing as noise. A reflection moves with the reflected point's mirror image, which
// lies the hit's ray length behind the surface along the view ray, so the history is reprojected
// from there (Stachowiak, "Stochastic Screen-Space Reflections", 2015); reprojecting the surface
// itself smeared reflections whenever the camera moved. History is clipped to the current
// neighbourhood's colour spread (variance clipping), so disocclusions don't ghost.

// Premultiplied SSR colour, a: confidence. Written by ssr.wgsl.
@group(0) @binding(0) var t_ssr:             texture_2d<f32>;
// This pass's output from the previous frame.
@group(0) @binding(1) var t_history:         texture_2d<f32>;
@group(0) @binding(2) var t_depth:           texture_depth_2d;
// rgb: reflection weight (0 = nothing to reflect). Written by lighting.wgsl.
@group(0) @binding(3) var t_specular_weight: texture_2d<f32>;
@group(0) @binding(4) var<uniform> camera:   CameraUniform;
@group(0) @binding(5) var s_linear:          sampler;
// World distance from the surface to the SSR hit, 0 without one. Written by ssr.wgsl.
@group(0) @binding(6) var t_ray_length:      texture_2d<f32>;

struct CameraUniform {
    view_proj:      mat4x4<f32>,
    inv_view_proj:  mat4x4<f32>,
    inv_view:       mat4x4<f32>,
    depth_params:   vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

// Share of the current frame in the result: about 10 frames of history.
const CURRENT_WEIGHT: f32 = 0.1;
// Width of the clip box in standard deviations of the 3×3 neighbourhood.
const CLIP_SIGMA: f32 = 1.25;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    return VertexOutput(vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let px = vec2<i32>(in.pos.xy);
    let current = textureLoad(t_ssr, px, 0);
    if all(textureLoad(t_specular_weight, px, 0).rgb == vec3f(0.0)) {
        return vec4f(0.0);
    }
    let depth = textureLoad(t_depth, px, 0);
    if depth == 0.0 {
        return current;
    }

    var m1 = vec4f(0.0);
    var m2 = vec4f(0.0);
    var lo = vec4f(1e10);
    var hi = vec4f(-1e10);
    // Jittered rays miss now and then: average the hits around the pixel.
    var ray_sum = 0.0;
    var ray_hits = 0.0;
    let dims = vec2<f32>(textureDimensions(t_ssr));
    let last = vec2<i32>(dims) - 1;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let p = clamp(px + vec2<i32>(x, y), vec2<i32>(0), last);
            let s = textureLoad(t_ssr, p, 0);
            m1 += s;
            m2 += s * s;
            if x != 0 || y != 0 {
                lo = min(lo, s);
                hi = max(hi, s);
            }
            let ray = textureLoad(t_ray_length, p, 0).r;
            if ray > 0.0 {
                ray_sum += ray;
                ray_hits += 1.0;
            }
        }
    }

    let uv = in.pos.xy / dims;
    let world4 = camera.inv_view_proj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let surface = world4.xyz / world4.w;
    let eye = camera.inv_view[3].xyz;
    var point = surface;
    if ray_hits > 0.0 {
        point += normalize(surface - eye) * (ray_sum / ray_hits);
    }
    let prev = camera.prev_view_proj * vec4f(point, 1.0);
    if prev.w <= 0.0 {
        return current;
    }
    let prev_uv = (prev.xy / prev.w) * vec2f(0.5, -0.5) + 0.5;
    if any(prev_uv < vec2f(0.0)) || any(prev_uv > vec2f(1.0)) {
        return current;
    }

    // A ray that lands on a tiny highlight none of its neighbours see is a firefly: keep it
    // within the neighbours' range so it can't brand itself into the history.
    let current_clamped = clamp(current, lo, hi);
    let mean = m1 / 9.0;
    let sigma = sqrt(max(m2 / 9.0 - mean * mean, vec4f(0.0)));
    let history = clamp(
        textureSampleLevel(t_history, s_linear, prev_uv, 0.0),
        mean - CLIP_SIGMA * sigma,
        mean + CLIP_SIGMA * sigma,
    );
    return mix(history, current_clamped, CURRENT_WEIGHT);
}
