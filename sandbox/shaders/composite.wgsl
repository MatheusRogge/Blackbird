// Adds reflections to the lit image: SSR where it hit, blurred by roughness, else the probe fallback.

@group(0) @binding(0) var t_lit:               texture_2d<f32>;
// rgb: reflection weight (0 = nothing to reflect), a: roughness. Written by lighting.wgsl.
@group(0) @binding(1) var t_specular_weight:   texture_2d<f32>;
// Probe-based reflection, already weighted and exposed. Written by lighting.wgsl.
@group(0) @binding(2) var t_specular_fallback: texture_2d<f32>;
// Premultiplied SSR color, a: confidence. Written by ssr.wgsl.
@group(0) @binding(3) var t_ssr:               texture_2d<f32>;
// G-buffer depth, reversed-Z.
@group(0) @binding(4) var t_depth:             texture_depth_2d;
@group(0) @binding(5) var s_linear:            sampler;

// Blur radius at roughness 1; SSR stops at 0.4, so the widest blur is 40% of this.
const BLUR_RADIUS_PX: f32 = 32.0;
const GOLDEN_ANGLE:   f32 = 2.39996323;
const BLUR_TAPS:      u32 = 8u;
// A tap off the centre pixel's plane by more than this fraction of its depth gets no weight.
const PLANE_TOLERANCE: f32 = 0.02;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    return VertexOutput(vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0));
}

fn interleaved_gradient_noise(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715))));
}

fn load_depth(p: vec2<i32>) -> f32 {
    return textureLoad(t_depth, clamp(p, vec2<i32>(0), vec2<i32>(textureDimensions(t_depth)) - 1), 0);
}

// Screen-space depth gradient at `px`. Each axis takes the smaller one-sided difference, so a
// neighbour across an edge doesn't skew it.
fn depth_gradient(px: vec2<i32>, d: f32) -> vec2<f32> {
    let l = d - load_depth(px - vec2<i32>(1, 0));
    let r = load_depth(px + vec2<i32>(1, 0)) - d;
    let u = d - load_depth(px - vec2<i32>(0, 1));
    let b = load_depth(px + vec2<i32>(0, 1)) - d;
    return vec2f(select(r, l, abs(l) < abs(r)), select(b, u, abs(u) < abs(b)));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let px = vec2<i32>(in.pos.xy);
    let lit = textureLoad(t_lit, px, 0);
    let weight = textureLoad(t_specular_weight, px, 0);
    if all(weight.rgb == vec3f(0.0)) {
        return lit;
    }

    // Vogel disc, rotated per pixel. Reversed-Z depth is affine in screen space across a plane, so
    // taps are kept only where the depth matches the centre's plane: reflections don't bleed across
    // object borders or creases, while grazing floors still blur.
    let dims = vec2<f32>(textureDimensions(t_ssr));
    let radius = weight.a * BLUR_RADIUS_PX;
    let rotation = interleaved_gradient_noise(in.pos.xy) * 6.28318531;
    let depth = load_depth(px);
    let gradient = depth_gradient(px, depth);
    var ssr = textureLoad(t_ssr, px, 0);
    var total = 1.0;
    for (var i = 0u; i < BLUR_TAPS; i++) {
        let r = radius * sqrt((f32(i) + 0.5) / f32(BLUR_TAPS));
        let angle = f32(i) * GOLDEN_ANGLE + rotation;
        let offset = vec2f(cos(angle), sin(angle)) * r;
        let tap_depth = load_depth(vec2<i32>(in.pos.xy + offset));
        let error = abs(tap_depth - (depth + dot(gradient, offset))) / depth;
        let w = 1.0 - smoothstep(0.5 * PLANE_TOLERANCE, PLANE_TOLERANCE, error);
        ssr += w * textureSampleLevel(t_ssr, s_linear, (in.pos.xy + offset) / dims, 0.0);
        total += w;
    }
    ssr /= total;

    let confidence = ssr.a;
    let reflection = ssr.rgb / max(confidence, 1e-4) * weight.rgb;
    let fallback = textureLoad(t_specular_fallback, px, 0).rgb;
    return vec4f(lit.rgb + mix(fallback, reflection, confidence), lit.a);
}
