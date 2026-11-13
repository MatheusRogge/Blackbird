// Adds reflections to the lit image: SSR where it hit, blurred by roughness, else the probe fallback.

@group(0) @binding(0) var t_lit:               texture_2d<f32>;
// rgb: reflection weight (0 = nothing to reflect), a: roughness. Written by lighting.wgsl.
@group(0) @binding(1) var t_specular_weight:   texture_2d<f32>;
// Probe-based reflection, already weighted and exposed. Written by lighting.wgsl.
@group(0) @binding(2) var t_specular_fallback: texture_2d<f32>;
// Premultiplied SSR color, a: confidence: `ssr_temporal`'s output, with `reflection_trace`'s
// traces filled in.
@group(0) @binding(3) var t_ssr:               texture_2d<f32>;
// G-buffer depth, reversed-Z.
@group(0) @binding(4) var t_depth:             texture_depth_2d;
// The `ssr` pass's own output (screen-space hits only), for the debug views.
@group(0) @binding(5) var t_ssr_raw:           texture_2d<f32>;

// Mirrors rendering::debug_flags::DebugFlagsGpu; only `view` is read here.
struct DebugFlags {
    disable_albedo:   u32,
    disable_indirect: u32,
    disable_direct:   u32,
    disable_shadows:  u32,
    view:             u32,
    disable_specular: u32,
    disable_reflections: u32,
    disable_ssr:      u32,
    disable_contact_shadows: u32,
    disable_shadow_mask: u32,
    disable_traced_reflections: u32,
}

@group(0) @binding(6) var<uniform> debug: DebugFlags;
@group(0) @binding(7) var s_linear:            sampler;

// rendering::debug_flags::GBufferView.
const VIEW_SSR:                u32 = 9u;
const VIEW_REFLECTIONS:        u32 = 10u;
const VIEW_REFLECTION_SOURCES: u32 = 11u;

// Blur radius at roughness 1; SSR stops at 0.4, so the widest blur is 40% of this.
const BLUR_RADIUS_PX: f32 = 32.0;
const GOLDEN_ANGLE:   f32 = 2.39996323;
const BLUR_TAPS:      u32 = 8u;
// A tap off the centre pixel's plane by more than this fraction of its depth gets no weight.
const PLANE_TOLERANCE: f32 = 0.02;
// A tap whose roughness differs from the centre's by more than this fades out, gone at twice it.
const ROUGHNESS_TOLERANCE: f32 = 0.1;

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
    let reflection_view = debug.view >= VIEW_SSR && debug.view <= VIEW_REFLECTION_SOURCES;
    if all(weight.rgb == vec3f(0.0)) {
        // Nothing reflects here: the reflection views show it as a dim grey of the scene.
        return select(lit, vec4f(vec3f(dot(lit.rgb, vec3f(0.2126, 0.7152, 0.0722)) * 0.15), lit.a), reflection_view);
    }
    if debug.view == VIEW_SSR {
        return vec4f(textureLoad(t_ssr_raw, px, 0).rgb, lit.a);
    }
    if debug.view == VIEW_REFLECTION_SOURCES {
        let screen = saturate(textureLoad(t_ssr_raw, px, 0).a);
        let resolved = saturate(textureLoad(t_ssr, px, 0).a);
        let traced = max(resolved - screen, 0.0);
        return vec4f(1.0 - max(resolved, screen), screen, traced, lit.a);
    }

    // Vogel disc, rotated per pixel. Reversed-Z depth is affine in screen space across a plane, so
    // taps are kept only where the depth matches the centre's plane: reflections don't bleed across
    // object borders or creases, while grazing floors still blur.
    let dims = vec2<f32>(textureDimensions(t_ssr));
    let radius = weight.a * BLUR_RADIUS_PX;
    // One rotation per 4×4 block: neighbouring pixels then read neighbouring texels, where a
    // rotation per pixel scattered every tap over the disc and the blur cost 3× as much.
    let rotation = interleaved_gradient_noise(floor(in.pos.xy * 0.25)) * 6.28318531;
    let depth = load_depth(px);
    let gradient = depth_gradient(px, depth);
    var ssr = textureLoad(t_ssr, px, 0);
    var total = 1.0;
    for (var i = 0u; i < BLUR_TAPS; i++) {
        let r = radius * sqrt((f32(i) + 0.5) / f32(BLUR_TAPS));
        let angle = f32(i) * GOLDEN_ANGLE + rotation;
        let offset = vec2f(cos(angle), sin(angle)) * r;
        let tap_px = vec2<i32>(in.pos.xy + offset);
        let tap_depth = load_depth(tap_px);
        let error = abs(tap_depth - (depth + dot(gradient, offset))) / depth;
        // Coplanar but a different material (a door's smooth window panel beside its rough
        // wood): the rough side's wide disc caught the panel's sharp, bright reflections in a few
        // taps and showed them as a spray of dots.
        let tap_roughness = textureLoad(t_specular_weight, clamp(tap_px, vec2<i32>(0), vec2<i32>(dims) - 1), 0).a;
        let material = 1.0 - smoothstep(ROUGHNESS_TOLERANCE, 2.0 * ROUGHNESS_TOLERANCE, abs(tap_roughness - weight.a));
        let w = (1.0 - smoothstep(0.5 * PLANE_TOLERANCE, PLANE_TOLERANCE, error)) * material;
        ssr += w * textureSampleLevel(t_ssr, s_linear, (in.pos.xy + offset) / dims, 0.0);
        total += w;
    }
    ssr /= total;

    let confidence = ssr.a;
    let reflection = ssr.rgb / max(confidence, 1e-4) * weight.rgb;
    let fallback = textureLoad(t_specular_fallback, px, 0).rgb;
    let reflected = mix(fallback, reflection, confidence);
    if debug.view == VIEW_REFLECTIONS {
        return vec4f(reflected, lit.a);
    }
    return vec4f(lit.rgb + reflected, lit.a);
}
