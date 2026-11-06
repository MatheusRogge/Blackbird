// Screen-space reflections. Traces the reflection ray through a Hi-Z depth pyramid (built by the
// `ssr` pass, rendering/src/pass/hiz.wgsl) and writes the color it hits, premultiplied by a
// confidence the composite pass blends with. `ssr_temporal` accumulates it over frames. The color
// comes from last frame's composite output where the hit was on screen then, so reflected surfaces
// show their own reflections (each frame adds a bounce); otherwise from this frame's `lit`.

// Hi-Z pyramid: level 0 is the G-buffer depth (reversed-Z), each further level the max (nearest)
// of the texels it covers.
@group(0) @binding(0) var t_hiz:             texture_2d<f32>;
@group(0) @binding(1) var t_normal:          texture_2d<f32>;
@group(0) @binding(2) var t_lit:             texture_2d<f32>;
// rgb: reflection weight (0 = nothing to reflect), a: roughness. Written by lighting.wgsl.
@group(0) @binding(3) var t_specular_weight: texture_2d<f32>;
@group(0) @binding(4) var<uniform> camera:   CameraUniform;
@group(0) @binding(5) var<uniform> debug:    DebugFlags;
@group(0) @binding(6) var s_linear:          sampler;
// Last frame's composite output (rendering::pass::prev_frame); `lit` on the first frame.
@group(0) @binding(7) var t_prev:            texture_2d<f32>;

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
    view_proj:      mat4x4<f32>,
    inv_view_proj:  mat4x4<f32>,
    inv_view:       mat4x4<f32>,
    // near, far, frame index
    depth_params:   vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

// Rougher surfaces get only the probe fallback; SSR fades out from FADE_ROUGHNESS.
const MAX_ROUGHNESS:  f32 = 0.4;
const FADE_ROUGHNESS: f32 = 0.25;
const MAX_DISTANCE:   f32 = 50.0;
// Hi-Z cell visits per ray: empty space is crossed in large cells, so this covers long rays.
const MAX_ITERATIONS: u32 = 64u;
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

// Hi-Z traversal in screen space, where NDC depth is linear along the ray. `p0` / `dir` are
// (pixel x, pixel y, NDC depth) at t = 0 and the change up to t = 1. A cell whose nearest depth is
// behind the ray over the whole stretch the ray spends in it is skipped and the next one is
// tested a level up; otherwise the trace descends, down to single pixels. Returns the hit's t, or
// −1 for a miss.
fn trace_hiz(p0: vec3<f32>, dir: vec3<f32>, t_end: f32, t_start: f32) -> f32 {
    let max_level = i32(textureNumLevels(t_hiz)) - 1;
    let eps = 0.01 / max(length(dir.xy), 1.0);
    let step_up = select(vec2f(0.0), vec2f(1.0), dir.xy > vec2f(0.0));
    var t = t_start;
    var level = 0;
    for (var i = 0u; i < MAX_ITERATIONS; i++) {
        if t > t_end {
            break;
        }
        let p = p0 + dir * t;
        let cell_size = f32(1 << u32(level));
        let cell = min(vec2<i32>(p.xy / cell_size), vec2<i32>(textureDimensions(t_hiz, level)) - 1);
        let nearest = textureLoad(t_hiz, cell, level).r;

        let bound = (vec2f(cell) + step_up) * cell_size;
        let tx = select(1e10, (bound.x - p0.x) / dir.x, abs(dir.x) > 1e-6);
        let ty = select(1e10, (bound.y - p0.y) / dir.y, abs(dir.y) > 1e-6);
        let t_exit = min(max(min(tx, ty), t), t_end);
        // Reversed-Z: smaller is farther. The ray's farthest depth in the cell is at one end.
        let z_far = min(p.z, p0.z + dir.z * t_exit);
        if z_far > nearest {
            t = t_exit + eps;
            level = min(level + 1, max_level);
            continue;
        }
        if level > 0 {
            level -= 1;
            continue;
        }
        // Level 0: the ray reaches this pixel's surface. Accept it if the ray isn't too far behind
        // (a thin object it passed behind), else keep walking.
        if nearest > 0.0 {
            let surface_w = linear_depth(nearest);
            if linear_depth(z_far) - surface_w < THICKNESS * surface_w {
                let t_surface = select(t, (nearest - p0.z) / dir.z, abs(dir.z) > 1e-9);
                return clamp(t_surface, t, t_exit);
            }
        }
        t = t_exit + eps;
    }
    return -1.0;
}

fn hit_color(hit_px: vec2<f32>, hit_uv: vec2<f32>) -> vec3<f32> {
    let hit_depth = textureLoad(t_hiz, vec2<i32>(hit_px), 0).r;
    let world4 = camera.inv_view_proj * vec4f(hit_uv.x * 2.0 - 1.0, 1.0 - hit_uv.y * 2.0, hit_depth, 1.0);
    let prev = camera.prev_view_proj * vec4f(world4.xyz / world4.w, 1.0);
    if prev.w > 0.0 {
        let prev_uv = (prev.xy / prev.w) * vec2f(0.5, -0.5) + 0.5;
        if all(prev_uv >= vec2f(0.0)) && all(prev_uv <= vec2f(1.0)) {
            return textureSampleLevel(t_prev, s_linear, prev_uv, 0.0).rgb;
        }
    }
    return textureSampleLevel(t_lit, s_linear, hit_uv, 0.0).rgb;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let px = vec2<i32>(in.pos.xy);
    let dims = vec2<f32>(textureDimensions(t_hiz));

    let depth = textureLoad(t_hiz, px, 0).r;
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

    let h0 = camera.view_proj * vec4f(pos, 1.0);
    let h1 = camera.view_proj * vec4f(pos + R * dist, 1.0);
    let p0 = vec3f(in.pos.xy, h0.z / h0.w);
    let dir = vec3f(clip_to_pixel(h1, dims), h1.z / h1.w) - p0;
    let len_px = length(dir.xy);
    if len_px < 1.0 {
        return vec4f(0.0);
    }

    // Stop where the ray leaves the screen.
    var t_end = 1.0;
    if dir.x > 0.0 { t_end = min(t_end, (dims.x - 0.5 - p0.x) / dir.x); }
    if dir.x < 0.0 { t_end = min(t_end, (0.5 - p0.x) / dir.x); }
    if dir.y > 0.0 { t_end = min(t_end, (dims.y - 0.5 - p0.y) / dir.y); }
    if dir.y < 0.0 { t_end = min(t_end, (0.5 - p0.y) / dir.y); }

    // Start a jittered 1–2 pixels out so the ray leaves its own surface; the jitter varies per
    // frame for `ssr_temporal`.
    let jitter = interleaved_gradient_noise(in.pos.xy + 5.588238 * camera.depth_params.z);
    let hit = trace_hiz(p0, dir, t_end, (1.0 + jitter) / len_px);
    if hit < 0.0 {
        return vec4f(0.0);
    }

    let hit_px = p0.xy + dir.xy * hit;
    let hit_N = normalize(textureLoad(t_normal, vec2<i32>(hit_px), 0).rgb * 2.0 - 1.0);
    if dot(hit_N, R) > 0.0 {
        return vec4f(0.0);
    }
    let hit_uv = hit_px / dims;
    let color = hit_color(hit_px, hit_uv);

    let edge = min(hit_uv, 1.0 - hit_uv);
    let confidence = smoothstep(0.0, 0.08, min(edge.x, edge.y))
        * (1.0 - smoothstep(FADE_ROUGHNESS, MAX_ROUGHNESS, roughness))
        * (1.0 - smoothstep(0.1, 0.5, R_vs.z))
        * (1.0 - smoothstep(0.7, 1.0, hit));
    return vec4f(color * confidence, confidence);
}
