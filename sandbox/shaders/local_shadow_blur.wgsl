// Edge-aware 5×5 blur of the local shadow mask: neighbors on a different surface (depth or normal)
// are left out, so shadows don't bleed across silhouettes.

@group(0) @binding(0) var t_mask:   texture_2d<f32>;
@group(0) @binding(1) var t_depth:  texture_depth_2d;
@group(0) @binding(2) var t_normal: texture_2d<f32>;
@group(0) @binding(3) var<uniform> camera: CameraUniform;
@group(0) @binding(4) var s_linear: sampler;

struct CameraUniform {
    view_proj:     mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    inv_view:      mat4x4<f32>,
    // near, far
    depth_params:  vec4<f32>,
}

const RADIUS: i32 = 2;
// Neighbors further than this fraction of the center's depth count as another surface.
const DEPTH_TOLERANCE: f32 = 0.02;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    return VertexOutput(vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0));
}

fn linear_depth(d: f32) -> f32 {
    let n = camera.depth_params.x;
    let f = camera.depth_params.y;
    return n * f / (d * (f - n) + n);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let px = vec2<i32>(in.pos.xy);
    let center_d = textureLoad(t_depth, px, 0);
    if center_d == 0.0 {
        return vec4f(1.0);
    }
    let dims = vec2<i32>(textureDimensions(t_mask));
    let z = linear_depth(center_d);
    let N = textureLoad(t_normal, px, 0).rgb * 2.0 - 1.0;

    var sum = vec4f(0.0);
    var weight = 0.0;
    for (var y = -RADIUS; y <= RADIUS; y++) {
        for (var x = -RADIUS; x <= RADIUS; x++) {
            let p = clamp(px + vec2<i32>(x, y), vec2<i32>(0), dims - 1);
            let d = textureLoad(t_depth, p, 0);
            if d == 0.0 { continue; }
            let dz = abs(linear_depth(d) - z) / (DEPTH_TOLERANCE * z);
            let n = textureLoad(t_normal, p, 0).rgb * 2.0 - 1.0;
            let w = exp(-f32(x * x + y * y) / 4.0) * max(1.0 - dz, 0.0) * pow(max(dot(n, N), 0.0), 8.0);
            sum += textureLoad(t_mask, p, 0) * w;
            weight += w;
        }
    }
    return select(textureLoad(t_mask, px, 0), sum / weight, weight > 0.0);
}
