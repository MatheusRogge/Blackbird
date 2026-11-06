// Mip chain generation for uploaded material textures (rendering::pass::mipmap).

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var<uniform> alpha_scale: vec4<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
    var out: VertexOutput;
    out.pos = vec4f(uv * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0);
    out.uv = uv;
    return out;
}

// One bilinear tap at the destination texel center averages the 2×2 source texels under it.
// sRGB views decode before filtering and encode on write, so the average is in linear space.
@fragment
fn fs_downsample(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSampleLevel(src, src_sampler, in.uv, 0.0);
}

@fragment
fn fs_downsample_normal(in: VertexOutput) -> @location(0) vec4<f32> {
    let c = textureSampleLevel(src, src_sampler, in.uv, 0.0);
    let n = c.xyz * 2.0 - 1.0;
    return vec4f(select(n / length(n), vec3f(0.0, 0.0, 1.0), dot(n, n) < 1e-8) * 0.5 + 0.5, c.a);
}

// Copies an unscaled level into the final texture with its alpha-coverage scale applied.
@fragment
fn fs_scale_alpha(in: VertexOutput) -> @location(0) vec4<f32> {
    let c = textureSampleLevel(src, src_sampler, in.uv, 0.0);
    return vec4f(c.rgb, saturate(c.a * alpha_scale.x));
}
