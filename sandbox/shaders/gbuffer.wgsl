struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var albedo_tex: texture_2d<f32>;
@group(1) @binding(1)
var normal_tex: texture_2d<f32>;
@group(1) @binding(2)
var tex_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) tangent: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) world_tangent: vec4<f32>,
};

struct GBufferOutput {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) material: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.world_normal = normalize(in.normal);
    out.color = in.color;
    out.uv = in.uv;
    out.world_tangent = in.tangent;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> GBufferOutput {
    var out: GBufferOutput;

    let N = normalize(in.world_normal);
    let T = normalize(in.world_tangent.xyz);
    let B = cross(N, T) * in.world_tangent.w;

    let normal_sample = textureSample(normal_tex, tex_sampler, in.uv).rgb;
    let tangent_normal = normalize(normal_sample * 2.0 - 1.0);
    let world_normal = normalize(T * tangent_normal.x + B * tangent_normal.y + N * tangent_normal.z);

    out.albedo = textureSample(albedo_tex, tex_sampler, in.uv);
    out.normal = vec4<f32>(world_normal * 0.5 + 0.5, 1.0);
    out.material = vec4<f32>(0.5, 0.0, 1.0, 1.0);

    return out;
}
