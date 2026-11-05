struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var albedo_tex: texture_2d<f32>;
@group(1) @binding(1)
var normal_tex: texture_2d<f32>;
// glTF layout: G = roughness, B = metallic.
@group(1) @binding(2)
var metallic_roughness_tex: texture_2d<f32>;
@group(1) @binding(3)
var tex_sampler: sampler;

// Mirrors rendering::arena::GpuModel.
struct Model {
    model: mat4x4<f32>,
    normal: mat4x4<f32>,
    // x: alpha cutoff (0 = never discard), y: 1 if double-sided, z: metallic, w: roughness.
    material: vec4<f32>,
    base_color: vec4<f32>,
};

@group(2) @binding(0)
var<storage, read> models: array<Model>;

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
    @location(4) @interpolate(flat) material: vec4<f32>,
    @location(5) @interpolate(flat) base_color: vec4<f32>,
};

struct GBufferOutput {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) material: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance: u32) -> VertexOutput {
    var out: VertexOutput;
    let m = models[instance];

    out.clip_position = camera.view_proj * m.model * vec4<f32>(in.position, 1.0);
    out.world_normal = normalize((m.normal * vec4<f32>(in.normal, 0.0)).xyz);
    out.color = in.color;
    out.uv = in.uv;
    out.world_tangent = vec4<f32>(normalize((m.model * vec4<f32>(in.tangent.xyz, 0.0)).xyz), in.tangent.w);
    out.material = m.material;
    out.base_color = m.base_color;

    return out;
}

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) front_facing: bool) -> GBufferOutput {
    var out: GBufferOutput;

    // Sample before discarding: implicit-derivative sampling must stay in uniform control flow.
    let albedo = textureSample(albedo_tex, tex_sampler, in.uv) * in.base_color * vec4<f32>(in.color, 1.0);
    let normal_sample = textureSample(normal_tex, tex_sampler, in.uv).rgb;
    let metallic_roughness = textureSample(metallic_roughness_tex, tex_sampler, in.uv);
    if albedo.a < in.material.x {
        discard;
    }

    var N = normalize(in.world_normal);
    if !front_facing && in.material.y > 0.5 {
        N = -N;
    }
    let T = normalize(in.world_tangent.xyz);
    let B = cross(N, T) * in.world_tangent.w;

    let tangent_normal = normalize(normal_sample * 2.0 - 1.0);
    let world_normal = normalize(T * tangent_normal.x + B * tangent_normal.y + N * tangent_normal.z);

    out.albedo = albedo;
    out.normal = vec4<f32>(world_normal * 0.5 + 0.5, 1.0);
    let roughness = metallic_roughness.g * in.material.w;
    let metallic = metallic_roughness.b * in.material.z;
    out.material = vec4<f32>(roughness, metallic, 1.0, 0.0);

    return out;
}
