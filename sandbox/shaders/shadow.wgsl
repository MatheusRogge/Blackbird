struct LightUniform {
    view_proj: mat4x4<f32>,
}

// Mirrors rendering::arena::GpuModel.
struct Model {
    model: mat4x4<f32>,
    normal: mat4x4<f32>,
    // x: alpha cutoff (0 = never discard), y: 1 if double-sided.
    material: vec4<f32>,
}

@group(0) @binding(0) var<uniform> light: LightUniform;
@group(1) @binding(0) var<storage, read> models: array<Model>;

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @builtin(instance_index) instance: u32,
) -> @builtin(position) vec4<f32> {
    return light.view_proj * models[instance].model * vec4<f32>(position, 1.0);
}

// Alpha-tested and double-sided meshes: drawn without culling, discarding cut-out texels.
@group(2) @binding(0) var albedo_tex: texture_2d<f32>;
@group(2) @binding(1) var albedo_sampler: sampler;

struct MaskedOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) cutoff: f32,
}

@vertex
fn vs_masked(
    @location(0) position: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @builtin(instance_index) instance: u32,
) -> MaskedOutput {
    let m = models[instance];
    var out: MaskedOutput;
    out.position = light.view_proj * m.model * vec4<f32>(position, 1.0);
    out.uv = uv;
    out.cutoff = m.material.x;
    return out;
}

@fragment
fn fs_masked(in: MaskedOutput) {
    if textureSample(albedo_tex, albedo_sampler, in.uv).a < in.cutoff {
        discard;
    }
}
