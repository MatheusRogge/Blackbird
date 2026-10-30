struct LightUniform {
    view_proj: mat4x4<f32>,
}

// Mirrors rendering::arena::GpuModel.
struct Model {
    model: mat4x4<f32>,
    normal: mat4x4<f32>,
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
