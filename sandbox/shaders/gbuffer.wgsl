struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct GBufferOutput {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) material: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    out.color = vec3(1.0, 0.0, 0.0);
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.world_normal = normalize(in.normal);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> GBufferOutput {
    var out: GBufferOutput;

    out.albedo = vec4<f32>(in.color, 1.0);
    out.normal = vec4<f32>(normalize(in.world_normal) * 0.5 + 0.5, 1.0);
    out.material = vec4<f32>(0.5, 0.0, 1.0, 1.0);

    return out;
}

