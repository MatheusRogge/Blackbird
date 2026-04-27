// GBuffer inputs
@group(0) @binding(0) var t_albedo:   texture_2d<f32>;
@group(0) @binding(1) var t_normal:   texture_2d<f32>;
@group(0) @binding(2) var t_material: texture_2d<f32>;
@group(0) @binding(3) var t_depth:    texture_depth_2d;
@group(0) @binding(4) var s_linear:   sampler;

// Cluster + light data
struct ClusterParams {
    tile_w: f32,
    tile_h: f32,
    z_near: f32,
    z_far: f32,
    log_ratio_recip: f32,
    num_point_lights: u32,
    inv_proj_00: f32,
    inv_proj_11: f32,
    screen_w: f32,
    screen_h: f32,
    debug_mode: u32,
    _pad: u32,
}

struct GpuPointLight {
    position_vs: vec3<f32>,
    radius: f32,
    color: vec3<f32>,
    intensity: f32,
}

@group(1) @binding(0) var<uniform>         cp:           ClusterParams;
@group(1) @binding(1) var<storage, read>   point_lights: array<GpuPointLight>;
@group(1) @binding(2) var<storage, read>   light_grid:   array<u32>;
@group(1) @binding(3) var<storage, read>   light_indices: array<u32>;

const CLUSTER_X: u32 = 16u;
const CLUSTER_Y: u32 = 9u;
const CLUSTER_Z: u32 = 24u;
const MAX_LIGHTS_PER_CLUSTER: u32 = 128u;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Fullscreen triangle — no vertex buffer needed.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var out: VertexOutput;
    out.pos = vec4<f32>(positions[vi], 0.0, 1.0);
    out.uv  = uvs[vi];
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let frag_xy = vec2<i32>(i32(in.pos.x), i32(in.pos.y));

    let albedo   = textureSample(t_albedo,   s_linear, in.uv).rgb;
    let norm_enc = textureSample(t_normal,   s_linear, in.uv).rgb;
    let depth_v  = textureLoad(t_depth, frag_xy, 0);

    // Reversed-Z: depth_v == 0 means "at far plane / sky" — no geometry.
    if depth_v == 0.0 {
        return vec4<f32>(albedo * 0.03, 1.0);
    }

    // Reconstruct view-space depth from reversed-Z depth buffer value.
    // z_view_pos = z_near * z_far / (depth * (z_far - z_near) + z_near)
    let depth_vs = cp.z_near * cp.z_far / (depth_v * (cp.z_far - cp.z_near) + cp.z_near);

    // Reconstruct view-space position.
    let ndc_x = in.pos.x / cp.screen_w * 2.0 - 1.0;
    let ndc_y = 1.0 - in.pos.y / cp.screen_h * 2.0;
    let pos_vs = vec3<f32>(
        ndc_x * depth_vs * cp.inv_proj_00,
        ndc_y * depth_vs * cp.inv_proj_11,
        -depth_vs,
    );

    // Decode normal (stored as (N + 1) / 2 in GBuffer)
    let N = normalize(norm_enc * 2.0 - vec3<f32>(1.0));

    // Cluster lookup
    let cx = u32(in.pos.x / cp.tile_w);
    let cy = u32(in.pos.y / cp.tile_h);
    let cz = min(u32(log(depth_vs / cp.z_near) * cp.log_ratio_recip), CLUSTER_Z - 1u);
    let cluster_idx = cx + CLUSTER_X * (cy + CLUSTER_Y * cz);

    let light_count = light_grid[cluster_idx];

    // Debug: visualise light density per cluster.
    if cp.debug_mode == 1u {
        let heat = f32(light_count) / f32(MAX_LIGHTS_PER_CLUSTER);
        return vec4<f32>(heat, 0.3, 1.0 - heat, 1.0);
    }

    var color = albedo * 0.03; // ambient

    for (var i = 0u; i < light_count; i++) {
        let li    = light_indices[cluster_idx * MAX_LIGHTS_PER_CLUSTER + i];
        let light = point_lights[li];

        let to_light = light.position_vs - pos_vs;
        let dist     = length(to_light);
        if dist >= light.radius {
            continue;
        }

        let L           = to_light / dist;
        let diff        = max(dot(N, L), 0.0);
        let attenuation = 1.0 - dist / light.radius;
        color += albedo * light.color * (light.intensity * diff * attenuation * attenuation);
    }

    return vec4<f32>(color, 1.0);
}
