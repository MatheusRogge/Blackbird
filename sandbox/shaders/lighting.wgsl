// GBuffer inputs
@group(0) @binding(0) var t_albedo:   texture_2d<f32>;
@group(0) @binding(1) var t_normal:   texture_2d<f32>;
@group(0) @binding(2) var t_material: texture_2d<f32>;
@group(0) @binding(3) var t_depth:    texture_depth_2d;
@group(0) @binding(4) var s_linear:   sampler;

// Probe atlases + params
const PROBES_PER_ROW: u32 = 30u;
const PROBES_PER_COL: u32 = 29u;

struct ProbeGridParams {
    grid_dim:               vec3<u32>,
    probe_count:            u32,
    grid_origin:            vec3<f32>,
    max_ray_distance:       f32,
    probe_spacing:          f32,
    rays_per_probe:         u32,
    irradiance_texel_size:  u32,
    visibility_texel_size:  u32,
    hysteresis:             f32,
    normal_bias:            f32,
    view_bias:              f32,
    _pad:                   f32,
}

struct CameraUniform {
    view_proj:     mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    inv_view:      mat4x4<f32>,
}

@group(3) @binding(0) var irradiance_atlas: texture_2d<f32>;
@group(3) @binding(1) var visibility_atlas:  texture_2d<f32>;
@group(3) @binding(2) var<uniform> probe_params: ProbeGridParams;
@group(3) @binding(3) var probe_sampler: sampler;
@group(3) @binding(4) var<uniform> camera: CameraUniform;

struct DebugFlags {
    disable_albedo:   u32,
    disable_indirect: u32,
    disable_direct:   u32,
    disable_shadows:  u32,
}

@group(4) @binding(0) var<uniform> debug: DebugFlags;

fn octahedral_encode(n: vec3<f32>) -> vec2<f32> {
    let l1 = abs(n.x) + abs(n.y) + abs(n.z);
    var p = n.xy / l1;
    if n.z < 0.0 {
        let sp = 1.0 - abs(p.yx);
        p = vec2f(
            select(-sp.x, sp.x, p.x >= 0.0),
            select(-sp.y, sp.y, p.y >= 0.0),
        );
    }
    return p * 0.5 + 0.5;
}

fn irradiance_atlas_uv(probe_index: u32, oct_uv: vec2<f32>) -> vec2<f32> {
    let ts    = f32(probe_params.irradiance_texel_size);
    let tile  = ts + 2.0;
    let col   = f32(probe_index % PROBES_PER_ROW);
    let row   = f32(probe_index / PROBES_PER_ROW);
    let px    = oct_uv * ts + 1.0;
    let aw    = f32(PROBES_PER_ROW) * tile;
    let ah    = f32(PROBES_PER_COL) * tile;
    return vec2f((col * tile + px.x) / aw, (row * tile + px.y) / ah);
}

fn visibility_atlas_uv(probe_index: u32, oct_uv: vec2<f32>) -> vec2<f32> {
    let ts    = f32(probe_params.visibility_texel_size);
    let tile  = ts + 2.0;
    let col   = f32(probe_index % PROBES_PER_ROW);
    let row   = f32(probe_index / PROBES_PER_ROW);
    let px    = oct_uv * ts + 1.0;
    let aw    = f32(PROBES_PER_ROW) * tile;
    let ah    = f32(PROBES_PER_COL) * tile;
    return vec2f((col * tile + px.x) / aw, (row * tile + px.y) / ah);
}

fn chebyshev_weight(mean_d: f32, mean_d2: f32, dist: f32) -> f32 {
    if dist <= mean_d { return 1.0; }
    let variance = max(mean_d2 - mean_d * mean_d, 1e-6);
    let d = dist - mean_d;
    return variance / (variance + d * d);
}

fn sample_probe_irradiance(pos_ws: vec3<f32>, N_ws: vec3<f32>, view_dir: vec3<f32>) -> vec3<f32> {
    let biased_pos = pos_ws
        + N_ws    * probe_params.normal_bias
        + view_dir * probe_params.view_bias;
    let grid_pos = (biased_pos - probe_params.grid_origin) / probe_params.probe_spacing;
    let base_f   = floor(grid_pos);
    let base_i   = vec3<i32>(base_f);
    let frac     = grid_pos - base_f;
    let dim      = vec3<i32>(probe_params.grid_dim);
    let oct_uv   = octahedral_encode(N_ws);

    var irradiance   = vec3f(0.0);
    var weight_total = 0.0;

    for (var dz = 0; dz <= 1; dz++) {
        for (var dy = 0; dy <= 1; dy++) {
            for (var dx = 0; dx <= 1; dx++) {
                let ci = base_i + vec3<i32>(dx, dy, dz);
                if any(ci < vec3<i32>(0)) || any(ci >= dim) { continue; }

                let probe_index = u32(ci.x) + probe_params.grid_dim.x
                    * (u32(ci.y) + probe_params.grid_dim.y * u32(ci.z));

                let wx = select(1.0 - frac.x, frac.x, dx == 1);
                let wy = select(1.0 - frac.y, frac.y, dy == 1);
                let wz = select(1.0 - frac.z, frac.z, dz == 1);
                let w_tri = wx * wy * wz;

                let probe_ws = probe_params.grid_origin
                    + vec3f(f32(ci.x), f32(ci.y), f32(ci.z)) * probe_params.probe_spacing;
                let probe_dist = length(biased_pos - probe_ws);

                let vis_uv = visibility_atlas_uv(probe_index, oct_uv);
                let vis    = textureSample(visibility_atlas, probe_sampler, vis_uv).rg;
                let w_vis  = chebyshev_weight(vis.r, vis.g, probe_dist);

                let irr_uv = irradiance_atlas_uv(probe_index, oct_uv);
                let irr    = textureSample(irradiance_atlas, probe_sampler, irr_uv).rgb;

                let w = max(w_tri * w_vis, 0.0001);
                irradiance   += irr * w;
                weight_total += w;
            }
        }
    }

    return irradiance / max(weight_total, 1e-5);
}

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
    num_sky_lights: u32,
    inv_view: mat4x4<f32>,
    // Pre-exposure: scales the physical radiance written to `lit`.
    exposure: f32,
}

const PI: f32 = 3.14159265;

// KHR_lights_punctual recommended falloff: inverse-square, smoothly windowed to zero at `range`.
fn range_attenuation(dist: f32, range: f32) -> f32 {
    let r = dist / range;
    let window = clamp(1.0 - r * r * r * r, 0.0, 1.0);
    return window * window / max(dist * dist, 1e-4);
}

struct GpuPointLight {
    position_vs: vec3<f32>,
    radius: f32,
    color: vec3<f32>,
    intensity: f32,
}

struct GpuSkyLight {
    direction_vs: vec3<f32>,
    intensity: f32,
    color: vec3<f32>,
    _pad: f32,
}

// Scalar padding: a trailing vec3 would be 16-aligned and break the 64-byte Rust layout.
struct GpuSpotLight {
    position_vs: vec3<f32>,
    radius: f32,
    direction_vs: vec3<f32>,
    inner_cos: f32,
    color: vec3<f32>,
    outer_cos: f32,
    intensity: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

// right_vs / up_vs are half-extent vectors of the rectangle.
struct GpuAreaLight {
    position_vs: vec3<f32>,
    intensity: f32,
    right_vs: vec3<f32>,
    _pad0: f32,
    up_vs: vec3<f32>,
    _pad1: f32,
    color: vec3<f32>,
    _pad2: f32,
}

struct LightCounts {
    num_point: u32,
    num_spot: u32,
    num_area: u32,
    num_sky: u32,
}

@group(1) @binding(0) var<uniform>         cp:            ClusterParams;
@group(1) @binding(1) var<storage, read>   point_lights:  array<GpuPointLight>;
@group(1) @binding(2) var<storage, read>   light_grid:    array<u32>;
@group(1) @binding(3) var<storage, read>   light_indices: array<u32>;
@group(1) @binding(4) var<storage, read>   sky_lights:    array<GpuSkyLight>;
@group(1) @binding(5) var<storage, read>   spot_lights:   array<GpuSpotLight>;
@group(1) @binding(6) var<storage, read>   area_lights:   array<GpuAreaLight>;
@group(1) @binding(7) var<uniform>         light_counts:  LightCounts;

// Shadow
struct ShadowParams {
    view_to_shadow: array<mat4x4<f32>, 4>,
    cascade_splits: vec4<f32>,
    bias: f32,
    inv_shadow_map_size: f32,
    _pad: vec2<f32>,
}

@group(2) @binding(0) var<uniform> sp:       ShadowParams;
@group(2) @binding(1) var          t_shadow: texture_depth_2d_array;
@group(2) @binding(2) var          s_shadow: sampler_comparison;

fn select_cascade(depth_vs: f32) -> u32 {
    if depth_vs < sp.cascade_splits.x { return 0u; }
    if depth_vs < sp.cascade_splits.y { return 1u; }
    if depth_vs < sp.cascade_splits.z { return 2u; }
    return 3u;
}

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

// Background radiance, scaled by each sky light's illuminance so it stays in proportion to the lit scene.
fn sky_color(ray_dir: vec3<f32>) -> vec3<f32> {
    let horizon = vec3f(0.55, 0.65, 0.80);
    let zenith  = vec3f(0.10, 0.25, 0.60);
    let gradient = mix(horizon, zenith, clamp(ray_dir.y, 0.0, 1.0));
    var col = vec3f(0.0);
    for (var i = 0u; i < cp.num_sky_lights; i++) {
        let L_ws = normalize((camera.inv_view * vec4f(-sky_lights[i].direction_vs, 0.0)).xyz);
        if L_ws.y < 0.0 { continue; }
        let illuminance = sky_lights[i].color * sky_lights[i].intensity / PI;
        col += illuminance * (gradient + pow(max(dot(ray_dir, L_ws), 0.0), 128.0));
    }
    return col;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let frag_xy = vec2<i32>(i32(in.pos.x), i32(in.pos.y));

    let ndc_x = in.pos.x / cp.screen_w * 2.0 - 1.0;
    let ndc_y = 1.0 - in.pos.y / cp.screen_h * 2.0;

    let depth_v  = textureLoad(t_depth, frag_xy, 0);

    // Reversed-Z: depth_v == 0 means "at far plane / sky" — no geometry.
    if depth_v == 0.0 {
        if cp.num_sky_lights == 0u {
            return vec4f(0.0, 0.0, 0.0, 1.0);
        }
        let far_ws4  = camera.inv_view_proj * vec4f(ndc_x, ndc_y, 0.0, 1.0);
        let near_ws4 = camera.inv_view_proj * vec4f(ndc_x, ndc_y, 1.0, 1.0);
        let ray_dir  = normalize(far_ws4.xyz / far_ws4.w - near_ws4.xyz / near_ws4.w);
        return vec4f(sky_color(ray_dir) * cp.exposure, 1.0);
    }

    let albedo_raw = textureSample(t_albedo, s_linear, in.uv).rgb;
    let albedo     = select(albedo_raw, vec3f(1.0), debug.disable_albedo != 0u);
    let norm_enc   = textureSample(t_normal, s_linear, in.uv).rgb;

    // Reconstruct view-space depth from reversed-Z depth buffer value.
    // z_view_pos = z_near * z_far / (depth * (z_far - z_near) + z_near)
    let depth_vs = cp.z_near * cp.z_far / (depth_v * (cp.z_far - cp.z_near) + cp.z_near);
    let pos_vs = vec3<f32>(
        ndc_x * depth_vs * cp.inv_proj_00,
        ndc_y * depth_vs * cp.inv_proj_11,
        -depth_vs,
    );

    // Decode world-space normal (stored as (N + 1) / 2 in GBuffer)
    let N = normalize(norm_enc * 2.0 - vec3<f32>(1.0));
    // Local lights are in view space. inv_view is rigid, so N * M == transpose(M) * N maps world -> view.
    let N_vs = normalize(N * mat3x3<f32>(camera.inv_view[0].xyz, camera.inv_view[1].xyz, camera.inv_view[2].xyz));

    // Reconstruct world-space position from NDC + raw depth
    let pos_ws4 = camera.inv_view_proj * vec4f(ndc_x, ndc_y, depth_v, 1.0);
    let pos_ws  = pos_ws4.xyz / pos_ws4.w;

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

    let cam_pos  = camera.inv_view[3].xyz;
    let view_dir = normalize(cam_pos - pos_ws);
    let indirect = sample_probe_irradiance(pos_ws, N, view_dir);
    var color: vec3f;
    if debug.disable_indirect != 0u {
        color = vec3f(0.0);
    } else {
        color = albedo * indirect;
    }

    if debug.disable_direct == 0u {
        for (var i = 0u; i < light_count; i++) {
            let li    = light_indices[cluster_idx * MAX_LIGHTS_PER_CLUSTER + i];
            let light = point_lights[li];

            let to_light = light.position_vs - pos_vs;
            let dist     = length(to_light);
            if dist >= light.radius {
                continue;
            }

            let L    = to_light / dist;
            let diff = max(dot(N_vs, L), 0.0);
            color += albedo / PI * light.color * (light.intensity * diff * range_attenuation(dist, light.radius));
        }

        for (var i = 0u; i < light_counts.num_spot; i++) {
            let light = spot_lights[i];

            let to_light = light.position_vs - pos_vs;
            let dist     = length(to_light);
            if dist >= light.radius {
                continue;
            }

            let L    = to_light / dist;
            let cd   = dot(-L, normalize(light.direction_vs));
            let t    = clamp((cd - light.outer_cos) / max(light.inner_cos - light.outer_cos, 1e-4), 0.0, 1.0);
            let diff = max(dot(N_vs, L), 0.0);
            color += albedo / PI * light.color * (light.intensity * diff * t * t * range_attenuation(dist, light.radius));
        }

        // Representative-point approximation: light from the closest point on the rectangle,
        // scaled by its area so the falloff stays bounded as the surface approaches the light.
        for (var i = 0u; i < light_counts.num_area; i++) {
            let light = area_lights[i];

            let half_r = length(light.right_vs);
            let half_u = length(light.up_vs);
            if half_r <= 0.0 || half_u <= 0.0 {
                continue;
            }
            let r_dir = light.right_vs / half_r;
            let u_dir = light.up_vs / half_u;

            let rel     = pos_vs - light.position_vs;
            let closest = light.position_vs
                + r_dir * clamp(dot(rel, r_dir), -half_r, half_r)
                + u_dir * clamp(dot(rel, u_dir), -half_u, half_u);

            let to_light = closest - pos_vs;
            let dist2    = max(dot(to_light, to_light), 1e-4);
            let L        = to_light * inverseSqrt(dist2);

            let area    = 4.0 * half_r * half_u;
            let facing  = abs(dot(L, normalize(cross(r_dir, u_dir))));
            let diff    = max(dot(N_vs, L), 0.0);
            let falloff = area / (dist2 + area);
            color += albedo / PI * light.color * (light.intensity * diff * facing * falloff);
        }

        for (var i = 0u; i < cp.num_sky_lights; i++) {
            let sky = sky_lights[i];
            let L = normalize((camera.inv_view * vec4f(-sky.direction_vs, 0.0)).xyz);
            if L.y < 0.0 { continue; }
            let diff = max(dot(N, L), 0.0);

            // Select cascade based on view-space depth and project into shadow space.
            let cascade     = select_cascade(depth_vs);
            let shadow_clip = sp.view_to_shadow[cascade] * vec4<f32>(pos_vs, 1.0);
            let shadow_ndc  = shadow_clip.xyz / shadow_clip.w;
            let shadow_uv   = shadow_ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
            let ref_depth   = shadow_ndc.z - sp.bias;

            var lit = 1.0;
            if debug.disable_shadows == 0u &&
                shadow_uv.x >= 0.0 && shadow_uv.x <= 1.0 &&
                shadow_uv.y >= 0.0 && shadow_uv.y <= 1.0 &&
                ref_depth >= 0.0 && ref_depth <= 1.0 {
                var shadow_sum = 0.0;
                for (var sy: i32 = -1; sy <= 1; sy += 1) {
                    for (var sx: i32 = -1; sx <= 1; sx += 1) {
                        let off = vec2<f32>(f32(sx), f32(sy)) * sp.inv_shadow_map_size;
                        shadow_sum += textureSampleCompare(
                            t_shadow, s_shadow, shadow_uv + off, i32(cascade), ref_depth,
                        );
                    }
                }
                lit = shadow_sum / 9.0;
            }

            color += albedo / PI * sky.color * (sky.intensity * diff * lit);
        }
    }

    return vec4<f32>(color * cp.exposure, 1.0);
}
