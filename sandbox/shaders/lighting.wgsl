// GBuffer inputs
@group(0) @binding(0) var t_albedo:   texture_2d<f32>;
@group(0) @binding(1) var t_normal:   texture_2d<f32>;
@group(0) @binding(2) var t_material: texture_2d<f32>;
@group(0) @binding(3) var t_depth:    texture_depth_2d;
@group(0) @binding(4) var s_linear:   sampler;
// Emitted radiance (texture × factor × strength), not yet exposed.
@group(0) @binding(5) var t_emissive: texture_2d<f32>;

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
    // near, far
    depth_params:  vec4<f32>,
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
    // 0 lit, 1 albedo, 2 normal, 3 material, 4 emissive, 5 local shadows (rendering::debug_flags::GBufferView).
    view:             u32,
    disable_specular: u32,
    disable_reflections: u32,
    disable_ssr:      u32,
}

@group(4) @binding(0) var<uniform> debug: DebugFlags;

// Mirrors rendering::light::ShadowedLights. Entry = kind << 16 | index.
// lights[slot]: shadow origin in world space (xyz), light half size (w, 0 for point/spot).
struct ShadowedLights {
    entries: vec4<u32>,
    count:   u32,
    lights:  array<vec4<f32>, 4>,
}

@group(4) @binding(1) var<uniform> shadowed: ShadowedLights;

const SHADOW_KIND_POINT: u32 = 0u;
const SHADOW_KIND_SPOT:  u32 = 1u;
const SHADOW_KIND_AREA:  u32 = 2u;

// 12-tap Poisson disc on the unit disc, for cube shadow PCF.
const POISSON: array<vec2<f32>, 12> = array<vec2<f32>, 12>(
    vec2f(-0.326, -0.406), vec2f(-0.840, -0.074), vec2f(-0.696,  0.457),
    vec2f(-0.203,  0.621), vec2f( 0.962, -0.195), vec2f( 0.473, -0.480),
    vec2f( 0.519,  0.767), vec2f( 0.185, -0.893), vec2f( 0.507,  0.064),
    vec2f( 0.896,  0.412), vec2f(-0.322, -0.933), vec2f(-0.792, -0.598),
);

// Visibility of shadowed light `slot` at world position `pos` (normal `N`), from its cube map.
// The face is the dominant axis of the light-to-point direction, projected with that face's own
// view-projection so the lookup matches what was rasterized. Area lights widen the PCF by their
// size over the distance, a cheap stand-in for a penumbra.
fn cube_shadow(slot: u32, pos: vec3<f32>, N: vec3<f32>, pixel: vec2<f32>) -> vec3<f32> {
    let light = shadowed.lights[slot];
    let d = pos - light.xyz;
    let dist = length(d);
    let a = abs(d);
    var face: u32;
    if a.x >= a.y && a.x >= a.z {
        face = select(1u, 0u, d.x > 0.0);
    } else if a.y >= a.z {
        face = select(3u, 2u, d.y > 0.0);
    } else {
        face = select(5u, 4u, d.z > 0.0);
    }
    let layer = slot * 6u + face;

    // A 90° face spans 2·dist across at this distance: offset by ~1.5 texels along the normal.
    let texel = 2.0 * dist * local_shadow.inv_size;
    let clip = local_shadow.face_vp[layer] * vec4f(pos + N * (1.5 * texel), 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = ndc.xy * vec2f(0.5, -0.5) + 0.5;

    let radius_texels = select(1.5, clamp(light.w / max(dist, 1e-3) / local_shadow.inv_size * 0.5, 1.5, 8.0), light.w > 0.0);
    let radius = radius_texels * local_shadow.inv_size;
    let angle = fract(52.9829189 * fract(dot(pixel, vec2f(0.06711056, 0.00583715)))) * 6.28318531;
    let rot = mat2x2<f32>(cos(angle), sin(angle), -sin(angle), cos(angle));
    var lit = 0.0;
    for (var i = 0u; i < 12u; i++) {
        let tap = clamp(uv + rot * POISSON[i] * radius, vec2f(0.0), vec2f(1.0));
        lit += textureSampleCompareLevel(t_local_shadow, s_shadow, tap, i32(layer), ndc.z);
    }
    if lit == 0.0 {
        return vec3f(0.0);
    }
    // One glass-tint texel of slack, so glass doesn't tint itself.
    let tint_texel = 2.0 * dist / f32(textureDimensions(t_local_glass_tint).x);
    return lit / 12.0 * glass_tint(t_local_glass_tint, uv, layer, clip.w - tint_texel);
}

// Visibility of a local light at this pixel, tinted by glass: its cube shadow if it is one of
// the shadowed lights, else 1.
fn local_shadow_of(kind: u32, index: u32, pos: vec3<f32>, N: vec3<f32>, pixel: vec2<f32>) -> vec3<f32> {
    if debug.disable_shadows != 0u { return vec3f(1.0); }
    let entry = kind << 16u | index;
    for (var slot = 0u; slot < min(shadowed.count, 4u); slot++) {
        if shadowed.entries[slot] == entry { return cube_shadow(slot, pos, N, pixel); }
    }
    return vec3f(1.0);
}

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

// `dir` is the direction the irradiance is looked up in: the normal for diffuse, or near the
// reflection vector as a stand-in for glossy reflections. Bias and visibility always follow `N_ws`.
fn sample_probe_irradiance(pos_ws: vec3<f32>, N_ws: vec3<f32>, dir: vec3<f32>, view_dir: vec3<f32>) -> vec3<f32> {
    let biased_pos = pos_ws
        + N_ws    * probe_params.normal_bias
        + view_dir * probe_params.view_bias;
    let grid_pos = (biased_pos - probe_params.grid_origin) / probe_params.probe_spacing;
    let base_f   = floor(grid_pos);
    let base_i   = vec3<i32>(base_f);
    let frac     = grid_pos - base_f;
    let dim      = vec3<i32>(probe_params.grid_dim);
    let oct_uv   = octahedral_encode(N_ws);
    let dir_uv   = octahedral_encode(dir);

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

                let irr_uv = irradiance_atlas_uv(probe_index, dir_uv);
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
// Keeps the GGX peak finite so point lights don't turn into single-pixel fireflies.
const MIN_ROUGHNESS: f32 = 0.045;

struct Surface {
    N: vec3<f32>,
    V: vec3<f32>,
    diffuse_color: vec3<f32>,
    f0: vec3<f32>,
    // GGX alpha (perceptual roughness squared).
    alpha: f32,
}

fn fresnel_schlick(f0: vec3<f32>, v_dot_h: f32) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(1.0 - v_dot_h, 5.0);
}

fn d_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// Height-correlated Smith, with the 1 / (4 N.L N.V) folded in.
fn v_smith_ggx(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let gv = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    let gl = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-5);
}

// Cook-Torrance specular times N.L, for light arriving from `L`.
fn specular_brdf(s: Surface, L: vec3<f32>, alpha: f32) -> vec3<f32> {
    let n_dot_l = max(dot(s.N, L), 0.0);
    if n_dot_l <= 0.0 { return vec3f(0.0); }
    let H = normalize(s.V + L);
    let n_dot_v = max(dot(s.N, s.V), 1e-4);
    let F = fresnel_schlick(s.f0, max(dot(s.V, H), 0.0));
    let spec = d_ggx(max(dot(s.N, H), 0.0), alpha) * v_smith_ggx(n_dot_v, n_dot_l, alpha) * F;
    return select(spec, vec3f(0.0), debug.disable_specular != 0u) * n_dot_l;
}

// Lambert, minus the energy Fresnel sent to specular, times N.L.
fn diffuse_brdf(s: Surface, L: vec3<f32>) -> vec3<f32> {
    let n_dot_l = max(dot(s.N, L), 0.0);
    let H = normalize(s.V + L);
    let F = select(fresnel_schlick(s.f0, max(dot(s.V, H), 0.0)), vec3f(0.0), debug.disable_specular != 0u);
    return (1.0 - F) * s.diffuse_color / PI * n_dot_l;
}

fn brdf(s: Surface, L: vec3<f32>) -> vec3<f32> {
    return diffuse_brdf(s, L) + specular_brdf(s, L, s.alpha);
}

// Karis' analytic fit of the split-sum environment BRDF (scale, bias applied to F0).
fn env_brdf_approx(f0: vec3<f32>, roughness: f32, n_dot_v: f32) -> vec3<f32> {
    let c0 = vec4f(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4f(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    let ab = vec2f(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}

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

// right_vs / up_vs are half-extent vectors of the rectangle. One-sided lights emit toward right × up.
struct GpuAreaLight {
    position_vs: vec3<f32>,
    intensity: f32,
    right_vs: vec3<f32>,
    two_sided: f32,
    up_vs: vec3<f32>,
    _pad1: f32,
    color: vec3<f32>,
    _pad2: f32,
}

// Cosine at an area light for light leaving along `dir`. One-sided lights are dark behind.
fn area_facing(light: GpuAreaLight, dir: vec3<f32>, normal: vec3<f32>) -> f32 {
    let c = dot(dir, normal);
    return select(max(c, 0.0), abs(c), light.two_sided > 0.5);
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
// One cube (6 layers, +X −X +Y −Y +Z −Z) per shadowed local light.
@group(2) @binding(3) var          t_local_shadow: texture_depth_2d_array;

// Mirrors rendering::pass::shadow::LocalShadowParams.
struct LocalShadowParams {
    face_vp:  array<mat4x4<f32>, 24>,
    inv_size: f32,
}

@group(2) @binding(4) var<uniform> local_shadow: LocalShadowParams;
// Glass tint maps, one layer per cascade / cube face (rendering::pass::shadow): rgb = transmittance
// of all glass along the light ray, a = depth of the glass nearest the light (clip z for cascades,
// clip w for cube faces).
@group(2) @binding(5) var          t_glass_tint: texture_2d_array<f32>;
@group(2) @binding(6) var          t_local_glass_tint: texture_2d_array<f32>;

// Transmittance of the glass in front of a receiver at `depth` (same units as the map's alpha).
// Each of the 4 bilinear taps is depth-tested before filtering, so glass behind the receiver never
// tints it and edges stay soft.
fn glass_tint(t: texture_2d_array<f32>, uv: vec2<f32>, layer: u32, depth: f32) -> vec3<f32> {
    let size = vec2<i32>(textureDimensions(t));
    let p = uv * vec2f(size) - 0.5;
    let base = floor(p);
    let f = p - base;
    var tint = vec3f(0.0);
    for (var i = 0u; i < 4u; i++) {
        let o = vec2f(f32(i & 1u), f32(i >> 1u));
        let texel = clamp(vec2<i32>(base + o), vec2<i32>(0), size - 1);
        let s = textureLoad(t, texel, i32(layer), 0);
        let w = mix(1.0 - f.x, f.x, o.x) * mix(1.0 - f.y, f.y, o.y);
        tint += select(vec3f(1.0), s.rgb, s.a < depth) * w;
    }
    return tint;
}

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

fn cluster_index(pixel: vec2<f32>, depth_vs: f32) -> u32 {
    let cx = u32(pixel.x / cp.tile_w);
    let cy = u32(pixel.y / cp.tile_h);
    let cz = min(u32(log(depth_vs / cp.z_near) * cp.log_ratio_recip), CLUSTER_Z - 1u);
    return cx + CLUSTER_X * (cy + CLUSTER_Y * cz);
}

// Direct light from every light, with shadows. `surf_vs` / `surf_ws` are the same surface in view
// and world space, and `pixel` is the fragment position (cluster lookup, PCF rotation).
fn direct_lighting(surf_vs: Surface, surf_ws: Surface, pos_vs: vec3<f32>, pos_ws: vec3<f32>, pixel: vec2<f32>, depth_vs: f32) -> vec3<f32> {
    let N = surf_ws.N;
    let N_vs = surf_vs.N;
    let alpha = surf_vs.alpha;
    let cluster_idx = cluster_index(pixel, depth_vs);
    let light_count = light_grid[cluster_idx];

    var color = vec3f(0.0);
    for (var i = 0u; i < light_count; i++) {
        let li    = light_indices[cluster_idx * MAX_LIGHTS_PER_CLUSTER + i];
        let light = point_lights[li];

        let to_light = light.position_vs - pos_vs;
        let dist     = length(to_light);
        if dist >= light.radius {
            continue;
        }

        let L = to_light / dist;
        let vis = local_shadow_of(SHADOW_KIND_POINT, li, pos_ws, N, pixel);
        color += brdf(surf_vs, L) * light.color * vis * (light.intensity * range_attenuation(dist, light.radius));
    }

    for (var i = 0u; i < light_counts.num_spot; i++) {
        let light = spot_lights[i];

        let to_light = light.position_vs - pos_vs;
        let dist     = length(to_light);
        if dist >= light.radius {
            continue;
        }

        let L  = to_light / dist;
        let cd = dot(-L, normalize(light.direction_vs));
        let t  = clamp((cd - light.outer_cos) / max(light.inner_cos - light.outer_cos, 1e-4), 0.0, 1.0);
        let vis = local_shadow_of(SHADOW_KIND_SPOT, i, pos_ws, N, pixel);
        color += brdf(surf_vs, L) * light.color * vis * (light.intensity * t * t * range_attenuation(dist, light.radius));
    }

    // Representative-point approximation. Diffuse takes light from the closest point on the
    // rectangle, specular from where the reflection ray meets it, both scaled by the area so
    // the falloff stays bounded as the surface approaches the light.
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

        let area     = 4.0 * half_r * half_u;
        let normal   = normalize(cross(r_dir, u_dir));
        let radiance = light.color * light.intensity * local_shadow_of(SHADOW_KIND_AREA, i, pos_ws, N, pixel);
        let facing   = area_facing(light, -L, normal);
        color += diffuse_brdf(surf_vs, L) * radiance * (facing * area / (dist2 + area));

        // A ray parallel to or facing away from the plane falls back to the closest point.
        let R     = reflect(-surf_vs.V, N_vs);
        let r_dot = dot(R, normal);
        let t_hit = dot(light.position_vs - pos_vs, normal) / select(r_dot, 1.0, abs(r_dot) < 1e-4);
        var spec_point = closest;
        if abs(r_dot) >= 1e-4 && t_hit > 0.0 {
            let hit = pos_vs + R * t_hit - light.position_vs;
            spec_point = light.position_vs
                + r_dir * clamp(dot(hit, r_dir), -half_r, half_r)
                + u_dir * clamp(dot(hit, u_dir), -half_u, half_u);
        }
        let to_spec     = spec_point - pos_vs;
        let spec_dist2  = max(dot(to_spec, to_spec), 1e-4);
        let L_spec      = to_spec * inverseSqrt(spec_dist2);
        // Karis' sphere-light widening, using the radius of a disc with the rectangle's area.
        let source_r    = sqrt(area / PI);
        let spec_alpha  = clamp(alpha + source_r / (2.0 * sqrt(spec_dist2)), 0.0, 1.0);
        let spec_facing = area_facing(light, -L_spec, normal);
        color += specular_brdf(surf_vs, L_spec, spec_alpha) * radiance
            * (spec_facing * area / (spec_dist2 + area));
    }

    for (var i = 0u; i < cp.num_sky_lights; i++) {
        let sky = sky_lights[i];
        let L = normalize((camera.inv_view * vec4f(-sky.direction_vs, 0.0)).xyz);
        if L.y < 0.0 { continue; }

        // Select cascade based on view-space depth and project into shadow space.
        let cascade     = select_cascade(depth_vs);
        let shadow_clip = sp.view_to_shadow[cascade] * vec4<f32>(pos_vs, 1.0);
        let shadow_ndc  = shadow_clip.xyz / shadow_clip.w;
        let shadow_uv   = shadow_ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
        let ref_depth   = shadow_ndc.z - sp.bias;

        var lit = vec3f(1.0);
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
            lit = shadow_sum / 9.0 * glass_tint(t_glass_tint, shadow_uv, cascade, ref_depth);
        }

        color += brdf(surf_ws, L) * sky.color * (sky.intensity * lit);
    }

    return color;
}

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

struct LightingOutput {
    // Diffuse + direct specular.
    @location(0) lit: vec4<f32>,
    // rgb: env BRDF weight for reflections (0 = no reflection), a: roughness.
    @location(1) specular_weight: vec4<f32>,
    // Probe-based reflection × weight, exposed. The composite pass uses it where SSR misses.
    @location(2) specular_fallback: vec4<f32>,
}

fn unreflective(color: vec3<f32>) -> LightingOutput {
    return LightingOutput(vec4f(color, 1.0), vec4f(0.0), vec4f(0.0));
}

@fragment
fn fs_main(in: VertexOutput) -> LightingOutput {
    let frag_xy = vec2<i32>(i32(in.pos.x), i32(in.pos.y));

    let ndc_x = in.pos.x / cp.screen_w * 2.0 - 1.0;
    let ndc_y = 1.0 - in.pos.y / cp.screen_h * 2.0;

    let depth_v  = textureLoad(t_depth, frag_xy, 0);

    // Reversed-Z: depth_v == 0 means "at far plane / sky" — no geometry.
    if depth_v == 0.0 {
        if cp.num_sky_lights == 0u {
            return unreflective(vec3f(0.0));
        }
        let far_ws4  = camera.inv_view_proj * vec4f(ndc_x, ndc_y, 0.0, 1.0);
        let near_ws4 = camera.inv_view_proj * vec4f(ndc_x, ndc_y, 1.0, 1.0);
        let ray_dir  = normalize(far_ws4.xyz / far_ws4.w - near_ws4.xyz / near_ws4.w);
        return unreflective(sky_color(ray_dir) * cp.exposure);
    }

    switch debug.view {
        case 1u: { return unreflective(textureSample(t_albedo, s_linear, in.uv).rgb); }
        case 2u: { return unreflective(textureSample(t_normal, s_linear, in.uv).rgb); }
        case 3u: { return unreflective(textureSample(t_material, s_linear, in.uv).rgb); }
        case 4u: { return unreflective(textureSample(t_emissive, s_linear, in.uv).rgb * cp.exposure); }
        default: {}
    }

    let albedo_raw = textureSample(t_albedo, s_linear, in.uv).rgb;
    let albedo     = select(albedo_raw, vec3f(1.0), debug.disable_albedo != 0u);
    let norm_enc   = textureSample(t_normal, s_linear, in.uv).rgb;
    let material   = textureSample(t_material, s_linear, in.uv);
    let roughness  = max(material.r, MIN_ROUGHNESS);
    let metallic   = material.g;
    let emissive   = textureSample(t_emissive, s_linear, in.uv).rgb;

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

    if debug.view == 5u {
        var vis = vec3f(1.0);
        for (var slot = 0u; slot < min(shadowed.count, 3u); slot++) {
            vis[slot] = dot(cube_shadow(slot, pos_ws, N, in.pos.xy), vec3f(1.0 / 3.0));
        }
        return unreflective(vis);
    }

    // Debug: visualise light density per cluster.
    if cp.debug_mode == 1u {
        let heat = f32(light_grid[cluster_index(in.pos.xy, depth_vs)]) / f32(MAX_LIGHTS_PER_CLUSTER);
        return unreflective(vec3f(heat, 0.3, 1.0 - heat));
    }

    let cam_pos  = camera.inv_view[3].xyz;
    let view_dir = normalize(cam_pos - pos_ws);

    let diffuse_color = albedo * (1.0 - metallic);
    let f0 = mix(vec3f(0.04), albedo, metallic);
    let alpha = roughness * roughness;
    // Local lights are shaded in view space, sky lights and probes in world space.
    let surf_vs = Surface(N_vs, normalize(-pos_vs), diffuse_color, f0, alpha);
    let surf_ws = Surface(N, view_dir, diffuse_color, f0, alpha);

    var color = vec3f(0.0);
    var out: LightingOutput;
    if debug.disable_indirect == 0u {
        color = diffuse_color * sample_probe_irradiance(pos_ws, N, N, view_dir);

        // Reflections are added by the composite pass: SSR where it hits, else probe irradiance
        // around the reflection vector, leaning toward N as roughness grows (Filament's dominant
        // direction).
        if debug.disable_specular == 0u && debug.disable_reflections == 0u {
            let n_dot_v = max(dot(N, view_dir), 1e-4);
            let R = normalize(mix(reflect(-view_dir, N), N, alpha));
            let weight = env_brdf_approx(f0, roughness, n_dot_v);
            let env = sample_probe_irradiance(pos_ws, N, R, view_dir);
            out.specular_weight = vec4f(weight, roughness);
            out.specular_fallback = vec4f(env * weight * cp.exposure, 1.0);
        }
    }

    if debug.disable_direct == 0u {
        color += direct_lighting(surf_vs, surf_ws, pos_vs, pos_ws, in.pos.xy, depth_vs);
    }

    color += emissive;
    out.lit = vec4<f32>(color * cp.exposure, 1.0);
    return out;
}

// ---------------------------------------------------------------------------------------------
// Forward shading of `Blend` meshes (rendering::pass::transparent). Shares groups 1–4 with the
// deferred path above; group 0 is replaced by the bindings below and group 5 holds the material.

// Mirrors rendering::arena::GpuModel.
struct Model {
    model: mat4x4<f32>,
    normal: mat4x4<f32>,
    // x: alpha cutoff, y: 1 if double-sided, z: metallic, w: roughness.
    material: vec4<f32>,
    base_color: vec4<f32>,
    // rgb: emissive factor × strength.
    emissive: vec4<f32>,
    // x: transmission, y: IOR, z: volume thickness (mesh-local), w: attenuation distance (0 = none).
    transmission: vec4<f32>,
    // rgb: volume attenuation color.
    attenuation: vec4<f32>,
}

// The composited opaque scene (exposed HDR).
@group(0) @binding(6) var t_scene: texture_2d<f32>;
@group(0) @binding(7) var<storage, read> models: array<Model>;
// `t_scene` with a blurred mip chain, sampled through `s_linear` (clamped) for refraction and
// rough transmission.
@group(0) @binding(8) var t_behind: texture_2d<f32>;
// Source level of the `t_behind` mip chain build.
@group(0) @binding(10) var t_blit: texture_2d<f32>;

@group(5) @binding(0) var m_albedo: texture_2d<f32>;
@group(5) @binding(1) var m_normal: texture_2d<f32>;
// glTF layout: G = roughness, B = metallic.
@group(5) @binding(2) var m_metallic_roughness: texture_2d<f32>;
@group(5) @binding(3) var m_emissive: texture_2d<f32>;
@group(5) @binding(4) var m_sampler: sampler;

// Copies the opaque scene into the transparent pass's target before blending onto it.
@fragment
fn fs_background(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureLoad(t_scene, vec2<i32>(in.pos.xy), 0);
}

@fragment
fn fs_scene_copy(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureLoad(t_blit, vec2<i32>(in.pos.xy), 0);
}

// Four bilinear taps one source texel off the destination center: a 4×4 tent over the source,
// smoother than a 2×2 box when the coarse levels are magnified.
@fragment
fn fs_scene_downsample(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2f(textureDimensions(t_blit));
    let a = textureSampleLevel(t_blit, s_linear, in.uv + vec2f(-texel.x, -texel.y), 0.0);
    let b = textureSampleLevel(t_blit, s_linear, in.uv + vec2f( texel.x, -texel.y), 0.0);
    let c = textureSampleLevel(t_blit, s_linear, in.uv + vec2f(-texel.x,  texel.y), 0.0);
    let d = textureSampleLevel(t_blit, s_linear, in.uv + vec2f( texel.x,  texel.y), 0.0);
    return (a + b + c + d) * 0.25;
}

// Where the scene behind a transmissive surface is seen, as a `t_behind` UV. Thin-walled surfaces
// (thickness 0) see straight through; volumes follow the refracted view ray `thickness` deep and
// look up the opaque scene where it leaves. Falls back to the straight-through UV near the screen
// edge and where the refracted point lands on something in front of the surface.
fn refracted_uv(m: Model, frag: vec4<f32>, pos_ws: vec3<f32>, N: vec3<f32>, view_dir: vec3<f32>, thickness: f32) -> vec2<f32> {
    let screen = vec2f(textureDimensions(t_behind));
    let straight = frag.xy / screen;
    if thickness <= 0.0 {
        return straight;
    }
    let ray = refract(-view_dir, N, 1.0 / m.transmission.y);
    if dot(ray, ray) == 0.0 {
        return straight;
    }
    let exit = camera.view_proj * vec4f(pos_ws + normalize(ray) * thickness, 1.0);
    if exit.w <= 0.0 {
        return straight;
    }
    let bent = (exit.xy / exit.w) * vec2f(0.5, -0.5) + 0.5;
    let edge = min(min(bent.x, bent.y), min(1.0 - bent.x, 1.0 - bent.y));
    let uv = mix(straight, bent, smoothstep(0.0, 0.05, edge));

    let pixel = clamp(vec2<i32>(uv * screen), vec2<i32>(0), vec2<i32>(screen) - 1);
    // Reversed-Z: a larger depth is nearer the camera.
    if textureLoad(t_depth, pixel, 0) > frag.z {
        return straight;
    }
    return uv;
}

// Screen-space reflections for blended surfaces, ported from ssr.wgsl with fewer steps: blended
// surfaces aren't in the G-buffer, so the opaque `ssr` pass can't cover them.
const GLASS_SSR_MAX_ROUGHNESS:  f32 = 0.4;
const GLASS_SSR_FADE_ROUGHNESS: f32 = 0.25;
const GLASS_SSR_MAX_DISTANCE:   f32 = 50.0;
const GLASS_SSR_MAX_STEPS:      u32 = 32u;
const GLASS_SSR_STEP_PIXELS:    f32 = 3.0;
const GLASS_SSR_REFINE_STEPS:   u32 = 5u;
const GLASS_SSR_THICKNESS:      f32 = 0.03;

fn linear_depth(d: f32) -> f32 {
    let n = camera.depth_params.x;
    let f = camera.depth_params.y;
    return n * f / (d * (f - n) + n);
}

fn clip_to_pixel(h: vec4<f32>, dims: vec2<f32>) -> vec2<f32> {
    let ndc = h.xy / h.w;
    return vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * dims;
}

// Marches the reflection ray `R` from `pos_ws` against the G-buffer depth. Returns the opaque
// scene where it lands (exposed, from `t_behind` blurred by roughness) and the confidence in a.
fn glass_ssr(pos_ws: vec3<f32>, R: vec3<f32>, frag: vec2<f32>, roughness: f32) -> vec4<f32> {
    let dims = vec2f(textureDimensions(t_depth));
    let cam_pos = camera.inv_view[3].xyz;
    let to_view = mat3x3<f32>(camera.inv_view[0].xyz, camera.inv_view[1].xyz, camera.inv_view[2].xyz);
    let R_vs = R * to_view;
    let pos_vs = (pos_ws - cam_pos) * to_view;

    var dist = GLASS_SSR_MAX_DISTANCE;
    if R_vs.z > 0.0 {
        dist = min(dist, (-1.01 * camera.depth_params.x - pos_vs.z) / R_vs.z);
    }
    if dist <= 0.0 {
        return vec4f(0.0);
    }

    let h0 = camera.view_proj * vec4f(pos_ws, 1.0);
    let h1 = camera.view_proj * vec4f(pos_ws + R * dist, 1.0);
    let k0 = 1.0 / h0.w;
    let k1 = 1.0 / h1.w;
    let p0 = clip_to_pixel(h0, dims);
    let delta = clip_to_pixel(h1, dims) - p0;
    let steps = u32(clamp(length(delta) / GLASS_SSR_STEP_PIXELS, 1.0, f32(GLASS_SSR_MAX_STEPS)));
    let jitter = fract(52.9829189 * fract(dot(frag, vec2f(0.06711056, 0.00583715))));

    var lo = 0.0;
    var hi = -1.0;
    for (var i = 1u; i <= steps; i++) {
        let s = (f32(i) - jitter) / f32(steps);
        let p = p0 + delta * s;
        if any(p < vec2f(0.0)) || any(p >= dims) {
            break;
        }
        let scene_d = textureLoad(t_depth, vec2<i32>(p), 0);
        if scene_d > 0.0 && 1.0 / mix(k0, k1, s) > linear_depth(scene_d) {
            hi = s;
            break;
        }
        lo = s;
    }
    if hi < 0.0 {
        return vec4f(0.0);
    }

    // Unlike ssr.wgsl, the thickness test runs after refining the first crossing: with few steps a
    // step can jump past a surface's whole thickness window. A crossing that is still far behind
    // after refining went behind a depth discontinuity, not into a surface.
    for (var i = 0u; i < GLASS_SSR_REFINE_STEPS; i++) {
        let mid = 0.5 * (lo + hi);
        let scene_d = textureLoad(t_depth, vec2<i32>(p0 + delta * mid), 0);
        if scene_d > 0.0 && 1.0 / mix(k0, k1, mid) > linear_depth(scene_d) {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    let hit_px = p0 + delta * hi;
    let hit_w = linear_depth(textureLoad(t_depth, vec2<i32>(hit_px), 0));
    if 1.0 / mix(k0, k1, hi) - hit_w > GLASS_SSR_THICKNESS * hit_w {
        return vec4f(0.0);
    }
    let hit_N = normalize(textureLoad(t_normal, vec2<i32>(hit_px), 0).rgb * 2.0 - 1.0);
    if dot(hit_N, R) > 0.0 {
        return vec4f(0.0);
    }
    let hit_uv = hit_px / dims;
    // The mip whose texel matches composite.wgsl's opaque SSR blur radius (roughness × 32 px).
    let lod = log2(max(roughness * 32.0, 1.0));
    let color = textureSampleLevel(t_behind, s_linear, hit_uv, lod).rgb;

    let edge = min(hit_uv, 1.0 - hit_uv);
    let confidence = smoothstep(0.0, 0.08, min(edge.x, edge.y))
        * (1.0 - smoothstep(GLASS_SSR_FADE_ROUGHNESS, GLASS_SSR_MAX_ROUGHNESS, roughness))
        * (1.0 - smoothstep(0.1, 0.5, R_vs.z))
        * (1.0 - smoothstep(0.7, 1.0, hi));
    return vec4f(color, confidence);
}

struct ForwardVertex {
    @builtin(position) pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) world_tangent: vec4<f32>,
    @location(5) @interpolate(flat) instance: u32,
}

@vertex
fn vs_forward(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) tangent: vec4<f32>,
    @builtin(instance_index) instance: u32,
) -> ForwardVertex {
    let m = models[instance];
    let world = m.model * vec4f(position, 1.0);
    var out: ForwardVertex;
    out.pos = camera.view_proj * world;
    out.world_pos = world.xyz / world.w;
    out.world_normal = normalize((m.normal * vec4f(normal, 0.0)).xyz);
    out.color = color;
    out.uv = uv;
    out.world_tangent = vec4f(normalize((m.model * vec4f(tangent.xyz, 0.0)).xyz), tangent.w);
    out.instance = instance;
    return out;
}

// Premultiplied output, blended as `src + dst × (1 − src.a)`. Base color alpha is coverage and
// scales the diffuse and emitted light; specular is kept at full strength so nearly clear glass
// still catches highlights and reflections. Of the covered part, `transmission` passes the scene
// behind through, tinted by the base color and minus what Fresnel reflects. The scene behind is
// offset by refraction through the volume thickness (see `refracted_uv`), blurred by roughness and
// absorbed along the thickness by the volume attenuation (Beer–Lambert).
@fragment
fn fs_forward(in: ForwardVertex, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    let m = models[in.instance];
    let base = textureSample(m_albedo, m_sampler, in.uv) * m.base_color * vec4f(in.color, 1.0);
    let normal_sample = textureSample(m_normal, m_sampler, in.uv).rgb;
    let mr = textureSample(m_metallic_roughness, m_sampler, in.uv);
    let emissive = textureSample(m_emissive, m_sampler, in.uv).rgb * m.emissive.rgb;

    var Ng = normalize(in.world_normal);
    if !front_facing {
        Ng = -Ng;
    }
    let T = normalize(in.world_tangent.xyz);
    let B = cross(Ng, T) * in.world_tangent.w;
    let tangent_normal = normalize(normal_sample * 2.0 - 1.0);
    let N = normalize(T * tangent_normal.x + B * tangent_normal.y + Ng * tangent_normal.z);

    let albedo = select(base.rgb, vec3f(1.0), debug.disable_albedo != 0u);
    let coverage = base.a;
    let roughness = max(mr.g * m.material.w, MIN_ROUGHNESS);
    let metallic = mr.b * m.material.z;
    let transmission = m.transmission.x * (1.0 - metallic);
    let ior = m.transmission.y;
    let f0_dielectric = pow((ior - 1.0) / (ior + 1.0), 2.0);
    let f0 = mix(vec3f(f0_dielectric), albedo, metallic);
    let alpha = roughness * roughness;

    // inv_view is rigid, so v * R maps world directions to view space.
    let cam_pos = camera.inv_view[3].xyz;
    let rot = mat3x3<f32>(camera.inv_view[0].xyz, camera.inv_view[1].xyz, camera.inv_view[2].xyz);
    let pos_ws = in.world_pos;
    let pos_vs = (pos_ws - cam_pos) * rot;
    let N_vs = normalize(N * rot);
    let view_dir = normalize(cam_pos - pos_ws);

    let diffuse_color = albedo * (1.0 - metallic) * (1.0 - transmission) * coverage;
    let surf_vs = Surface(N_vs, normalize(-pos_vs), diffuse_color, f0, alpha);
    let surf_ws = Surface(N, view_dir, diffuse_color, f0, alpha);

    let n_dot_v = max(dot(N, view_dir), 1e-4);
    let reflectance = select(env_brdf_approx(f0, roughness, n_dot_v), vec3f(0.0), debug.disable_specular != 0u);

    var color = emissive * coverage;
    if debug.disable_direct == 0u {
        color += direct_lighting(surf_vs, surf_ws, pos_vs, pos_ws, in.pos.xy, -pos_vs.z);
    }
    // Exposed, unlike `color`: screen-space hits come from the exposed `t_behind`.
    var reflection = vec3f(0.0);
    if debug.disable_indirect == 0u {
        color += diffuse_color * sample_probe_irradiance(pos_ws, N, N, view_dir);
        if debug.disable_reflections == 0u && any(reflectance > vec3f(0.0)) {
            let R = normalize(mix(reflect(-view_dir, N), N, alpha));
            reflection = sample_probe_irradiance(pos_ws, N, R, view_dir) * cp.exposure;
            if debug.disable_ssr == 0u && roughness < GLASS_SSR_MAX_ROUGHNESS {
                let hit = glass_ssr(pos_ws, reflect(-view_dir, N), in.pos.xy, roughness);
                reflection = mix(reflection, hit.rgb, hit.a);
            }
        }
    }

    // Volume properties are mesh-local: scale them by the model's mean axis scale.
    let scale = (length(m.model[0].xyz) + length(m.model[1].xyz) + length(m.model[2].xyz)) / 3.0;
    let thickness = m.transmission.z * scale;
    var behind = vec3f(0.0);
    if transmission > 0.0 {
        let uv = refracted_uv(m, in.pos, pos_ws, N, view_dir, thickness);
        // Rough transmission blurs the scene behind; roughness matters less as the IOR nears 1
        // (glTF sample viewer convention).
        let screen = vec2f(textureDimensions(t_behind));
        let lod = log2(max(screen.x, screen.y)) * mr.g * m.material.w * clamp(ior * 2.0 - 2.0, 0.0, 1.0);
        behind = textureSampleLevel(t_behind, s_linear, uv, lod).rgb;
        if m.transmission.w > 0.0 && thickness > 0.0 {
            behind *= pow(m.attenuation.rgb, vec3f(thickness / m.transmission.w));
        }
    }
    let transmitted = behind * albedo * (1.0 - reflectance) * (transmission * coverage);
    return vec4f(color * cp.exposure + reflectance * reflection + transmitted, coverage);
}
