// GBuffer inputs
@group(0) @binding(0) var t_albedo:   texture_2d<f32>;
@group(0) @binding(1) var t_normal:   texture_2d<f32>;
@group(0) @binding(2) var t_material: texture_2d<f32>;
@group(0) @binding(3) var t_depth:    texture_depth_2d;
@group(0) @binding(4) var s_linear:   sampler;
// Emitted radiance (texture × factor × strength), not yet exposed.
@group(0) @binding(5) var t_emissive: texture_2d<f32>;
// rgb: diffuse transmission color, a: diffuse transmission factor.
@group(0) @binding(15) var t_transmission: texture_2d<f32>;
// Screen-space ambient occlusion (rendering::pass::ao), 1 = unoccluded.
@group(0) @binding(16) var t_ao: texture_2d<f32>;
// The `ao` pass's glass prepass: the nearest glass in front of the opaque depth (reversed-Z, 0 =
// none) and the light it lets through (rgb), for contact shadows.
@group(0) @binding(17) var t_glass_depth: texture_depth_2d;
@group(0) @binding(18) var t_glass_transmittance: texture_2d<f32>;

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

// Mirrors rendering::pass::reflection_probe::ReflectionProbeParams.
struct ReflectionProbeParams {
    // xyz: world-space center; w: 1 once captured.
    center:  vec4<f32>,
    max_lod: f32,
}

// The camera-following reflection probe, GGX-prefiltered: mip = perceptual roughness × max_lod.
// rgb: unexposed radiance; a: distance from the center to what was captured in that direction.
@group(3) @binding(5) var t_reflection: texture_cube<f32>;
@group(3) @binding(6) var<uniform> reflection: ReflectionProbeParams;
@group(3) @binding(7) var s_reflection: sampler;

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
    disable_contact_shadows: u32,
}

@group(4) @binding(0) var<uniform> debug: DebugFlags;

// Mirrors rendering::light::ShadowedLights. Lights point at their slot through `shadow_slot`; the
// slot's maps and the light they were rendered for are in `local_shadow.faces`.
struct ShadowedLights {
    count: u32,
}

// Mirrors rendering::light::SHADOW_SLOT_NONE.
const SHADOW_SLOT_NONE: u32 = 0xffffffffu;

@group(4) @binding(1) var<uniform> shadowed: ShadowedLights;


// Widest blocker search and PCF kernel of a local light shadow, in texels of its face: 12 taps
// get sparse beyond this.
const MAX_PENUMBRA_TEXELS: f32 = 16.0;

// 12-tap Poisson disc on the unit disc, for local light shadow PCF.
const POISSON: array<vec2<f32>, 12> = array<vec2<f32>, 12>(
    vec2f(-0.326, -0.406), vec2f(-0.840, -0.074), vec2f(-0.696,  0.457),
    vec2f(-0.203,  0.621), vec2f( 0.962, -0.195), vec2f( 0.473, -0.480),
    vec2f( 0.519,  0.767), vec2f( 0.185, -0.893), vec2f( 0.507,  0.064),
    vec2f( 0.896,  0.412), vec2f(-0.322, -0.933), vec2f(-0.792, -0.598),
);

// Local light shadows are virtual shadow maps (rendering::light::virtual_shadow): each face is a
// VSM_SIZE² map with VSM_MIPS mips in VSM_PAGE² pages, of which the ones visible pixels asked for
// (rendering::pass::shadow_pages) are backed by physical pages of the atlas. `local_pages` holds a
// physical page + 1 per virtual page, 0 where none is resident; the coarsest mip always is.
const VSM_SIZE: f32 = 8192.0;
const VSM_PAGES0: u32 = 64u;
const VSM_MIPS: u32 = 7u;
const VSM_PAGES_PER_FACE: u32 = 5461u;
const VSM_PAGE: f32 = 128.0;
const VSM_POOL_SIDE: u32 = 32u;
@group(2) @binding(7) var<storage, read> local_pages: array<u32>;
// The mip the last `light_shadow` call resolved to (-1 if it had no map), for the Shadow detail
// debug view.
var<private> vsm_resolved_mip: f32 = -1.0;

fn vsm_mip_offset(mip: u32) -> u32 {
    var offset = 0u;
    for (var m = 0u; m < mip; m++) {
        let n = VSM_PAGES0 >> m;
        offset += n * n;
    }
    return offset;
}

// Where face UV `uv` of face `face_index` is in the atlas (xy, atlas UV) at `mip`, or at the
// nearest coarser mip with a resident page (z), with w = 1; w = 0 if no page is resident at all.
// The position is kept half a texel inside its page, so bilinear taps never read a neighbour's.
fn vsm_lookup(face_index: u32, uv: vec2<f32>, mip: u32) -> vec4<f32> {
    let atlas = f32(textureDimensions(t_local_shadow).x);
    for (var m = mip; m < VSM_MIPS; m++) {
        let n = VSM_PAGES0 >> m;
        let p = clamp(uv, vec2f(0.0), vec2f(1.0)) * f32(n);
        let page = min(vec2<u32>(p), vec2<u32>(n - 1u));
        let entry = local_pages[face_index * VSM_PAGES_PER_FACE + vsm_mip_offset(m) + page.y * n + page.x];
        if entry != 0u {
            let physical = entry - 1u;
            let inside = clamp(p - vec2f(page), vec2f(0.5 / VSM_PAGE), vec2f(1.0 - 0.5 / VSM_PAGE));
            let origin = vec2f(f32(physical % VSM_POOL_SIDE), f32(physical / VSM_POOL_SIDE));
            return vec4f((origin + inside) * VSM_PAGE / atlas, f32(m), 1.0);
        }
    }
    return vec4f(0.0);
}

// The mip `kernel_texels` (texels at `mip`) needs to fit within MAX_PENUMBRA_TEXELS, and the
// kernel in that mip's texels: wide penumbrae move to coarser mips rather than shrink.
fn vsm_kernel_mip(mip: u32, kernel_texels: f32) -> vec2<f32> {
    let up = u32(clamp(ceil(log2(max(kernel_texels / MAX_PENUMBRA_TEXELS, 1.0))), 0.0, f32(VSM_MIPS - 1u - mip)));
    return vec2f(f32(mip + up), kernel_texels / f32(1u << up));
}

// Visibility of shadowed light `slot` at world position `pos` (normal `N`), from its virtual
// shadow map: a cube picks the face by the dominant axis of the light-to-point direction, a spot
// has one. The lookup mip is the one whose texels match a screen pixel here (the rule the page
// requests use), or the nearest coarser resident one. Area and sphere lights get penumbrae from
// their size (PCSS): a blocker search finds the average occluder depth, opaque or glass, and the
// PCF radius is the light's size × (receiver − blocker) / blocker, so shadows are sharp at contact
// and soften away from it. Search and filter move to coarser mips when wider than
// MAX_PENUMBRA_TEXELS, so a penumbra keeps its world size whatever the mip.
fn light_shadow(slot: u32, pos: vec3<f32>, N: vec3<f32>, pixel: vec2<f32>) -> vec3<f32> {
    let first = local_shadow.faces[slot * 6u];
    let light = first.light;
    let shape = first.shape;
    let d = pos - light.xyz;
    let dist = length(d);
    var face = 0u;
    if shape.x > 1.5 {
        let a = abs(d);
        if a.x >= a.y && a.x >= a.z {
            face = select(1u, 0u, d.x > 0.0);
        } else if a.y >= a.z {
            face = select(3u, 2u, d.y > 0.0);
        } else {
            face = select(5u, 4u, d.z > 0.0);
        }
    }
    let face_index = slot * 6u + face;
    let f = local_shadow.faces[face_index];
    // No pages: no shadowed light, or behind a one-sided light (which gives no light there anyway).
    if shape.x == 0.0 || f.info.x == 0u {
        return vec3f(1.0);
    }
    let tan_half = shape.y;
    let vp = f.view_proj;

    let eye = camera.inv_view[3].xyz;
    let footprint = max(dot(pos - eye, -camera.inv_view[2].xyz), 0.0) * local_shadow.lod.x;
    let texel0 = 2.0 * (vp * vec4f(pos, 1.0)).w * tan_half / VSM_SIZE;
    let wanted = u32(clamp(floor(log2(max(footprint / texel0, 1.0))), 0.0, f32(VSM_MIPS - 1u)));

    // Offset ~1.5 texels along the normal, at the mip actually resident there.
    var mip = wanted;
    var resolved = vec4f(0.0);
    var p = pos;
    var clip = vec4f(0.0);
    var uv = vec2f(0.0);
    for (var attempt = 0u; attempt < 2u; attempt++) {
        let texel = 2.0 * dist * tan_half / (VSM_SIZE / f32(1u << mip));
        p = pos + N * (1.5 * texel);
        clip = vp * vec4f(p, 1.0);
        uv = (clip.xy / clip.w) * vec2f(0.5, -0.5) + 0.5;
        resolved = vsm_lookup(face_index, uv, mip);
        if resolved.w == 0.0 || u32(resolved.z) == mip {
            break;
        }
        mip = u32(resolved.z);
    }
    let ndc = clip.xyz / clip.w;
    // Outside a spot's map is outside its cone, where it gives no light anyway. A cube lookup can
    // land just past its face's edge (normal offset); `vsm_lookup` clamps it onto the face.
    if resolved.w == 0.0 || (shape.x < 1.5 && (clip.w <= 0.0 || any(abs(ndc.xy) > vec2f(1.0)))) {
        return vec3f(1.0);
    }
    vsm_resolved_mip = f32(mip);
    let size = VSM_SIZE / f32(1u << mip);
    let texel = 2.0 * dist * tan_half / size;

    // Receiver-plane depth bias: the depth slope of the receiver's tangent plane in face UV, from
    // two projected tangent points, so each tap of a wide kernel is compared against the plane's
    // depth under that tap instead of the center's (no self-shadowing grain on surfaces tilted
    // away from the light). The slope is clamped where the plane turns edge-on.
    let t1 = normalize(cross(N, select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(N.y) > 0.9)));
    let t2 = cross(N, t1);
    let c1 = vp * vec4f(p + t1 * texel, 1.0);
    let c2 = vp * vec4f(p + t2 * texel, 1.0);
    let du1 = (c1.xy / c1.w) * vec2f(0.5, -0.5) + 0.5 - uv;
    let du2 = (c2.xy / c2.w) * vec2f(0.5, -0.5) + 0.5 - uv;
    let dz1 = c1.z / c1.w - ndc.z;
    let dz2 = c2.z / c2.w - ndc.z;
    let det = du1.x * du2.y - du1.y * du2.x;
    var slope = vec2f(0.0);
    if abs(det) > 1e-12 {
        slope = vec2f(dz1 * du2.y - dz2 * du1.y, du1.x * dz2 - du2.x * dz1) / det;
    }
    // The slope's offset is capped at the depth change of moving along the light ray by as far
    // as the tap moved sideways: per texel of `mip` here, times the kernel radius below.
    let cap = vp * vec4f(p - (d / dist) * texel, 1.0);
    let max_dz_per_texel = abs(cap.z / cap.w - ndc.z);

    let angle = fract(52.9829189 * fract(dot(pixel, vec2f(0.06711056, 0.00583715)))) * 6.28318531;
    let rot = mat2x2<f32>(cos(angle), sin(angle), -sin(angle), cos(angle));
    // Tangent of the field of view to texels of this face at `mip`.
    let to_texels = size / (2.0 * tan_half);
    // Glass in the tint atlas (alpha: its distance along the face axis, like `clip.w`) shadows the
    // receiver when it's in front by more than one texel, so glass doesn't tint itself.
    let atlas = vec2f(textureDimensions(t_local_shadow));
    let tint_atlas = vec2f(textureDimensions(t_local_glass_tint));
    let glass_reference = clip.w - texel;

    var radius_texels = 1.5;
    var filter_mip = mip;
    if light.w > 0.0 {
        // Blocker search over where occluders as far as two thirds of the way back to the light
        // can still partly hide it: light size × (receiver − blocker) / (receiver × blocker).
        // Glass counts as a blocker too, so its shadows soften with distance like opaque ones.
        // The search stays at `mip`, capped at MAX_PENUMBRA_TEXELS: at a coarser mip, a tilted
        // receiver's own coarse depth reads as an occluder.
        let receiver = clip.w;
        let search_texels = clamp(2.0 * light.w / receiver * to_texels, 1.5, MAX_PENUMBRA_TEXELS);
        let max_dz = max_dz_per_texel * search_texels;
        var blockers = 0.0;
        var blocker_depth = 0.0;
        for (var i = 0u; i < 12u; i++) {
            let offset = rot * POISSON[i] * (search_texels / size);
            let at = vsm_lookup(face_index, uv + offset, mip);
            if at.w == 0.0 {
                continue;
            }
            let reference = ndc.z + clamp(dot(slope, offset), -max_dz, max_dz);
            let z = textureLoad(t_local_shadow, vec2<i32>(at.xy * atlas), 0);
            if z < reference {
                blockers += 1.0;
                blocker_depth += shape.z * shape.w / (shape.w - z * (shape.w - shape.z));
            }
            let glass = textureLoad(t_local_glass_tint, vec2<i32>(at.xy * tint_atlas), 0).a;
            if glass < glass_reference {
                blockers += 1.0;
                blocker_depth += glass;
            }
        }
        if blockers == 0.0 {
            return local_glass_tint(resolved.xy, glass_reference);
        }
        let blocker = blocker_depth / blockers;
        let radius = vsm_kernel_mip(mip, light.w * max(receiver - blocker, 0.0) / (receiver * blocker) * to_texels);
        filter_mip = u32(radius.x);
        radius_texels = max(radius.y, 1.5);
    }

    // Each tap is the opaque comparison times the glass transmittance there, so glass shadows get
    // the same kernel (and penumbra) as opaque ones. A coarser filter mip stores depth at coarser
    // texel centres, off the receiver plane by up to its texel's depth slope, so the reference
    // moves toward the light by that much more than the normal offset at `mip` covers.
    let filter_scale = f32(1u << (filter_mip - mip));
    let radius = radius_texels * filter_scale / size;
    let max_dz = max_dz_per_texel * filter_scale * radius_texels;
    let coarse_bias = max_dz_per_texel * (filter_scale - 1.0);
    var lit = vec3f(0.0);
    var taps = 0.0;
    for (var i = 0u; i < 12u; i++) {
        let offset = rot * POISSON[i] * radius;
        let at = vsm_lookup(face_index, uv + offset, filter_mip);
        if at.w == 0.0 {
            continue;
        }
        taps += 1.0;
        let depth = ndc.z + clamp(dot(slope, offset), -max_dz, max_dz) - coarse_bias;
        let opaque = textureSampleCompareLevel(t_local_shadow, s_shadow, at.xy, depth);
        if opaque > 0.0 {
            lit += opaque * local_glass_tint(at.xy, glass_reference);
        }
    }
    return lit / max(taps, 1.0);
}

// Visibility of a local light at this pixel, tinted by glass: its shadow if it has a shadow slot,
// else 1.
fn local_shadow_of(slot: u32, pos: vec3<f32>, N: vec3<f32>, pixel: vec2<f32>) -> vec3<f32> {
    if debug.disable_shadows != 0u || slot == SHADOW_SLOT_NONE { return vec3f(1.0); }
    return light_shadow(slot, pos, N, pixel);
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
    // Diffuse transmission (KHR_materials_diffuse_transmission, color × factor): Lambertian light
    // arriving from behind the surface, for thin foliage.
    transmission: vec3<f32>,
}

// `v_dot_h` can round to just above 1 when L is tiny (a point on an area light's rectangle), and
// pow of a negative base is NaN.
fn fresnel_schlick(f0: vec3<f32>, v_dot_h: f32) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(saturate(1.0 - v_dot_h), 5.0);
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
    return (1.0 - F) * s.diffuse_color / PI * n_dot_l + s.transmission / PI * max(-dot(s.N, L), 0.0);
}

// The normal to offset shadow lookups along: toward the light, which is behind transmissive surfaces.
fn shadow_normal(N: vec3<f32>, N_vs: vec3<f32>, to_light_vs: vec3<f32>) -> vec3<f32> {
    return select(N, -N, dot(N_vs, to_light_vs) < 0.0);
}

fn diffuse_transmission(t: vec4<f32>, metallic: f32) -> vec3<f32> {
    return t.rgb * t.a * (1.0 - metallic);
}

fn brdf(s: Surface, L: vec3<f32>) -> vec3<f32> {
    return diffuse_brdf(s, L) + specular_brdf(s, L, s.alpha);
}

// A spherical light of radius `source_radius` at `to_light` (Karis 2013, representative point):
// diffuse from its center, specular from the point on the sphere closest to the reflection ray,
// with the roughness widened by the sphere's angular size, as for area lights.
fn sphere_light_brdf(s: Surface, to_light: vec3<f32>, dist: f32, source_radius: f32) -> vec3<f32> {
    let L = to_light / dist;
    if source_radius <= 0.0 {
        return brdf(s, L);
    }
    let R = reflect(-s.V, s.N);
    let center_to_ray = dot(to_light, R) * R - to_light;
    let closest = to_light + center_to_ray * saturate(source_radius / max(length(center_to_ray), 1e-4));
    let spec_alpha = saturate(s.alpha + source_radius / (2.0 * dist));
    return diffuse_brdf(s, L) + specular_brdf(s, normalize(closest), spec_alpha);
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
// Radiance (unexposed) arriving along `R` at `pos_ws`, for reflections SSR can't resolve: the
// reflection probe once captured, else DDGI irradiance around `R`. The probe was captured from its
// center, not from `pos_ws`, so the lookup direction is parallax-corrected with the captured
// distance: a few fixed-point steps find where the ray from `pos_ws` meets the captured surface.
fn reflected_radiance(pos_ws: vec3<f32>, N: vec3<f32>, R: vec3<f32>, view_dir: vec3<f32>, roughness: f32) -> vec3<f32> {
    if reflection.center.w < 0.5 {
        return sample_probe_irradiance(pos_ws, N, R, view_dir);
    }
    let c = reflection.center.xyz;
    var dir = R;
    var t = textureSampleLevel(t_reflection, s_reflection, R, 0.0).a;
    for (var i = 0u; i < 4u; i++) {
        let v = pos_ws + R * t - c;
        dir = select(R, normalize(v), dot(v, v) > 1e-6);
        let hit = c + dir * textureSampleLevel(t_reflection, s_reflection, dir, 0.0).a;
        t = max(dot(hit - pos_ws, R), 0.0);
    }
    let v = pos_ws + R * t - c;
    dir = select(R, normalize(v), dot(v, v) > 1e-6);
    return textureSampleLevel(t_reflection, s_reflection, dir, roughness * reflection.max_lod).rgb;
}

fn range_attenuation(dist: f32, range: f32) -> f32 {
    let r = dist / range;
    let window = clamp(1.0 - r * r * r * r, 0.0, 1.0);
    return window * window / max(dist * dist, 1e-4);
}

// Scalar padding: a trailing vec3 would be 16-aligned and break the 48-byte Rust layout.
struct GpuPointLight {
    position_vs: vec3<f32>,
    radius: f32,
    color: vec3<f32>,
    intensity: f32,
    source_radius: f32,
    // Index into `shadowed`, or SHADOW_SLOT_NONE.
    shadow_slot: u32,
    _pad1: f32,
    _pad2: f32,
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
    source_radius: f32,
    // Index into `shadowed`, or SHADOW_SLOT_NONE.
    shadow_slot: u32,
    _pad2: f32,
}

// right_vs / up_vs are half-extent vectors of the rectangle. One-sided lights emit toward right × up.
struct GpuAreaLight {
    position_vs: vec3<f32>,
    intensity: f32,
    right_vs: vec3<f32>,
    two_sided: f32,
    up_vs: vec3<f32>,
    // Index into `shadowed`, or SHADOW_SLOT_NONE.
    shadow_slot: u32,
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
// Physical pages of the local lights' virtual shadow maps (rendering::light::virtual_shadow).
@group(2) @binding(3) var          t_local_shadow: texture_depth_2d;

// Mirrors rendering::pass::shadow::LocalShadowFace. info.x: 1 if the face has pages. light:
// shadow origin (xyz), half size (w). shape: face count (6 cube, 1 spot; 0 = no light), tan of
// half the field of view, near, far.
struct LocalShadowFace {
    view_proj: mat4x4<f32>,
    info:      vec4<u32>,
    light:     vec4<f32>,
    shape:     vec4<f32>,
}

// Mirrors rendering::pass::shadow::LocalShadowParams: six faces per slot (+X −X +Y −Y +Z −Z; a
// spot light uses the first).
struct LocalShadowParams {
    // x: world size of a screen pixel per metre of view depth.
    lod:   vec4<f32>,
    faces: array<LocalShadowFace, 192>,
}

@group(2) @binding(4) var<uniform> local_shadow: LocalShadowParams;
// Glass tint maps, one layer per cascade / cube face (rendering::pass::shadow): rgb = transmittance
// of all glass along the light ray, a = depth of the glass nearest the light (clip z for cascades,
// clip w for cube faces).
@group(2) @binding(5) var          t_glass_tint: texture_2d_array<f32>;
@group(2) @binding(6) var          t_local_glass_tint: texture_2d<f32>;

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

// `glass_tint` at `atlas_uv` in the local tint atlas (from `vsm_lookup`, which keeps it half a
// texel inside its page, so the 4 taps stay on that page).
fn local_glass_tint(atlas_uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let p = atlas_uv * vec2f(textureDimensions(t_local_glass_tint)) - 0.5;
    let base = floor(p);
    let f = p - base;
    var tint = vec3f(0.0);
    for (var i = 0u; i < 4u; i++) {
        let o = vec2f(f32(i & 1u), f32(i >> 1u));
        let s = textureLoad(t_local_glass_tint, vec2<i32>(base + o), 0);
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

// Contact shadows: a short march toward the light through the depth buffer catches the small-scale
// contact the shadow maps are too coarse for. After Bend Studio's screen-space shadows (Days Gone):
// the ray is stepped about a pixel at a time in screen space with no jitter, and each sample
// occludes by how far behind the depth buffer it is, so edges come out soft and stable without
// temporal accumulation.
const CONTACT_LENGTH: f32 = 0.15;
const CONTACT_MAX_STEPS: u32 = 24u;
// A sample starts to be occluded once the depth buffer is in front of it by this fraction of its
// depth (self-shadowing margin), fully by twice that, and stops past CONTACT_THICKNESS (so it
// doesn't hide behind everything in front of it).
const CONTACT_BIAS: f32 = 0.004;
const CONTACT_THICKNESS: f32 = 0.1;
// Occlusion fades out over this last fraction of the ray, so its end leaves no edge.
const CONTACT_END_FADE: f32 = 0.25;

// Transmittance along `L_ws` from `pos_ws` over up to `max_length`: down to 0 past an opaque
// occluder, and with `glass`, toward the transmittance of the glass prepass's glass it passes
// behind. `geo_N` is the surface's geometric normal (not the shading normal), or 0 for none. Lights
// behind the geometric surface are left to the shadow maps: normal-mapped detail facing them would
// otherwise march into the surface. Blended surfaces don't march the glass, which holds their own
// surface.
//
// Combine it with a shadow map's visibility by `min`, not a product: glass the tint map already
// caught would otherwise tint twice.
fn contact_shadow(geo_N: vec3<f32>, pos_ws: vec3<f32>, L_ws: vec3<f32>, max_length: f32, glass: bool) -> vec3<f32> {
    if all(geo_N == vec3f(0.0)) || dot(geo_N, L_ws) <= 0.0 || debug.disable_shadows != 0u
        || debug.disable_contact_shadows != 0u {
        return vec3f(1.0);
    }
    let dims = vec2f(cp.screen_w, cp.screen_h);
    let start = pos_ws + geo_N * (CONTACT_BIAS * distance(pos_ws, camera.inv_view[3].xyz));
    let h0 = camera.view_proj * vec4f(start, 1.0);
    var h1 = camera.view_proj * vec4f(start + L_ws * min(CONTACT_LENGTH, max_length), 1.0);
    // Clip the ray at the near plane (clip w is linear along it).
    let near = camera.depth_params.x;
    if h1.w < near {
        h1 = mix(h0, h1, (h0.w - near) / (h0.w - h1.w));
    }
    let k0 = 1.0 / h0.w;
    let k1 = 1.0 / h1.w;
    let p0 = clip_to_pixel(h0, dims);
    let delta = clip_to_pixel(h1, dims) - p0;
    // Steps land on pixel centers along the ray's dominant screen axis, and depth is interpolated
    // across the other: a point-sampled pixel's depth is off the ray's own by up to half a pixel of
    // the surface's slope, which speckles surfaces the light grazes with self-shadowing.
    let x_major = abs(delta.x) >= abs(delta.y);
    let major = select(vec2f(delta.y, p0.y), vec2f(delta.x, p0.x), x_major);
    let pixels = abs(major.x);
    let stride = max(ceil(pixels / f32(CONTACT_MAX_STEPS)), 1.0);
    let first = floor(major.y) + 0.5;

    var shadow = 0.0;
    var through = vec3f(1.0);
    for (var i = 1u; i <= CONTACT_MAX_STEPS; i++) {
        let s = (first + sign(major.x) * f32(i) * stride - major.y) / major.x;
        if s > 1.0 || pixels < 0.5 {
            break;
        }
        let px = p0 + delta * s;
        if any(px < vec2f(0.0)) || any(px >= dims) {
            break;
        }
        let k = mix(k0, k1, s);
        let ray_w = 1.0 / k;
        // `s` is linear in screen space; the fade wants the distance along the ray.
        let t = s * k1 / k;
        let fade = saturate((1.0 - t) / CONTACT_END_FADE);
        let scene_w = contact_depth(px, x_major);
        if scene_w > 0.0 {
            shadow = max(shadow, contact_occlusion(ray_w - scene_w, ray_w) * fade);
            if shadow >= 1.0 {
                return vec3f(0.0);
            }
        }
        if glass {
            let g = textureLoad(t_glass_depth, vec2<i32>(px), 0);
            if g != 0.0 {
                let occlusion = contact_occlusion(ray_w - linear_depth(g), ray_w) * fade;
                through = min(through, mix(vec3f(1.0), textureLoad(t_glass_transmittance, vec2<i32>(px), 0).rgb, occlusion));
            }
        }
    }
    return through * (1.0 - shadow);
}

// Linear depth of the depth buffer at `px`, interpolated across the minor axis (y when `x_major`)
// between the two nearest pixel centers; 0 for sky. Across a depth edge (the two differ by more than
// the contact thickness) it takes the nearer, so silhouettes don't blend into what's behind them.
fn contact_depth(px: vec2<f32>, x_major: bool) -> f32 {
    let minor = select(px.x, px.y, x_major) - 0.5;
    let base = floor(minor);
    let a = vec2<i32>(select(vec2f(base, px.y), vec2f(px.x, base), x_major));
    let b = a + select(vec2<i32>(1, 0), vec2<i32>(0, 1), x_major);
    let size = vec2<i32>(i32(cp.screen_w), i32(cp.screen_h)) - 1;
    let da = textureLoad(t_depth, clamp(a, vec2<i32>(0), size), 0);
    let db = textureLoad(t_depth, clamp(b, vec2<i32>(0), size), 0);
    if da == 0.0 || db == 0.0 {
        return select(0.0, linear_depth(max(da, db)), max(da, db) != 0.0);
    }
    let wa = linear_depth(da);
    let wb = linear_depth(db);
    if abs(wa - wb) > CONTACT_THICKNESS {
        return min(wa, wb);
    }
    return mix(wa, wb, minor - base);
}

// How much a sample `in_front` behind the depth buffer is occluded.
fn contact_occlusion(in_front: f32, ray_w: f32) -> f32 {
    let bias = CONTACT_BIAS * ray_w;
    return saturate((in_front - bias) / bias) * f32(in_front < CONTACT_THICKNESS);
}

fn gbuffer_pos_ws(pixel: vec2<i32>) -> vec4<f32> {
    let p = clamp(pixel, vec2<i32>(0), vec2<i32>(i32(cp.screen_w), i32(cp.screen_h)) - 1);
    let d = textureLoad(t_depth, p, 0);
    let ndc = vec2f((f32(p.x) + 0.5) / cp.screen_w * 2.0 - 1.0, 1.0 - (f32(p.y) + 0.5) / cp.screen_h * 2.0);
    let h = camera.inv_view_proj * vec4f(ndc, d, 1.0);
    return vec4f(h.xyz / h.w, d);
}

// The normal of the depth buffer's surface at `pixel`, facing `view_dir`. Each axis takes the
// neighbour nearer in depth, so it holds up at object edges.
fn geometric_normal(pixel: vec2<i32>, pos_ws: vec3<f32>, view_dir: vec3<f32>) -> vec3<f32> {
    var axes: array<vec3<f32>, 2>;
    for (var a = 0; a < 2; a++) {
        let step = vec2<i32>(i32(a == 0), i32(a == 1));
        let fwd = gbuffer_pos_ws(pixel + step);
        let back = gbuffer_pos_ws(pixel - step);
        let to_fwd = fwd.xyz - pos_ws;
        let from_back = pos_ws - back.xyz;
        let use_back = fwd.w == 0.0 || (back.w != 0.0 && dot(from_back, from_back) < dot(to_fwd, to_fwd));
        axes[a] = select(to_fwd, from_back, use_back);
    }
    let n = normalize(cross(axes[0], axes[1]));
    return select(n, -n, dot(n, view_dir) < 0.0);
}

fn view_to_world_dir(v: vec3<f32>) -> vec3<f32> {
    return (camera.inv_view * vec4f(v, 0.0)).xyz;
}

// Direct light from every light, with shadows. `surf_vs` / `surf_ws` are the same surface in view
// and world space, and `pixel` is the fragment position (cluster lookup, PCF rotation).
// `probe_capture` is for points seen by the reflection probe rather than the camera: the camera's
// light clusters don't apply, so every point light is looped over, and the sky shadow comes from
// whichever cascade contains the point. A nonzero `geo_N` (the geometric normal) adds contact
// shadows, which march the camera view's depth, so the probe capture passes 0; `contact_glass` is
// `contact_shadow`'s `glass`.
fn direct_lighting(surf_vs: Surface, surf_ws: Surface, pos_vs: vec3<f32>, pos_ws: vec3<f32>, pixel: vec2<f32>, depth_vs: f32, probe_capture: bool, geo_N: vec3<f32>, contact_glass: bool) -> vec3<f32> {
    let N = surf_ws.N;
    let N_vs = surf_vs.N;
    let alpha = surf_vs.alpha;
    var cluster_idx = 0u;
    var light_count = light_counts.num_point;
    if !probe_capture {
        cluster_idx = cluster_index(pixel, depth_vs);
        light_count = light_grid[cluster_idx];
    }

    var color = vec3f(0.0);
    for (var i = 0u; i < light_count; i++) {
        var li = i;
        if !probe_capture {
            li = light_indices[cluster_idx * MAX_LIGHTS_PER_CLUSTER + i];
        }
        let light = point_lights[li];

        let to_light = light.position_vs - pos_vs;
        let dist     = length(to_light);
        if dist >= light.radius {
            continue;
        }

        let vis = min(local_shadow_of(light.shadow_slot, pos_ws, shadow_normal(N, N_vs, to_light), pixel),
            contact_shadow(select(vec3f(0.0), geo_N, light.shadow_slot != SHADOW_SLOT_NONE), pos_ws, view_to_world_dir(to_light / dist), dist, contact_glass));
        let falloff = range_attenuation(max(dist, light.source_radius), light.radius);
        color += sphere_light_brdf(surf_vs, to_light, dist, light.source_radius) * light.color * vis
            * (light.intensity * falloff);
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
        let vis = min(local_shadow_of(light.shadow_slot, pos_ws, shadow_normal(N, N_vs, to_light), pixel),
            contact_shadow(select(vec3f(0.0), geo_N, light.shadow_slot != SHADOW_SLOT_NONE), pos_ws, view_to_world_dir(to_light / dist), dist, contact_glass));
        let falloff = range_attenuation(max(dist, light.source_radius), light.radius);
        color += sphere_light_brdf(surf_vs, to_light, dist, light.source_radius) * light.color * vis
            * (light.intensity * t * t * falloff);
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
        let radiance = light.color * light.intensity
            * min(local_shadow_of(light.shadow_slot, pos_ws, shadow_normal(N, N_vs, to_light), pixel),
                contact_shadow(select(vec3f(0.0), geo_N, light.shadow_slot != SHADOW_SLOT_NONE), pos_ws, view_to_world_dir(L), sqrt(dist2), contact_glass));
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

        // Select cascade based on view-space depth and project into shadow space. The cascades
        // only cover the camera's frustum, so a probe capture takes the first cascade containing
        // the point and treats a point none covers as shadowed: indoor reflections must not gain
        // moonlight.
        var cascade = select_cascade(depth_vs);
        var covered = true;
        if probe_capture {
            covered = false;
            for (var c = 0u; c < 4u; c++) {
                let clip = sp.view_to_shadow[c] * vec4f(pos_vs, 1.0);
                let ndc = clip.xyz / clip.w;
                if all(abs(ndc.xy) <= vec2f(0.98)) && ndc.z >= 0.0 && ndc.z <= 1.0 {
                    cascade = c;
                    covered = true;
                    break;
                }
            }
        }
        let shadow_clip = sp.view_to_shadow[cascade] * vec4<f32>(pos_vs, 1.0);
        let shadow_ndc  = shadow_clip.xyz / shadow_clip.w;
        let shadow_uv   = shadow_ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
        let ref_depth   = shadow_ndc.z - sp.bias;

        var lit = vec3f(1.0);
        if debug.disable_shadows == 0u && !covered {
            lit = vec3f(0.0);
        } else if debug.disable_shadows == 0u &&
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

        lit = min(lit, contact_shadow(geo_N, pos_ws, L, CONTACT_LENGTH, contact_glass));
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

// Ambient occlusion brightened by interreflections off the occluders, assumed to share the
// surface's albedo (Jimenez et al. 2016, cubic fit).
fn multi_bounce_ao(ao: f32, albedo: vec3<f32>) -> vec3<f32> {
    let a = 2.0404 * albedo - 0.3324;
    let b = -4.7951 * albedo + 0.6417;
    let c = 2.7552 * albedo + 0.6903;
    return max(vec3f(ao), ((ao * a + b) * ao + c) * ao);
}

// Occlusion of reflections from ambient occlusion (Lagarde & de Rousiers 2014): creases lose
// reflections, more so for rough surfaces and at grazing angles.
fn specular_occlusion(n_dot_v: f32, ao: f32, roughness: f32) -> f32 {
    return saturate(pow(n_dot_v + ao, exp2(-16.0 * roughness - 1.0)) - 1.0 + ao);
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
        case 3u: {
            let material = textureSample(t_material, s_linear, in.uv);
            return unreflective(vec3f(material.rg, material.b * textureLoad(t_ao, frag_xy, 0).r));
        }
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
    let transmission_raw = textureSample(t_transmission, s_linear, in.uv);
    let ao         = material.b * textureLoad(t_ao, frag_xy, 0).r;

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

    if debug.view == 6u {
        let colors = array<vec3<f32>, 7>(
            vec3f(1.0, 0.0, 0.0), vec3f(1.0, 0.5, 0.0), vec3f(1.0, 1.0, 0.0), vec3f(0.0, 1.0, 0.0),
            vec3f(0.0, 1.0, 1.0), vec3f(0.0, 0.0, 1.0), vec3f(0.5, 0.0, 1.0),
        );
        for (var slot = 0u; slot < shadowed.count; slot++) {
            vsm_resolved_mip = -1.0;
            let first = local_shadow.faces[slot * 6u];
            if first.shape.x == 0.0 || distance(pos_ws, first.light.xyz) > first.shape.w {
                continue;
            }
            light_shadow(slot, pos_ws, N, in.pos.xy);
            if vsm_resolved_mip >= 0.0 {
                return unreflective(colors[u32(vsm_resolved_mip)] * 0.5 + albedo * 0.5);
            }
        }
        return unreflective(albedo * 0.3);
    }
    if debug.view == 5u {
        var vis = vec3f(1.0);
        for (var slot = 0u; slot < min(shadowed.count, 3u); slot++) {
            vis[slot] = dot(light_shadow(slot, pos_ws, N, in.pos.xy), vec3f(1.0 / 3.0));
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

    let diffuse_color = albedo * (1.0 - metallic) * (1.0 - transmission_raw.a);
    let transmission = diffuse_transmission(transmission_raw, metallic);
    let f0 = mix(vec3f(0.04), albedo, metallic);
    let alpha = roughness * roughness;
    // Local lights are shaded in view space, sky lights and probes in world space.
    let surf_vs = Surface(N_vs, normalize(-pos_vs), diffuse_color, f0, alpha, transmission);
    let surf_ws = Surface(N, view_dir, diffuse_color, f0, alpha, transmission);

    var color = vec3f(0.0);
    var out: LightingOutput;
    if debug.disable_indirect == 0u {
        color = diffuse_color * multi_bounce_ao(ao, albedo) * sample_probe_irradiance(pos_ws, N, N, view_dir);
        if any(transmission > vec3f(0.0)) {
            color += transmission * sample_probe_irradiance(pos_ws, -N, -N, view_dir);
        }

        // Reflections are added by the composite pass: SSR where it hits, else probe irradiance
        // around the reflection vector, leaning toward N as roughness grows (Filament's dominant
        // direction).
        if debug.disable_specular == 0u && debug.disable_reflections == 0u {
            let n_dot_v = max(dot(N, view_dir), 1e-4);
            let R = normalize(mix(reflect(-view_dir, N), N, alpha));
            let weight = env_brdf_approx(f0, roughness, n_dot_v) * specular_occlusion(n_dot_v, ao, roughness);
            let env = reflected_radiance(pos_ws, N, R, view_dir, roughness);
            out.specular_weight = vec4f(weight, roughness);
            out.specular_fallback = vec4f(env * weight * cp.exposure, 1.0);
        }
    }

    if debug.disable_direct == 0u {
        color += direct_lighting(surf_vs, surf_ws, pos_vs, pos_ws, in.pos.xy, depth_vs, false, geometric_normal(frag_xy, pos_ws, view_dir), true);
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
    // rgb: diffuse transmission color, a: diffuse transmission factor.
    diffuse_transmission: vec4<f32>,
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

// Invariant so the reflection probe's depth prepass and shading pass produce identical depths.
struct ForwardVertex {
    @builtin(position) @invariant pos: vec4<f32>,
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

    let diffuse_color = albedo * (1.0 - metallic) * (1.0 - transmission) * (1.0 - m.diffuse_transmission.a) * coverage;
    let back_lit = diffuse_transmission(m.diffuse_transmission, metallic) * (1.0 - transmission) * coverage;
    let surf_vs = Surface(N_vs, normalize(-pos_vs), diffuse_color, f0, alpha, back_lit);
    let surf_ws = Surface(N, view_dir, diffuse_color, f0, alpha, back_lit);

    let n_dot_v = max(dot(N, view_dir), 1e-4);
    let reflectance = select(env_brdf_approx(f0, roughness, n_dot_v), vec3f(0.0), debug.disable_specular != 0u);

    var color = emissive * coverage;
    if debug.disable_direct == 0u {
        color += direct_lighting(surf_vs, surf_ws, pos_vs, pos_ws, in.pos.xy, -pos_vs.z, false, Ng, false);
    }
    // Exposed, unlike `color`: screen-space hits come from the exposed `t_behind`.
    var reflected = vec3f(0.0);
    if debug.disable_indirect == 0u {
        color += diffuse_color * sample_probe_irradiance(pos_ws, N, N, view_dir);
        if any(back_lit > vec3f(0.0)) {
            color += back_lit * sample_probe_irradiance(pos_ws, -N, -N, view_dir);
        }
        if debug.disable_reflections == 0u && any(reflectance > vec3f(0.0)) {
            let R = normalize(mix(reflect(-view_dir, N), N, alpha));
            reflected = reflected_radiance(pos_ws, N, R, view_dir, roughness) * cp.exposure;
            if debug.disable_ssr == 0u && roughness < GLASS_SSR_MAX_ROUGHNESS {
                let hit = glass_ssr(pos_ws, reflect(-view_dir, N), in.pos.xy, roughness);
                reflected = mix(reflected, hit.rgb, hit.a);
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
    return vec4f(color * cp.exposure + reflectance * reflected + transmitted, coverage);
}

// ---------------------------------------------------------------------------------------------
// Reflection probe capture (rendering::pass::reflection_probe). Opaque meshes are shaded from the
// probe center into one cube face; group 0 holds the instance buffer (binding 7, shared with the
// forward path) and the face below, groups 1–4 are the usual lighting bindings, group 5 the material.

// Mirrors rendering::pass::reflection_probe::CaptureView.
struct ProbeCapture {
    view_proj: mat4x4<f32>,
    center:    vec4<f32>,
    // The face's basis: the direction through NDC (x, y) is forward + x·right + y·up.
    forward:   vec4<f32>,
    right:     vec4<f32>,
    up:        vec4<f32>,
}

@group(0) @binding(11) var<uniform> probe_capture: ProbeCapture;
// The instance slot of each instance of the current draw (`InstanceList`).
@group(0) @binding(9) var<storage, read> probe_instance_ids: array<u32>;

// Distance stored for the sky, beyond any geometry.
const PROBE_SKY_DISTANCE: f32 = 1.0e4;
// The capture is unexposed Rgba16Float: anything brighter than this (a daylight sun) would become
// inf, and filtering an inf texel produces NaN.
const PROBE_MAX_RADIANCE: f32 = 6.0e4;

@vertex
fn vs_probe(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) tangent: vec4<f32>,
    @builtin(instance_index) instance: u32,
) -> ForwardVertex {
    let slot = probe_instance_ids[instance];
    let m = models[slot];
    let world = m.model * vec4f(position, 1.0);
    var out: ForwardVertex;
    out.pos = probe_capture.view_proj * world;
    out.world_pos = world.xyz / world.w;
    out.world_normal = normalize((m.normal * vec4f(normal, 0.0)).xyz);
    out.color = color;
    out.uv = uv;
    // Left unnormalized: meshes without tangents have zero ones, which fs_probe must detect.
    out.world_tangent = vec4f((m.model * vec4f(tangent.xyz, 0.0)).xyz, tangent.w);
    out.instance = slot;
    return out;
}

// Depth prepass for alpha-tested meshes: the capture is then shaded with an `Equal` depth test, so
// every texel runs `fs_probe` once, however much geometry overlaps.
@fragment
fn fs_probe_depth(in: ForwardVertex) {
    let m = models[in.instance];
    if textureSample(m_albedo, m_sampler, in.uv).a * m.base_color.a < m.material.x {
        discard;
    }
}

@fragment
fn fs_probe(in: ForwardVertex, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    let m = models[in.instance];
    let base = textureSample(m_albedo, m_sampler, in.uv) * m.base_color * vec4f(in.color, 1.0);
    return shade_probe(in, front_facing, m, base);
}

// rgb: unexposed radiance toward the probe center; a: distance to the center. Same material and
// lights as the deferred path; reflections inside the capture use DDGI irradiance, not the probe.
fn shade_probe(in: ForwardVertex, front_facing: bool, m: Model, base: vec4<f32>) -> vec4<f32> {
    let normal_sample = textureSample(m_normal, m_sampler, in.uv).rgb;
    let mr = textureSample(m_metallic_roughness, m_sampler, in.uv);
    let emissive = textureSample(m_emissive, m_sampler, in.uv).rgb * m.emissive.rgb;

    var Ng = normalize(in.world_normal);
    if !front_facing {
        Ng = -Ng;
    }
    // The capture target is float, so a NaN normal from a mesh without tangents would survive
    // (the G-buffer's unorm targets flush it); such meshes keep the geometric normal.
    var N = Ng;
    let tangent_len = length(in.world_tangent.xyz);
    if tangent_len > 1e-6 {
        let T = in.world_tangent.xyz / tangent_len;
        let B = cross(Ng, T) * in.world_tangent.w;
        let tangent_normal = normalize(normal_sample * 2.0 - 1.0);
        N = normalize(T * tangent_normal.x + B * tangent_normal.y + Ng * tangent_normal.z);
    }

    let albedo = select(base.rgb, vec3f(1.0), debug.disable_albedo != 0u);
    let roughness = max(mr.g * m.material.w, MIN_ROUGHNESS);
    let metallic = mr.b * m.material.z;
    let f0 = mix(vec3f(0.04), albedo, metallic);
    let alpha = roughness * roughness;
    let diffuse_color = albedo * (1.0 - metallic) * (1.0 - m.diffuse_transmission.a);
    let transmission = diffuse_transmission(m.diffuse_transmission, metallic);

    // Lights are in the camera's view space; the viewer is the probe center.
    let cam_pos = camera.inv_view[3].xyz;
    let rot = mat3x3<f32>(camera.inv_view[0].xyz, camera.inv_view[1].xyz, camera.inv_view[2].xyz);
    let pos_ws = in.world_pos;
    let pos_vs = (pos_ws - cam_pos) * rot;
    let center = probe_capture.center.xyz;
    let view_dir = normalize(center - pos_ws);
    let surf_vs = Surface(normalize(N * rot), normalize(view_dir * rot), diffuse_color, f0, alpha, transmission);
    let surf_ws = Surface(N, view_dir, diffuse_color, f0, alpha, transmission);

    var color = emissive;
    if debug.disable_direct == 0u {
        color += direct_lighting(surf_vs, surf_ws, pos_vs, pos_ws, in.pos.xy, -pos_vs.z, true, vec3f(0.0), false);
    }
    if debug.disable_indirect == 0u {
        color += diffuse_color * sample_probe_irradiance(pos_ws, N, N, view_dir);
        if any(transmission > vec3f(0.0)) {
            color += transmission * sample_probe_irradiance(pos_ws, -N, -N, view_dir);
        }
        let n_dot_v = max(dot(N, view_dir), 1e-4);
        let R = normalize(mix(reflect(-view_dir, N), N, alpha));
        color += env_brdf_approx(f0, roughness, n_dot_v) * sample_probe_irradiance(pos_ws, N, R, view_dir);
    }
    return vec4f(min(color, vec3f(PROBE_MAX_RADIANCE)), length(pos_ws - center));
}

struct ProbeSkyVertex {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

// A fullscreen triangle at the far plane (standard depth), so it only fills where no mesh was drawn.
@vertex
fn vs_probe_sky(@builtin(vertex_index) vi: u32) -> ProbeSkyVertex {
    let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
    let ndc = uv * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0);
    return ProbeSkyVertex(vec4f(ndc, 1.0, 1.0), ndc);
}

@fragment
fn fs_probe_sky(in: ProbeSkyVertex) -> @location(0) vec4<f32> {
    let dir = normalize(probe_capture.forward.xyz + in.ndc.x * probe_capture.right.xyz + in.ndc.y * probe_capture.up.xyz);
    return vec4f(min(sky_color(dir), vec3f(PROBE_MAX_RADIANCE)), PROBE_SKY_DISTANCE);
}

// ---------------------------------------------------------------------------------------------
// Particles (rendering::pass::particles), simulated by `particle_sim.wgsl`: camera-facing discs
// drawn additively over the finished HDR image. Group 0 holds the G-buffer depth (binding 3, for
// soft fades), the particle slots (12), the emitters (13) and the draw list (14); groups 1–4 are the usual lighting
// bindings. Particles scatter light with a Henyey–Greenstein phase function.

// Mirrors rendering::pass::particle_sim::MAX_EMITTERS (and particle_sim.wgsl).
const MAX_EMITTERS: u32 = 16u;
const PARTICLE_FLAG_LIGHT_ONLY: u32 = 2u;
const PARTICLE_FLAG_UNLIT: u32 = 4u;
// Particles smaller than this (radius, pixels) are drawn at this size with their energy scaled
// down to match, so sub-pixel dust doesn't flicker as it crosses pixel centers.
const MIN_PARTICLE_RADIUS_PX: f32 = 1.0;
// Distance (m) over which a particle fades out as it nears the surface behind it.
const PARTICLE_SOFT_DISTANCE: f32 = 0.05;
// Forward scattering of dust: brighter looking toward the light.
const PARTICLE_PHASE_G: f32 = 0.4;
// Particle::flags bit set by particle_sim.wgsl when a particle settles on a surface.
const PARTICLE_RESTING: u32 = 1u;
// A settled particle stays visible until the surface in front of it is this much (m) nearer.
const PARTICLE_REST_TOLERANCE: f32 = 0.02;

// Mirrors `Particle` in particle_sim.wgsl.
struct Particle {
    pos:  vec3<f32>,
    age:  f32,
    size: f32,
    life: f32,
    rand: f32,
    // Bit 0: resting on a surface (settled).
    flags: u32,
}

// Mirrors rendering::pass::particle_sim::GpuEmitter (and particle_sim.wgsl).
struct GpuEmitter {
    transform: mat4x4<f32>,
    extent:   vec4<f32>,
    shear:    vec4<f32>,
    velocity: vec4<f32>,
    // rgb: scattering albedo (lit) or radiance (unlit); a: opacity.
    color:    vec4<f32>,
    ranges:   vec4<f32>,
    slots:    vec4<u32>,
    step:     vec4<f32>,
    info:     vec4<u32>,
}

struct ParticleParams {
    // Emitter count, slots in use.
    counts:   vec4<u32>,
    // Surface width, height, projection y scale (1 / tan(fovy / 2)).
    screen:   vec4<f32>,
    emitters: array<GpuEmitter, MAX_EMITTERS>,
}

@group(0) @binding(12) var<storage, read> particle_slots: array<Particle>;
@group(0) @binding(13) var<uniform> particle_params: ParticleParams;
// Slots of the live particles, one per instance (particle_sim.wgsl's draw list).
@group(0) @binding(14) var<storage, read> particle_draw_list: array<u32>;

struct ParticleVertex {
    @builtin(position) pos: vec4<f32>,
    // Position on the unit disc.
    @location(0) corner: vec2<f32>,
    @location(1) @interpolate(flat) center: vec3<f32>,
    // Opacity × lifetime fade × the energy kept by the minimum size.
    @location(2) @interpolate(flat) weight: f32,
    @location(3) @interpolate(flat) emitter: u32,
    // View-space distance along −Z.
    @location(4) @interpolate(flat) depth: f32,
    // Settled on a surface: no soft fade, which would hide it.
    @location(5) @interpolate(flat) resting: u32,
}

// Henyey–Greenstein, normalized so an isotropic scatterer gives 1.
fn particle_phase(cos_theta: f32) -> f32 {
    let g = PARTICLE_PHASE_G;
    let denom = 1.0 + g * g - 2.0 * g * cos_theta;
    return (1.0 - g * g) / (denom * sqrt(denom));
}

// Visibility of sky light `L` (world, toward the light) at a point, from the camera's cascades.
fn particle_sky_shadow(pos_vs: vec3<f32>, depth_vs: f32) -> vec3<f32> {
    if debug.disable_shadows != 0u { return vec3f(1.0); }
    let cascade = select_cascade(depth_vs);
    let clip = sp.view_to_shadow[cascade] * vec4f(pos_vs, 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = ndc.xy * vec2f(0.5, -0.5) + 0.5;
    let ref_depth = ndc.z - sp.bias;
    if any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) || ref_depth < 0.0 || ref_depth > 1.0 {
        return vec3f(1.0);
    }
    var lit = 0.0;
    for (var sy: i32 = -1; sy <= 1; sy += 1) {
        for (var sx: i32 = -1; sx <= 1; sx += 1) {
            let off = vec2f(f32(sx), f32(sy)) * sp.inv_shadow_map_size;
            lit += textureSampleCompareLevel(t_shadow, s_shadow, uv + off, i32(cascade), ref_depth);
        }
    }
    return lit / 9.0 * glass_tint(t_glass_tint, uv, cascade, ref_depth);
}

// Illuminance from every light at a particle, each weighted by the phase function toward the
// camera (`view_dir`, world). Same lights, falloffs and shadows as `direct_lighting`; shadow
// lookups offset toward the light instead of along a normal.
fn particle_direct(pos_vs: vec3<f32>, pos_ws: vec3<f32>, view_dir: vec3<f32>, pixel: vec2<f32>) -> vec3<f32> {
    let rot = mat3x3<f32>(camera.inv_view[0].xyz, camera.inv_view[1].xyz, camera.inv_view[2].xyz);
    let V_vs = normalize(view_dir * rot);
    let depth_vs = -pos_vs.z;
    var color = vec3f(0.0);

    let cluster_idx = cluster_index(pixel, depth_vs);
    for (var i = 0u; i < light_grid[cluster_idx]; i++) {
        let light = point_lights[light_indices[cluster_idx * MAX_LIGHTS_PER_CLUSTER + i]];
        let to_light = light.position_vs - pos_vs;
        let dist = length(to_light);
        if dist >= light.radius { continue; }
        let L = to_light / dist;
        let vis = local_shadow_of(light.shadow_slot, pos_ws, rot * L, pixel);
        let falloff = range_attenuation(max(dist, light.source_radius), light.radius);
        color += light.color * vis * (light.intensity * falloff * particle_phase(dot(-L, V_vs)));
    }

    for (var i = 0u; i < light_counts.num_spot; i++) {
        let light = spot_lights[i];
        let to_light = light.position_vs - pos_vs;
        let dist = length(to_light);
        if dist >= light.radius { continue; }
        let L = to_light / dist;
        let cd = dot(-L, normalize(light.direction_vs));
        let t = clamp((cd - light.outer_cos) / max(light.inner_cos - light.outer_cos, 1e-4), 0.0, 1.0);
        let vis = local_shadow_of(light.shadow_slot, pos_ws, rot * L, pixel);
        let falloff = range_attenuation(max(dist, light.source_radius), light.radius);
        color += light.color * vis * (light.intensity * t * t * falloff * particle_phase(dot(-L, V_vs)));
    }

    for (var i = 0u; i < light_counts.num_area; i++) {
        let light = area_lights[i];
        let half_r = length(light.right_vs);
        let half_u = length(light.up_vs);
        if half_r <= 0.0 || half_u <= 0.0 { continue; }
        let r_dir = light.right_vs / half_r;
        let u_dir = light.up_vs / half_u;
        let rel = pos_vs - light.position_vs;
        let closest = light.position_vs
            + r_dir * clamp(dot(rel, r_dir), -half_r, half_r)
            + u_dir * clamp(dot(rel, u_dir), -half_u, half_u);
        let to_light = closest - pos_vs;
        let dist2 = max(dot(to_light, to_light), 1e-4);
        let L = to_light * inverseSqrt(dist2);
        let area = 4.0 * half_r * half_u;
        let facing = area_facing(light, -L, normalize(cross(r_dir, u_dir)));
        if facing <= 0.0 { continue; }
        let vis = local_shadow_of(light.shadow_slot, pos_ws, rot * L, pixel);
        color += light.color * vis
            * (light.intensity * facing * area / (dist2 + area) * particle_phase(dot(-L, V_vs)));
    }

    for (var i = 0u; i < cp.num_sky_lights; i++) {
        let sky = sky_lights[i];
        let L = normalize((camera.inv_view * vec4f(-sky.direction_vs, 0.0)).xyz);
        if L.y < 0.0 { continue; }
        color += sky.color * particle_sky_shadow(pos_vs, depth_vs) * (sky.intensity * particle_phase(dot(-L, view_dir)));
    }

    return color;
}

@vertex
fn vs_particle(@builtin(vertex_index) vi: u32, @builtin(instance_index) instance: u32) -> ParticleVertex {
    var out: ParticleVertex;
    // Outside the clip volume: particles behind the camera draw nothing.
    out.pos = vec4f(2.0, 2.0, 2.0, 1.0);
    let slot = particle_draw_list[instance];
    let p = particle_slots[slot];

    var ei = 0u;
    for (var k = 0u; k < particle_params.counts.x; k++) {
        let s = particle_params.emitters[k].slots;
        if slot >= s.x && slot < s.x + s.y { ei = k; break; }
    }
    let e = particle_params.emitters[ei];

    let clip = camera.view_proj * vec4f(p.pos, 1.0);
    if clip.w <= camera.depth_params.x { return out; }

    let screen = particle_params.screen;
    let radius_px = 0.5 * p.size * screen.z * 0.5 * screen.y / clip.w;
    let drawn_px = max(radius_px, MIN_PARTICLE_RADIUS_PX);
    let t = p.age / p.life;
    let fade = smoothstep(0.0, 0.1, t) * (1.0 - smoothstep(0.85, 1.0, t));

    var corners = array<vec2<f32>, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
        vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
    );
    let corner = corners[vi];
    out.pos = clip + vec4f(corner * (2.0 * drawn_px / screen.xy) * clip.w, 0.0, 0.0);
    out.corner = corner;
    out.center = p.pos;
    out.weight = e.color.a * fade * (radius_px / drawn_px) * (radius_px / drawn_px);
    out.emitter = ei;
    out.depth = clip.w;
    out.resting = p.flags & PARTICLE_RESTING;
    return out;
}

// Additive: rgb is added to the scene, alpha leaves it unchanged.
@fragment
fn fs_particle(in: ParticleVertex) -> @location(0) vec4<f32> {
    let r2 = dot(in.corner, in.corner);
    if r2 >= 1.0 { discard; }
    // A soft disc that integrates to the disc's area: 3 (1 − r²)² over the unit disc gives π.
    let shape = 3.0 * (1.0 - r2) * (1.0 - r2);

    let d = textureLoad(t_depth, vec2<i32>(in.pos.xy), 0);
    let n = camera.depth_params.x;
    let f = camera.depth_params.y;
    let scene = select(n * f / (d * (f - n) + n), 1e30, d <= 0.0);
    var soft = saturate((scene - in.depth) / PARTICLE_SOFT_DISTANCE);
    if in.resting != 0u {
        soft = select(0.0, 1.0, scene - in.depth > -PARTICLE_REST_TOLERANCE);
    }
    if soft <= 0.0 { discard; }

    let e = particle_params.emitters[in.emitter];
    var radiance = e.color.rgb;
    if (e.info.y & PARTICLE_FLAG_UNLIT) == 0u {
        let cam_pos = camera.inv_view[3].xyz;
        let rot = mat3x3<f32>(camera.inv_view[0].xyz, camera.inv_view[1].xyz, camera.inv_view[2].xyz);
        let pos_vs = (in.center - cam_pos) * rot;
        let view_dir = normalize(cam_pos - in.center);
        var light = vec3f(0.0);
        if debug.disable_direct == 0u {
            light += particle_direct(pos_vs, in.center, view_dir, in.pos.xy) / PI;
        }
        if debug.disable_indirect == 0u && (e.info.y & PARTICLE_FLAG_LIGHT_ONLY) == 0u {
            // DDGI is irradiance per hemisphere: a mote sees both, toward and away from the camera.
            light += 0.5 * (sample_probe_irradiance(in.center, view_dir, view_dir, view_dir)
                + sample_probe_irradiance(in.center, -view_dir, -view_dir, view_dir));
        }
        radiance *= light;
    }
    return vec4f(radiance * (cp.exposure * in.weight * shape * soft), 0.0);
}
