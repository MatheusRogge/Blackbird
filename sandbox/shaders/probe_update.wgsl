const PROBES_PER_ROW: u32 = 30u;
// Sharpness of the lobe each visibility texel averages hit distances over. RTXGI uses 50; with
// 128 rays per probe that showed the texel pattern as blotches on walls near probes.
const VISIBILITY_EXPONENT: f32 = 16.0;

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
    indirect_intensity:     f32,
    cell_origin:            vec3<i32>,
    bounce:                 f32,
}

@group(0) @binding(0) var<storage, read> ray_radiance:   array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> ray_direction:  array<vec4<f32>>;
@group(0) @binding(2) var<uniform>       probe_params:   ProbeGridParams;
@group(0) @binding(3) var irr_prev: texture_2d<f32>;
@group(0) @binding(4) var irr_next: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var vis_prev: texture_2d<f32>;
@group(0) @binding(6) var vis_next: texture_storage_2d<rgba16float, write>;

fn octahedral_decode(uv: vec2<f32>) -> vec3<f32> {
    var f = uv * 2.0 - vec2f(1.0);
    var n = vec3f(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    let t = max(-n.z, 0.0);
    n.x += select(t, -t, n.x >= 0.0);
    n.y += select(t, -t, n.y >= 0.0);
    return normalize(n);
}

// Probes are stored by world cell modulo the grid (mirrors `probe_slot` in lighting.wgsl), so
// a probe keeps its history while the grid slides with the camera. The slot's visibility texels
// carry a tag of the world cell they hold (blue channel); a slot whose tag doesn't match was
// just handed to a probe entering the grid, and starts over instead of blending a probe ~60 m
// away into it.
fn probe_cell(probe_index: u32) -> vec3<i32> {
    let dim = probe_params.grid_dim;
    let local = vec3<u32>(probe_index % dim.x, (probe_index / dim.x) % dim.y, probe_index / (dim.x * dim.y));
    return probe_params.cell_origin + vec3<i32>(local);
}

fn probe_slot(cell: vec3<i32>) -> u32 {
    let dim = vec3<i32>(probe_params.grid_dim);
    let s = (cell % dim + dim) % dim;
    return u32(s.x + dim.x * (s.y + dim.y * s.z));
}

// Which wrap of the grid the cell is in, packed into an integer f16 holds exactly; 0 = empty.
fn probe_tag(cell: vec3<i32>) -> f32 {
    let dim = vec3<i32>(probe_params.grid_dim);
    let wrap = (cell - (cell % dim + dim) % dim) / dim;
    return f32(1 + (wrap.x & 15) + ((wrap.y & 7) << 4u) + ((wrap.z & 15) << 7u));
}

fn slot_is_fresh(slot: u32, tag: f32) -> bool {
    return textureLoad(vis_prev, visibility_atlas_coord(slot, 0u, 0u), 0).b != tag;
}

fn irradiance_atlas_coord(probe_index: u32, tx: u32, ty: u32) -> vec2<i32> {
    let tile = probe_params.irradiance_texel_size + 2u;
    let col  = probe_index % PROBES_PER_ROW;
    let row  = probe_index / PROBES_PER_ROW;
    return vec2<i32>(i32(col * tile + 1u + tx), i32(row * tile + 1u + ty));
}

fn visibility_atlas_coord(probe_index: u32, tx: u32, ty: u32) -> vec2<i32> {
    let tile = probe_params.visibility_texel_size + 2u;
    let col  = probe_index % PROBES_PER_ROW;
    let row  = probe_index / PROBES_PER_ROW;
    return vec2<i32>(i32(col * tile + 1u + tx), i32(row * tile + 1u + ty));
}

const NO_BORDER: vec2<u32> = vec2<u32>(0xffffffffu);

// Tile-space border texels (0 and n + 1 on either axis) that copy interior texel (tx, ty) of an
// n² tile: octahedral wrap mirrors each edge, and corners take the opposite corner. Bilinear
// lookups at the tile edges blend these in.
fn border_copies(tx: u32, ty: u32, n: u32) -> array<vec2<u32>, 3> {
    var out = array<vec2<u32>, 3>(NO_BORDER, NO_BORDER, NO_BORDER);
    var i = 0u;
    if ty == 0u      { out[i] = vec2u(n - tx, 0u);      i++; }
    if ty == n - 1u  { out[i] = vec2u(n - tx, n + 1u);  i++; }
    if tx == 0u      { out[i] = vec2u(0u, n - ty);      i++; }
    if tx == n - 1u  { out[i] = vec2u(n + 1u, n - ty);  i++; }
    if i == 2u {
        out[i] = vec2u(select(n + 1u, 0u, tx == n - 1u), select(n + 1u, 0u, ty == n - 1u));
    }
    return out;
}

fn tile_origin(probe_index: u32, texel_size: u32) -> vec2<i32> {
    let tile = texel_size + 2u;
    return vec2<i32>(i32((probe_index % PROBES_PER_ROW) * tile), i32((probe_index / PROBES_PER_ROW) * tile));
}

// Not NaN or inf, checked on the bits (`x != x` may be folded away by the compiler).
fn is_finite(v: vec3<f32>) -> bool {
    return all((bitcast<vec3<u32>>(v) & vec3<u32>(0x7f800000u)) != vec3<u32>(0x7f800000u));
}

// The offset the probe in `slot` was traced from (see `relocate`).
fn probe_offset(slot: u32) -> vec3<f32> {
    let o = visibility_atlas_coord(slot, 0u, 0u);
    return vec3f(
        textureLoad(vis_prev, o, 0).a,
        textureLoad(vis_prev, o + vec2<i32>(1, 0), 0).a,
        textureLoad(vis_prev, o + vec2<i32>(2, 0), 0).a,
    );
}

// Probe relocation, after RTXGI: probes on a lattice land inside walls or in their planes (any
// wall on a multiple of the spacing), where they see the walls edge-on or from behind and light
// the surfaces around them with what's across the room. Each update moves the probe, from its
// rays (negative distance = a back face, see probe_trace.wgsl):
// - out through the nearest back face, when enough rays hit back faces to be inside geometry;
// - away from the nearest front face, when it is closer than `RELOCATE_MIN_DISTANCE`;
// - otherwise back toward its grid point, as far as that keeps the distance.
// Offsets past `RELOCATE_MAX_OFFSET` are refused, so a probe never leaves its cell.
const RELOCATE_MIN_DISTANCE: f32 = 0.2;
const RELOCATE_MAX_OFFSET: f32 = 0.45;
const RELOCATE_BACKFACE_SHARE: f32 = 0.25;

fn relocate(base: u32, offset: vec3<f32>) -> vec3<f32> {
    let spacing  = probe_params.probe_spacing;
    let min_dist = RELOCATE_MIN_DISTANCE * spacing;

    var backfaces = 0u;
    var closest_back = 1e27;
    var closest_back_dir = vec3f(0.0);
    var closest_front = 1e27;
    var closest_front_dir = vec3f(0.0);
    for (var r = 0u; r < probe_params.rays_per_probe; r++) {
        let d = ray_radiance[base + r].w;
        let ray_dir = ray_direction[base + r].xyz;
        if d < 0.0 {
            backfaces++;
            if -d < closest_back { closest_back = -d; closest_back_dir = ray_dir; }
        } else if d < closest_front {
            closest_front = d;
            closest_front_dir = ray_dir;
        }
    }

    var moved = offset;
    if f32(backfaces) > RELOCATE_BACKFACE_SHARE * f32(probe_params.rays_per_probe) {
        moved = offset + closest_back_dir * (closest_back + min_dist * 0.5);
    } else if closest_front < min_dist {
        moved = offset - closest_front_dir * (min_dist - closest_front);
    } else if length(offset) > 0.0 {
        moved = offset - normalize(offset) * min(closest_front - min_dist, length(offset));
    }
    if length(moved) > RELOCATE_MAX_OFFSET * spacing { return offset; }
    return moved;
}

@compute @workgroup_size(8, 8, 1)
fn update_irradiance(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tx          = gid.x;
    let ty          = gid.y;
    let probe_index = gid.z;

    if probe_index >= probe_params.probe_count      { return; }
    let cell = probe_cell(probe_index);
    let slot = probe_slot(cell);
    let hysteresis = select(probe_params.hysteresis, 0.0, slot_is_fresh(slot, probe_tag(cell)));
    if tx          >= probe_params.irradiance_texel_size { return; }
    if ty          >= probe_params.irradiance_texel_size { return; }

    let n      = probe_params.irradiance_texel_size;
    let uv     = (vec2f(f32(tx), f32(ty)) + 0.5) / f32(n);
    let dir    = octahedral_decode(uv);
    let base   = probe_index * probe_params.rays_per_probe;

    var irradiance = vec3f(0.0);
    var weight_sum = 0.0;
    for (var r = 0u; r < probe_params.rays_per_probe; r++) {
        let ray_dir = ray_direction[base + r].xyz;
        let radiance = ray_radiance[base + r].rgb;
        // Bounces read this atlas back, so one non-finite ray would spread to every probe.
        if !is_finite(radiance) { continue; }
        let w = max(0.0, dot(dir, ray_dir));
        irradiance += radiance * w;
        weight_sum += w;
    }
    irradiance /= max(weight_sum, 1e-5);

    let coord = irradiance_atlas_coord(slot, tx, ty);
    let prev  = textureLoad(irr_prev, coord, 0).rgb;
    let out   = vec4f(mix(irradiance, prev, hysteresis), 1.0);
    textureStore(irr_next, coord, out);
    let borders = border_copies(tx, ty, n);
    for (var i = 0u; i < 3u; i++) {
        if borders[i].x == NO_BORDER.x { break; }
        textureStore(irr_next, tile_origin(slot, n) + vec2<i32>(borders[i]), out);
    }
}

@compute @workgroup_size(16, 16, 1)
fn update_visibility(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tx          = gid.x;
    let ty          = gid.y;
    let probe_index = gid.z;

    if probe_index >= probe_params.probe_count       { return; }
    let cell = probe_cell(probe_index);
    let slot = probe_slot(cell);
    let tag = probe_tag(cell);
    let fresh = slot_is_fresh(slot, tag);
    let hysteresis = select(probe_params.hysteresis, 0.0, fresh);
    if tx          >= probe_params.visibility_texel_size { return; }
    if ty          >= probe_params.visibility_texel_size { return; }

    let n      = probe_params.visibility_texel_size;
    let uv     = (vec2f(f32(tx), f32(ty)) + 0.5) / f32(n);
    let dir    = octahedral_decode(uv);
    let base   = probe_index * probe_params.rays_per_probe;

    // Texels (0..2, 0) also carry the probe's offset in alpha.
    var alpha = 0.0;
    if ty == 0u && tx < 3u {
        let prev_offset = select(probe_offset(slot), vec3f(0.0), fresh);
        alpha = relocate(base, prev_offset)[tx];
    }

    var mean_d  = 0.0;
    var mean_d2 = 0.0;
    var weight_sum = 0.0;
    for (var r = 0u; r < probe_params.rays_per_probe; r++) {
        let ray_dir  = ray_direction[base + r].xyz;
        let hit_dist = abs(ray_radiance[base + r].w);
        // A cosine lobe averaged distances over a hemisphere, so the occlusion test couldn't tell
        // a probe behind a wall from one in front of it.
        let w = pow(max(0.0, dot(dir, ray_dir)), VISIBILITY_EXPONENT);
        mean_d  += hit_dist * w;
        mean_d2 += hit_dist * hit_dist * w;
        weight_sum += w;
    }
    mean_d  /= max(weight_sum, 1e-5);
    mean_d2 /= max(weight_sum, 1e-5);

    let coord    = visibility_atlas_coord(slot, tx, ty);
    let prev     = textureLoad(vis_prev, coord, 0).rg;
    let out_d    = mix(mean_d,  prev.r, hysteresis);
    let out_d2   = mix(mean_d2, prev.g, hysteresis);
    let out      = vec4f(out_d, out_d2, tag, alpha);
    textureStore(vis_next, coord, out);
    let borders = border_copies(tx, ty, n);
    for (var i = 0u; i < 3u; i++) {
        if borders[i].x == NO_BORDER.x { break; }
        textureStore(vis_next, tile_origin(slot, n) + vec2<i32>(borders[i]), out);
    }
}
