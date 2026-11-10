const PROBES_PER_ROW: u32 = 30u;

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
        let w = max(0.0, dot(dir, ray_dir));
        irradiance += ray_radiance[base + r].rgb * w;
        weight_sum += w;
    }
    irradiance /= max(weight_sum, 1e-5);

    let coord = irradiance_atlas_coord(slot, tx, ty);
    let prev  = textureLoad(irr_prev, coord, 0).rgb;
    let out   = mix(irradiance, prev, hysteresis);
    textureStore(irr_next, coord, vec4f(out, 1.0));
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
    let hysteresis = select(probe_params.hysteresis, 0.0, slot_is_fresh(slot, tag));
    if tx          >= probe_params.visibility_texel_size { return; }
    if ty          >= probe_params.visibility_texel_size { return; }

    let n      = probe_params.visibility_texel_size;
    let uv     = (vec2f(f32(tx), f32(ty)) + 0.5) / f32(n);
    let dir    = octahedral_decode(uv);
    let base   = probe_index * probe_params.rays_per_probe;

    var mean_d  = 0.0;
    var mean_d2 = 0.0;
    var weight_sum = 0.0;
    for (var r = 0u; r < probe_params.rays_per_probe; r++) {
        let ray_dir  = ray_direction[base + r].xyz;
        let hit_dist = ray_radiance[base + r].w;
        let w = max(0.0, dot(dir, ray_dir));
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
    textureStore(vis_next, coord, vec4f(out_d, out_d2, tag, 1.0));
}
