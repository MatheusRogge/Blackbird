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
    _pad:                   f32,
}

@group(0) @binding(0) var<storage, read> ray_radiance:      array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> ray_direction:     array<vec4<f32>>;
@group(0) @binding(2) var<uniform>       probe_params:      ProbeGridParams;
@group(0) @binding(3) var irradiance_atlas: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var visibility_atlas: texture_storage_2d<rgba16float, write>;

fn octahedral_decode(uv: vec2<f32>) -> vec3<f32> {
    var f = uv * 2.0 - vec2f(1.0);
    var n = vec3f(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    let t = max(-n.z, 0.0);
    n.x += select(t, -t, n.x >= 0.0);
    n.y += select(t, -t, n.y >= 0.0);
    return normalize(n);
}

fn irradiance_atlas_coord(probe_index: u32, tx: u32, ty: u32) -> vec2<i32> {
    let tile  = probe_params.irradiance_texel_size + 2u;
    let col   = probe_index % PROBES_PER_ROW;
    let row   = probe_index / PROBES_PER_ROW;
    return vec2<i32>(
        i32(col * tile + 1u + tx),
        i32(row * tile + 1u + ty),
    );
}

fn visibility_atlas_coord(probe_index: u32, tx: u32, ty: u32) -> vec2<i32> {
    let tile  = probe_params.visibility_texel_size + 2u;
    let col   = probe_index % PROBES_PER_ROW;
    let row   = probe_index / PROBES_PER_ROW;
    return vec2<i32>(
        i32(col * tile + 1u + tx),
        i32(row * tile + 1u + ty),
    );
}

@compute @workgroup_size(8, 8, 1)
fn update_irradiance(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tx            = gid.x;
    let ty            = gid.y;
    let probe_index   = gid.z;

    if probe_index >= probe_params.probe_count { return; }
    if tx >= probe_params.irradiance_texel_size { return; }
    if ty >= probe_params.irradiance_texel_size { return; }

    let n = probe_params.irradiance_texel_size;
    let uv = (vec2f(f32(tx), f32(ty)) + vec2f(0.5)) / f32(n);
    let texel_dir = octahedral_decode(uv);

    let base = probe_index * probe_params.rays_per_probe;
    var irradiance = vec3f(0.0);
    var weight_sum = 0.0;

    for (var r = 0u; r < probe_params.rays_per_probe; r++) {
        let ray_dir = ray_direction[base + r].xyz;
        let w = max(0.0, dot(texel_dir, ray_dir));
        irradiance += ray_radiance[base + r].rgb * w;
        weight_sum += w;
    }

    irradiance /= max(weight_sum, 1e-5);

    let coord = irradiance_atlas_coord(probe_index, tx, ty);
    textureStore(irradiance_atlas, coord, vec4f(irradiance, 1.0));
}

@compute @workgroup_size(16, 16, 1)
fn update_visibility(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tx            = gid.x;
    let ty            = gid.y;
    let probe_index   = gid.z;

    if probe_index >= probe_params.probe_count { return; }
    if tx >= probe_params.visibility_texel_size { return; }
    if ty >= probe_params.visibility_texel_size { return; }

    let n = probe_params.visibility_texel_size;
    let uv = (vec2f(f32(tx), f32(ty)) + vec2f(0.5)) / f32(n);
    let texel_dir = octahedral_decode(uv);

    let base = probe_index * probe_params.rays_per_probe;
    var mean_d  = 0.0;
    var mean_d2 = 0.0;
    var weight_sum = 0.0;

    for (var r = 0u; r < probe_params.rays_per_probe; r++) {
        let ray_dir  = ray_direction[base + r].xyz;
        let hit_dist = ray_radiance[base + r].w;
        let w        = max(0.0, dot(texel_dir, ray_dir));
        mean_d  += hit_dist * w;
        mean_d2 += hit_dist * hit_dist * w;
        weight_sum += w;
    }

    mean_d  /= max(weight_sum, 1e-5);
    mean_d2 /= max(weight_sum, 1e-5);

    let coord = visibility_atlas_coord(probe_index, tx, ty);
    textureStore(visibility_atlas, coord, vec4f(mean_d, mean_d2, 0.0, 1.0));
}
