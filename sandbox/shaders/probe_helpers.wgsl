// Probe grid helpers — no bindings. Concatenate before probe shaders.
// Requires ProbeGridParams to be declared before including this file.

const IRRADIANCE_TEXEL_SIZE: u32 = 8u;
const IRRADIANCE_ATLAS_W: u32    = 300u;
const IRRADIANCE_ATLAS_H: u32    = 290u;
const VISIBILITY_TEXEL_SIZE: u32 = 16u;
const VISIBILITY_ATLAS_W: u32    = 540u;
const VISIBILITY_ATLAS_H: u32    = 522u;

const PROBES_PER_ROW: u32 = 30u;

// Direction → octahedral UV in [0, 1]²
fn octahedral_encode(dir: vec3<f32>) -> vec2<f32> {
    let n = dir / (abs(dir.x) + abs(dir.y) + abs(dir.z));
    var uv = n.xy;
    if n.z < 0.0 {
        uv = (1.0 - abs(n.yx)) * sign(n.xy);
    }
    return uv * 0.5 + 0.5;
}

// Octahedral UV in [0, 1]² → unit direction
fn octahedral_decode(uv: vec2<f32>) -> vec3<f32> {
    let f = uv * 2.0 - 1.0;
    var n = vec3<f32>(f, 1.0 - abs(f.x) - abs(f.y));
    let t = max(-n.z, 0.0);
    n.x += select(t, -t, n.x >= 0.0);
    n.y += select(t, -t, n.y >= 0.0);
    return normalize(n);
}

// Atlas UV for a probe's irradiance patch texel given an octahedral direction UV.
fn probe_irradiance_atlas_uv(probe_index: u32, oct_uv: vec2<f32>) -> vec2<f32> {
    let tile = IRRADIANCE_TEXEL_SIZE + 2u; // 10
    let probe_x = probe_index % PROBES_PER_ROW;
    let probe_y = probe_index / PROBES_PER_ROW;
    let base = vec2<f32>(
        f32(probe_x * tile) + 1.0,
        f32(probe_y * tile) + 1.0,
    );
    let texel = base + oct_uv * f32(IRRADIANCE_TEXEL_SIZE);
    return texel / vec2<f32>(f32(IRRADIANCE_ATLAS_W), f32(IRRADIANCE_ATLAS_H));
}

// Atlas UV for a probe's visibility patch texel given an octahedral direction UV.
fn probe_visibility_atlas_uv(probe_index: u32, oct_uv: vec2<f32>) -> vec2<f32> {
    let tile = VISIBILITY_TEXEL_SIZE + 2u; // 18
    let probe_x = probe_index % PROBES_PER_ROW;
    let probe_y = probe_index / PROBES_PER_ROW;
    let base = vec2<f32>(
        f32(probe_x * tile) + 1.0,
        f32(probe_y * tile) + 1.0,
    );
    let texel = base + oct_uv * f32(VISIBILITY_TEXEL_SIZE);
    return texel / vec2<f32>(f32(VISIBILITY_ATLAS_W), f32(VISIBILITY_ATLAS_H));
}

// World-space position of probe at the given linear index.
fn probe_world_position(probe_index: u32, params: ProbeGridParams) -> vec3<f32> {
    let gx = probe_index % params.grid_dim.x;
    let gy = (probe_index / params.grid_dim.x) % params.grid_dim.y;
    let gz = probe_index / (params.grid_dim.x * params.grid_dim.y);
    return params.grid_origin + vec3<f32>(f32(gx), f32(gy), f32(gz)) * params.probe_spacing;
}
