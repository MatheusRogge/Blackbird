struct BvhNode {
    aabb_min:             vec3<f32>,
    left_or_first_prim:   u32,
    aabb_max:             vec3<f32>,
    right_or_prim_count:  u32,
}

struct BvhInfo {
    node_count: u32,
    tri_count:  u32,
    enabled:    u32,
    _pad:       u32,
}

struct CameraUniforms {
    view_proj:     mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    inv_view:      mat4x4<f32>,
}

struct ProbeGridParams {
    grid_dim:              vec3<u32>,
    probe_count:           u32,
    grid_origin:           vec3<f32>,
    max_ray_distance:      f32,
    probe_spacing:         f32,
    rays_per_probe:        u32,
    irradiance_texel_size: u32,
    visibility_texel_size: u32,
    hysteresis:            f32,
    normal_bias:           f32,
    view_bias:             f32,
    _pad:                  f32,
}

struct OverlayParams {
    bvh_leaves_only: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> bvh_nodes:    array<BvhNode>;
@group(0) @binding(1) var<uniform>       bvh_info:     BvhInfo;
@group(0) @binding(2) var<uniform>       camera:       CameraUniforms;
@group(0) @binding(3) var<uniform>       probe_params: ProbeGridParams;
@group(0) @binding(4) var<uniform>       overlay:      OverlayParams;

// 12 box edges as pairs of corner indices.
// Corner bit encoding: bit0=X, bit1=Y, bit2=Z (0=min, 1=max).
const EDGE_CORNERS = array<u32, 24>(
    0u, 1u,   2u, 3u,   4u, 5u,   6u, 7u,   // 4 X-axis edges
    0u, 2u,   1u, 3u,   4u, 6u,   5u, 7u,   // 4 Y-axis edges
    0u, 4u,   1u, 5u,   2u, 6u,   3u, 7u,   // 4 Z-axis edges
);

// ── Editor grid ───────────────────────────────────────────────────────────────

fn grid_line(xz: vec2<f32>, cell_size: f32) -> f32 {
    let coord = xz / cell_size;
    let d = max(fwidth(coord), vec2<f32>(0.0001));
    let g = abs(fract(coord - 0.5) - 0.5) / d;
    return 1.0 - min(min(g.x, g.y), 1.0);
}

struct GridVertOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)       ndc: vec2<f32>,
}

@vertex
fn vs_grid(@builtin(vertex_index) vi: u32) -> GridVertOut {
    let uv  = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    let ndc = uv * 2.0 - 1.0;
    var out: GridVertOut;
    out.pos = vec4<f32>(ndc, 0.0, 1.0);
    out.ndc = ndc;
    return out;
}

@fragment
fn fs_grid(in: GridVertOut) -> @location(0) vec4<f32> {
    let cam_pos = camera.inv_view[3].xyz;
    let p_far_h = camera.inv_view_proj * vec4<f32>(in.ndc, 0.0, 1.0);
    let p_far   = p_far_h.xyz / p_far_h.w;
    let ray_dir = normalize(p_far - cam_pos);

    if abs(ray_dir.y) < 1e-5 { discard; }
    let t = -cam_pos.y / ray_dir.y;
    if t < 0.0 { discard; }

    let world = cam_pos + t * ray_dir;
    let dist  = t;

    let a1   = grid_line(world.xz, 1.0);
    let a10  = grid_line(world.xz, 10.0);
    let fade = 1.0 - smoothstep(40.0, 100.0, dist);
    let alpha = max(a1 * 0.35, a10 * 0.75) * fade;

    if alpha < 0.01 { discard; }

    return vec4<f32>(0.55, 0.55, 0.55, alpha);
}

// ── BVH AABB lines ────────────────────────────────────────────────────────────

@vertex
fn vs_bvh(
    @builtin(vertex_index)   vi: u32,
    @builtin(instance_index) ii: u32,
) -> @builtin(position) vec4<f32> {
    // Emit degenerate geometry (behind near plane → clipped) for skipped nodes.
    let behind = vec4<f32>(0.0, 0.0, -1.0, 1.0);

    if bvh_info.enabled == 0u {
        return behind;
    }

    let node = bvh_nodes[ii];
    let is_leaf = (node.right_or_prim_count & 0x80000000u) != 0u;

    if overlay.bvh_leaves_only == 1u && !is_leaf {
        return behind;
    }

    let c = EDGE_CORNERS[vi];
    let corner = vec3<f32>(
        select(node.aabb_min.x, node.aabb_max.x, (c & 1u) != 0u),
        select(node.aabb_min.y, node.aabb_max.y, (c & 2u) != 0u),
        select(node.aabb_min.z, node.aabb_max.z, (c & 4u) != 0u),
    );
    return camera.view_proj * vec4<f32>(corner, 1.0);
}

@fragment
fn fs_bvh() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.5, 0.0, 1.0); // orange
}

// ── DDGI probe billboards ─────────────────────────────────────────────────────

const QUAD_OFFSETS = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
);

struct ProbeVertOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
}

@vertex
fn vs_probe(
    @builtin(vertex_index)   vi: u32,
    @builtin(instance_index) ii: u32,
) -> ProbeVertOut {
    let dim = probe_params.grid_dim;
    let ix = ii % dim.x;
    let iy = (ii / dim.x) % dim.y;
    let iz = ii / (dim.x * dim.y);
    let probe_pos = probe_params.grid_origin
        + vec3<f32>(f32(ix), f32(iy), f32(iz)) * probe_params.probe_spacing;

    // Camera right and up from the inverse-view matrix columns.
    let right = camera.inv_view[0].xyz;
    let up    = camera.inv_view[1].xyz;

    let size   = 0.15; // world-space radius
    let offset = QUAD_OFFSETS[vi];
    let world_pos = probe_pos + right * offset.x * size + up * offset.y * size;

    var out: ProbeVertOut;
    out.pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv  = offset;
    return out;
}

@fragment
fn fs_probe(in: ProbeVertOut) -> @location(0) vec4<f32> {
    let d = length(in.uv);
    if d > 1.0 { discard; }
    let shade = 1.0 - d * 0.4;
    return vec4<f32>(0.0, shade * 0.9, shade, 1.0); // cyan-green
}
