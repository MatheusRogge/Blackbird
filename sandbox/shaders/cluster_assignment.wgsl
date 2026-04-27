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

@group(0) @binding(0) var<uniform> params: ClusterParams;
@group(0) @binding(1) var<storage, read> point_lights: array<GpuPointLight>;
@group(0) @binding(2) var<storage, read_write> light_grid: array<u32>;
@group(0) @binding(3) var<storage, read_write> light_indices: array<u32>;

const CLUSTER_X: u32 = 16u;
const CLUSTER_Y: u32 = 9u;
const CLUSTER_Z: u32 = 24u;
const TOTAL_CLUSTERS: u32 = CLUSTER_X * CLUSTER_Y * CLUSTER_Z; // 3456
const MAX_LIGHTS_PER_CLUSTER: u32 = 128u;

fn sphere_aabb_intersect(
    center: vec3<f32>,
    radius: f32,
    aabb_min: vec3<f32>,
    aabb_max: vec3<f32>,
) -> bool {
    let closest = clamp(center, aabb_min, aabb_max);
    let d = center - closest;
    return dot(d, d) <= radius * radius;
}

// Each thread owns one cluster. 64 clusters run in parallel per workgroup.
// No shared memory, no barriers, no atomics — parallelism is between clusters,
// not within them.
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cluster_idx = gid.x;
    if cluster_idx >= TOTAL_CLUSTERS {
        return;
    }

    // Decode (cx, cy, cz) from the flat cluster index.
    let cz  = cluster_idx / (CLUSTER_X * CLUSTER_Y);
    let rem = cluster_idx % (CLUSTER_X * CLUSTER_Y);
    let cy  = rem / CLUSTER_X;
    let cx  = rem % CLUSTER_X;

    let x0 = f32(cx) * params.tile_w;
    let x1 = x0 + params.tile_w;
    let y0 = f32(cy) * params.tile_h;
    let y1 = y0 + params.tile_h;

    // Screen → NDC (Y flipped: screen Y=0 is top, NDC Y=+1 is top)
    let ndc_x0 = x0 / (params.screen_w * 0.5) - 1.0;
    let ndc_x1 = x1 / (params.screen_w * 0.5) - 1.0;
    let ndc_y0 = 1.0 - y1 / (params.screen_h * 0.5);
    let ndc_y1 = 1.0 - y0 / (params.screen_h * 0.5);

    let z_near_k = params.z_near * exp(f32(cz)      / params.log_ratio_recip);
    let z_far_k  = params.z_near * exp(f32(cz + 1u) / params.log_ratio_recip);

    var aabb_min = vec3<f32>(1e38);
    var aabb_max = vec3<f32>(-1e38);

    for (var zi = 0u; zi < 2u; zi++) {
        let depth = select(z_near_k, z_far_k, zi == 1u);
        for (var yi = 0u; yi < 2u; yi++) {
            let ny = select(ndc_y0, ndc_y1, yi == 1u);
            for (var xi = 0u; xi < 2u; xi++) {
                let nx = select(ndc_x0, ndc_x1, xi == 1u);
                let p = vec3<f32>(
                    nx * depth * params.inv_proj_00,
                    ny * depth * params.inv_proj_11,
                    -depth,
                );
                aabb_min = min(aabb_min, p);
                aabb_max = max(aabb_max, p);
            }
        }
    }

    var count = 0u;
    for (var li = 0u; li < params.num_point_lights; li++) {
        let light = point_lights[li];
        if sphere_aabb_intersect(light.position_vs, light.radius, aabb_min, aabb_max) {
            if count < MAX_LIGHTS_PER_CLUSTER {
                light_indices[cluster_idx * MAX_LIGHTS_PER_CLUSTER + count] = li;
                count += 1u;
            }
        }
    }

    light_grid[cluster_idx] = count;
}
