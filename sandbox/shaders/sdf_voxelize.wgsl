struct SdfParams {
    world_min: vec3<f32>,
    voxel_size: f32,
    world_max: vec3<f32>,
    triangle_count: u32,
    resolution: vec3<u32>,
    _pad: u32,
}

@group(0) @binding(0) var sdf_volume: texture_storage_3d<r32float, write>;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<uniform> params: SdfParams;

fn point_triangle_distance(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 { return length(ap); }

    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 { return length(bp); }

    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 { return length(cp); }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return length(ap - v * ab);
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return length(ap - w * ac);
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return length(p - b - w * (c - b));
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    return length(ap - v * ab - w * ac);
}

fn load_position(vertex_index: u32) -> vec3<f32> {
    let base = vertex_index * 3u;
    return vec3<f32>(positions[base], positions[base + 1u], positions[base + 2u]);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if any(gid >= params.resolution) { return; }

    let voxel_center = params.world_min + (vec3<f32>(gid) + 0.5) * params.voxel_size;

    var min_dist = 1e30;
    var sign = 1.0;

    for (var tri = 0u; tri < params.triangle_count; tri = tri + 1u) {
        let i0 = indices[tri * 3u];
        let i1 = indices[tri * 3u + 1u];
        let i2 = indices[tri * 3u + 2u];
        let v0 = load_position(i0);
        let v1 = load_position(i1);
        let v2 = load_position(i2);

        let d = point_triangle_distance(voxel_center, v0, v1, v2);
        if d < min_dist {
            min_dist = d;
            let normal = normalize(cross(v1 - v0, v2 - v0));
            sign = select(-1.0, 1.0, dot(voxel_center - v0, normal) >= 0.0);
        }
    }

    textureStore(sdf_volume, vec3<i32>(gid), vec4<f32>(sign * min_dist, 0.0, 0.0, 1.0));
}
