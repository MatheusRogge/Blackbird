use bytemuck::{Pod, Zeroable};

pub const MAX_BVH_NODES: u64 = 1_000_000;
pub const MAX_BVH_TRIS: u64 = 500_000;

/// Leaf flag packed into the high bit of `right_or_prim_count`.
const LEAF_FLAG: u32 = 0x8000_0000;
const LEAF_SIZE: u32 = 4;
/// Deepest node (the root is 1). `trace_bvh` in `lighting.wgsl` keeps a 32-entry stack, one entry
/// per level above the node it is in.
pub const MAX_BVH_DEPTH: u32 = 32;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BvhNode {
    pub aabb_min: [f32; 3],
    /// Internal: left child index.  Leaf: first primitive index.
    pub left_or_first_prim: u32,
    pub aabb_max: [f32; 3],
    /// Internal: right child index (no high bit).  Leaf: (LEAF_FLAG | prim_count).
    pub right_or_prim_count: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuTriangle {
    pub v0: [f32; 3],
    /// Linear albedo RGB + metallic, packed unorm8 (`unpack4x8unorm`).
    pub albedo_metallic: u32,
    pub v1: [f32; 3],
    /// Roughness, alpha coverage (the fraction of the triangle that isn't cut out) and double-sided
    /// (1 or 0), packed unorm8 in x, y and z. Front faces wind counter-clockwise.
    pub roughness_coverage: u32,
    pub v2: [f32; 3],
    /// Emitted radiance and whether the mesh is lit as an area light (see `pack_emissive`). 0 = none.
    pub emissive: u32,
}

/// Set in a packed emissive when the mesh is also an area light (`light_upload`): GI rays skip
/// its emission, since the light already reaches their hits, while reflection rays show it.
pub const EMISSIVE_AREA_LIGHT: u32 = 0x8000_0000;

/// Packs linear RGB radiance as unorm8 mantissas with a shared 7-bit exponent (bias 64) in bits
/// 24–30, and `EMISSIVE_AREA_LIGHT` in bit 31. Decode with
/// `vec3f(e & 0xff, (e >> 8) & 0xff, (e >> 16) & 0xff) / 255 * exp2(((e >> 24) & 0x7f) - 64)`.
pub fn pack_emissive(rgb: [f32; 3], area_light: bool) -> u32 {
    let max = rgb.iter().copied().fold(0.0f32, f32::max);
    if max <= 0.0 || !max.is_finite() {
        return 0;
    }
    let exponent = max.log2().ceil().clamp(-63.0, 63.0);
    let scale = (-exponent).exp2();
    let [r, g, b] = rgb.map(|c| ((c * scale).clamp(0.0, 1.0) * 255.0).round() as u32);
    let flag = if area_light { EMISSIVE_AREA_LIGHT } else { 0 };
    r | (g << 8) | (b << 16) | (((exponent + 64.0) as u32) << 24) | flag
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unpack(e: u32) -> [f32; 3] {
        let scale = ((((e >> 24) & 0x7f) as f32) - 64.0).exp2();
        [0, 8, 16].map(|shift| ((e >> shift) & 0xff) as f32 / 255.0 * scale)
    }

    #[test]
    fn emissive_round_trips_hdr_values() {
        for area_light in [false, true] {
            let packed = pack_emissive([6.0, 3.3, 1.2], area_light);
            let [r, g, b] = unpack(packed);
            assert!((r - 6.0).abs() < 0.05 && (g - 3.3).abs() < 0.05 && (b - 1.2).abs() < 0.05, "{r} {g} {b}");
            assert_eq!(packed & EMISSIVE_AREA_LIGHT != 0, area_light);
        }
    }

    #[test]
    fn black_packs_to_zero() {
        assert_eq!(pack_emissive([0.0; 3], true), 0);
    }

    fn triangle(at: [f32; 3], size: f32) -> GpuTriangle {
        GpuTriangle {
            v0: at,
            v1: [at[0] + size, at[1], at[2]],
            v2: [at[0], at[1] + size, at[2]],
            ..GpuTriangle::zeroed()
        }
    }

    /// Checks the tree's invariants and returns its depth.
    fn check(build: &BvhBuild, triangle_count: usize) -> u32 {
        let contains = |outer: &BvhNode, min: [f32; 3], max: [f32; 3]| {
            (0..3).all(|k| outer.aabb_min[k] <= min[k] && max[k] <= outer.aabb_max[k])
        };
        let mut seen = vec![false; triangle_count];
        let mut depth = 0;
        let mut stack = vec![(0u32, 1u32)];
        while let Some((index, d)) = stack.pop() {
            depth = depth.max(d);
            let node = &build.nodes[index as usize];
            if node.right_or_prim_count & LEAF_FLAG != 0 {
                let count = node.right_or_prim_count & !LEAF_FLAG;
                assert!(count <= LEAF_SIZE);
                for i in node.left_or_first_prim..node.left_or_first_prim + count {
                    let t = &build.triangles[i as usize];
                    for v in [t.v0, t.v1, t.v2] {
                        assert!(contains(node, v, v));
                    }
                    let original = build.order[i as usize] as usize;
                    assert!(!seen[original], "triangle {original} in two leaves");
                    seen[original] = true;
                }
            } else {
                for child in [node.left_or_first_prim, node.right_or_prim_count] {
                    let c = &build.nodes[child as usize];
                    assert!(contains(node, c.aabb_min, c.aabb_max));
                    stack.push((child, d + 1));
                }
            }
        }
        assert!(seen.iter().all(|&s| s), "every triangle is in a leaf");
        depth
    }

    #[test]
    fn every_triangle_lands_in_one_leaf_inside_its_boxes() {
        // A large floor and a cluster of small detail, as in a street scene.
        let mut tris = vec![triangle([-50.0, 0.0, -50.0], 100.0), triangle([-50.0, 0.0, 50.0], 100.0)];
        for i in 0..500 {
            let f = |k: u32| ((i * 7919 + k * 104729) % 1000) as f32 / 100.0;
            tris.push(triangle([f(1), f(2), f(3)], 0.1));
        }
        let count = tris.len();
        let build = build(tris);
        check(&build, count);
    }

    #[test]
    fn depth_stays_within_the_traversal_stack() {
        // Spacing that doubles each time: the SAH's best split peels one triangle off per level.
        let tris: Vec<GpuTriangle> = (0..2000).map(|i| triangle([1.01f32.powi(i * 8), 0.0, 0.0], 0.01)).collect();
        let build = build(tris);
        assert!(check(&build, 2000) <= MAX_BVH_DEPTH);
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BvhInfo {
    pub node_count: u32,
    pub tri_count: u32,
    pub enabled: u32,
    pub _pad: u32,
}

pub struct BvhBuild {
    pub nodes: Vec<BvhNode>,
    pub triangles: Vec<GpuTriangle>,
    /// For each of `triangles`, its index in the input.
    pub order: Vec<u32>,
}

pub fn build(triangles: Vec<GpuTriangle>) -> BvhBuild {
    if triangles.is_empty() {
        return BvhBuild { nodes: vec![], triangles: vec![], order: vec![] };
    }

    let centroids: Vec<[f32; 3]> = triangles
        .iter()
        .map(|t| {
            [
                (t.v0[0] + t.v1[0] + t.v2[0]) / 3.0,
                (t.v0[1] + t.v1[1] + t.v2[1]) / 3.0,
                (t.v0[2] + t.v1[2] + t.v2[2]) / 3.0,
            ]
        })
        .collect();

    let mut prim_indices: Vec<u32> = (0..triangles.len() as u32).collect();
    let mut nodes: Vec<BvhNode> = Vec::with_capacity(triangles.len() * 2);

    build_node(&triangles, &centroids, &mut prim_indices, &mut nodes, 0, triangles.len() as u32, 1);

    let ordered: Vec<GpuTriangle> =
        prim_indices.iter().map(|&i| triangles[i as usize]).collect();

    BvhBuild { nodes, triangles: ordered, order: prim_indices }
}

fn build_node(
    tris: &[GpuTriangle],
    centroids: &[[f32; 3]],
    prim_indices: &mut Vec<u32>,
    nodes: &mut Vec<BvhNode>,
    start: u32,
    end: u32,
    depth: u32,
) -> u32 {
    let node_idx = nodes.len() as u32;
    nodes.push(BvhNode::zeroed()); // placeholder — patched below

    let count = end - start;

    let mut aabb_min = [f32::MAX; 3];
    let mut aabb_max = [f32::MIN; 3];
    for i in start..end {
        let t = &tris[prim_indices[i as usize] as usize];
        for v in [t.v0, t.v1, t.v2] {
            for k in 0..3 {
                aabb_min[k] = aabb_min[k].min(v[k]);
                aabb_max[k] = aabb_max[k].max(v[k]);
            }
        }
    }

    if count <= LEAF_SIZE {
        nodes[node_idx as usize] = BvhNode {
            aabb_min,
            left_or_first_prim: start,
            aabb_max,
            right_or_prim_count: LEAF_FLAG | count,
        };
        return node_idx;
    }

    // Levels a balanced subtree of `n` triangles needs below its root.
    let levels = |n: u32| n.div_ceil(LEAF_SIZE).next_power_of_two().trailing_zeros();
    let prims = &mut prim_indices[start as usize..end as usize];
    let left_count = sah_split(tris, centroids, prims)
        .filter(|&left| depth + 1 + levels(left.max(count - left)) <= MAX_BVH_DEPTH)
        .unwrap_or_else(|| median_split(centroids, prims));
    let mid = start + left_count;

    let left = build_node(tris, centroids, prim_indices, nodes, start, mid, depth + 1);
    let right = build_node(tris, centroids, prim_indices, nodes, mid, end, depth + 1);

    nodes[node_idx as usize] = BvhNode {
        aabb_min,
        left_or_first_prim: left,
        aabb_max,
        right_or_prim_count: right,
    };

    node_idx
}

/// Splits `prims` in half along the longest axis of their centroids; returns the left half's
/// size. Keeps the tree within `MAX_BVH_DEPTH` where the SAH's split would go too deep.
fn median_split(centroids: &[[f32; 3]], prims: &mut [u32]) -> u32 {
    let mut cen_min = [f32::MAX; 3];
    let mut cen_max = [f32::MIN; 3];
    for &p in prims.iter() {
        for (k, &c) in centroids[p as usize].iter().enumerate() {
            cen_min[k] = cen_min[k].min(c);
            cen_max[k] = cen_max[k].max(c);
        }
    }
    let axis = (0..3).max_by(|&a, &b| (cen_max[a] - cen_min[a]).total_cmp(&(cen_max[b] - cen_min[b]))).unwrap_or(0);
    let mid = prims.len() / 2;
    prims.select_nth_unstable_by(mid, |&a, &b| centroids[a as usize][axis].total_cmp(&centroids[b as usize][axis]));
    mid as u32
}

/// Bins per axis for `sah_split`.
const SAH_BINS: usize = 16;

fn area(min: [f32; 3], max: [f32; 3]) -> f32 {
    let d = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    if d[0] < 0.0 {
        return 0.0;
    }
    d[0] * d[1] + d[1] * d[2] + d[2] * d[0]
}

/// Partitions `prims` at the binned surface area heuristic's best split over all three axes and
/// returns how many went left, or `None` when no split separates them (the caller splits at the
/// median).
fn sah_split(tris: &[GpuTriangle], centroids: &[[f32; 3]], prims: &mut [u32]) -> Option<u32> {
    let mut cen_min = [f32::MAX; 3];
    let mut cen_max = [f32::MIN; 3];
    for &p in prims.iter() {
        let c = centroids[p as usize];
        for k in 0..3 {
            cen_min[k] = cen_min[k].min(c[k]);
            cen_max[k] = cen_max[k].max(c[k]);
        }
    }
    let bin_of = |axis: usize, p: u32| -> usize {
        let extent = cen_max[axis] - cen_min[axis];
        (((centroids[p as usize][axis] - cen_min[axis]) / extent * SAH_BINS as f32) as usize).min(SAH_BINS - 1)
    };
    let mut best: Option<(f32, usize, usize)> = None;
    for axis in 0..3 {
        if cen_max[axis] - cen_min[axis] <= 1e-6 {
            continue;
        }
        let mut bins = [([f32::MAX; 3], [f32::MIN; 3], 0u32); SAH_BINS];
        for &p in prims.iter() {
            let bin = &mut bins[bin_of(axis, p)];
            let t = &tris[p as usize];
            for v in [t.v0, t.v1, t.v2] {
                for (k, &x) in v.iter().enumerate() {
                    bin.0[k] = bin.0[k].min(x);
                    bin.1[k] = bin.1[k].max(x);
                }
            }
            bin.2 += 1;
        }
        // Cost of splitting after bin i: area × count on each side.
        let mut right_cost = [0f32; SAH_BINS];
        let (mut rmin, mut rmax, mut rcount) = ([f32::MAX; 3], [f32::MIN; 3], 0u32);
        for i in (1..SAH_BINS).rev() {
            for k in 0..3 {
                rmin[k] = rmin[k].min(bins[i].0[k]);
                rmax[k] = rmax[k].max(bins[i].1[k]);
            }
            rcount += bins[i].2;
            right_cost[i - 1] = area(rmin, rmax) * rcount as f32;
        }
        let (mut lmin, mut lmax, mut lcount) = ([f32::MAX; 3], [f32::MIN; 3], 0u32);
        for i in 0..SAH_BINS - 1 {
            for k in 0..3 {
                lmin[k] = lmin[k].min(bins[i].0[k]);
                lmax[k] = lmax[k].max(bins[i].1[k]);
            }
            lcount += bins[i].2;
            if lcount == 0 || lcount as usize == prims.len() {
                continue;
            }
            let cost = area(lmin, lmax) * lcount as f32 + right_cost[i];
            if best.is_none_or(|(c, ..)| cost < c) {
                best = Some((cost, axis, i));
            }
        }
    }
    let (_, axis, split) = best?;
    let mut left = 0;
    for i in 0..prims.len() {
        if bin_of(axis, prims[i]) <= split {
            prims.swap(i, left);
            left += 1;
        }
    }
    Some(left as u32)
}
