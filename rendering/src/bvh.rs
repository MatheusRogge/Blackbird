use bytemuck::{Pod, Zeroable};

pub const MAX_BVH_NODES: u64 = 1_000_000;
pub const MAX_BVH_TRIS: u64 = 500_000;

/// Leaf flag packed into the high bit of `right_or_prim_count`.
const LEAF_FLAG: u32 = 0x8000_0000;
const LEAF_SIZE: u32 = 4;

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
    pub _pad0: u32,
    pub v1: [f32; 3],
    pub _pad1: u32,
    pub v2: [f32; 3],
    pub _pad2: u32,
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
}

pub fn build(triangles: Vec<GpuTriangle>) -> BvhBuild {
    if triangles.is_empty() {
        return BvhBuild { nodes: vec![], triangles: vec![] };
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

    build_node(&triangles, &centroids, &mut prim_indices, &mut nodes, 0, triangles.len() as u32);

    let ordered: Vec<GpuTriangle> =
        prim_indices.iter().map(|&i| triangles[i as usize]).collect();

    BvhBuild { nodes, triangles: ordered }
}

fn build_node(
    tris: &[GpuTriangle],
    centroids: &[[f32; 3]],
    prim_indices: &mut Vec<u32>,
    nodes: &mut Vec<BvhNode>,
    start: u32,
    end: u32,
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

    // Choose split axis: longest span of triangle centroids.
    let mut cen_min = [f32::MAX; 3];
    let mut cen_max = [f32::MIN; 3];
    for i in start..end {
        let c = centroids[prim_indices[i as usize] as usize];
        for k in 0..3 {
            cen_min[k] = cen_min[k].min(c[k]);
            cen_max[k] = cen_max[k].max(c[k]);
        }
    }
    let axis = (0usize..3)
        .max_by(|&a, &b| {
            (cen_max[a] - cen_min[a])
                .partial_cmp(&(cen_max[b] - cen_min[b]))
                .unwrap()
        })
        .unwrap();

    let mid = (start + end) / 2;
    prim_indices[start as usize..end as usize].select_nth_unstable_by(
        (mid - start) as usize,
        |&a, &b| {
            centroids[a as usize][axis]
                .partial_cmp(&centroids[b as usize][axis])
                .unwrap()
        },
    );

    let left = build_node(tris, centroids, prim_indices, nodes, start, mid);
    let right = build_node(tris, centroids, prim_indices, nodes, mid, end);

    nodes[node_idx as usize] = BvhNode {
        aabb_min,
        left_or_first_prim: left,
        aabb_max,
        right_or_prim_count: right,
    };

    node_idx
}
