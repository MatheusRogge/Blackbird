use ultraviolet::{Mat4, Vec3};

use crate::mesh::Mesh;

/// Below this area-weighted normal coherence (|Σ n·A| / ΣA) a mesh is not flat enough to stand in
/// for a rectangle light.
const MIN_PLANARITY: f32 = 0.95;
/// Triangle edges tried as rectangle axes. Emitters are small meshes, so this is rarely reached.
const MAX_EDGE_CANDIDATES: usize = 192;

/// A rectangle fitted to a flat emissive mesh, in the mesh's local space. `right` and `up` are
/// half-extent vectors and `right × up` points out of the emitting side, as in `AreaLight`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmissiveRect {
    pub center: Vec3,
    pub right: Vec3,
    pub up: Vec3,
}

impl EmissiveRect {
    pub fn transformed(&self, transform: Mat4) -> Self {
        Self {
            center: transform.transform_point3(self.center),
            right: transform.transform_vec3(self.right),
            up: transform.transform_vec3(self.up),
        }
    }

    pub fn area(&self) -> f32 {
        4.0 * self.right.mag() * self.up.mag()
    }
}

/// Fits the oriented rectangle covering `mesh`, or `None` if its triangles aren't (nearly)
/// coplanar or have no area.
pub fn fit(mesh: &Mesh) -> Option<EmissiveRect> {
    let position = |i: u32| mesh.vertices.get(i as usize).map(|v| Vec3::from(v.position));
    let mut weighted_normal = Vec3::zero();
    let mut total_area = 0.0;
    let mut used = Vec::new();
    for &[i0, i1, i2] in mesh.indices.as_chunks::<3>().0 {
        let (Some(a), Some(b), Some(c)) = (position(i0), position(i1), position(i2)) else {
            continue;
        };
        let cross: Vec3 = (b - a).cross(c - a);
        weighted_normal += cross * 0.5;
        total_area += cross.mag() * 0.5;
        used.extend([a, b, c]);
    }
    if total_area <= 0.0 || weighted_normal.mag() < MIN_PLANARITY * total_area {
        return None;
    }
    let normal = weighted_normal.normalized();

    let centroid = used.iter().fold(Vec3::zero(), |acc, &p| acc + p) / used.len() as f32;
    let helper = if normal.x.abs() < 0.9 { Vec3::unit_x() } else { Vec3::unit_y() };
    let t = normal.cross(helper).normalized();
    let b = normal.cross(t);

    // Smallest rectangle aligned to a triangle edge (exact for quads and boxes), falling back to the
    // principal axis of the points.
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for &p in &used {
        let d = p - centroid;
        let (x, y) = (d.dot(t), d.dot(b));
        sxx += x * x;
        sxy += x * y;
        syy += y * y;
    }
    let angle = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let edges = used.as_chunks::<3>().0.iter().take(MAX_EDGE_CANDIDATES / 3).flat_map(|&[a, b, c]| [b - a, c - b, a - c]);
    let candidates = edges
        .map(|e| e - normal * e.dot(normal))
        .filter(|e| e.mag_sq() > 0.0)
        .map(|e| e.normalized())
        .chain([t * angle.cos() + b * angle.sin()]);

    let bounds = |axis_u: Vec3| {
        let axis_v = normal.cross(axis_u);
        let (mut min_u, mut max_u, mut min_v, mut max_v) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for &p in &used {
            let d = p - centroid;
            let (u, v) = (d.dot(axis_u), d.dot(axis_v));
            min_u = min_u.min(u);
            max_u = max_u.max(u);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        (axis_u, axis_v, min_u, max_u, min_v, max_v)
    };
    let area = |&(_, _, min_u, max_u, min_v, max_v): &(Vec3, Vec3, f32, f32, f32, f32)| (max_u - min_u) * (max_v - min_v);
    let (axis_u, axis_v, min_u, max_u, min_v, max_v) =
        candidates.map(bounds).min_by(|x, y| area(x).total_cmp(&area(y)))?;

    let (half_u, half_v) = (0.5 * (max_u - min_u), 0.5 * (max_v - min_v));
    if half_u <= 0.0 || half_v <= 0.0 {
        return None;
    }
    Some(EmissiveRect {
        center: centroid + axis_u * (0.5 * (max_u + min_u)) + axis_v * (0.5 * (max_v + min_v)),
        right: axis_u * half_u,
        up: axis_v * half_v,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Vertex;

    fn mesh(positions: &[[f32; 3]], indices: Vec<u32>) -> Mesh {
        let vertices = positions
            .iter()
            .map(|&position| Vertex { position, normal: [0.0; 3], color: [1.0; 3], uv: [0.0; 2], tangent: [1.0, 0.0, 0.0, 1.0] })
            .collect();
        Mesh::new(vertices, indices, None, None)
    }

    #[test]
    fn fits_a_quad_facing_its_winding() {
        // 0.4 × 0.2 quad in the XY plane at z = 1, wound counter-clockwise seen from +Z.
        let quad = mesh(
            &[[-0.2, -0.1, 1.0], [0.2, -0.1, 1.0], [0.2, 0.1, 1.0], [-0.2, 0.1, 1.0]],
            vec![0, 1, 2, 0, 2, 3],
        );
        let rect = fit(&quad).expect("a quad is planar");
        assert!((rect.center - Vec3::new(0.0, 0.0, 1.0)).mag() < 1e-5);
        assert!((rect.area() - 0.08).abs() < 1e-5, "{rect:?}");
        assert!(rect.right.cross(rect.up).normalized().dot(Vec3::unit_z()) > 0.999);
    }

    #[test]
    fn fits_a_rotated_square_tightly() {
        let (c, s) = (0.5f32.cos(), 0.5f32.sin());
        let corners = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0f32]].map(|[x, y]| [x * c - y * s, x * s + y * c, 0.0]);
        let rect = fit(&mesh(&corners, vec![0, 1, 2, 0, 2, 3])).expect("a square is planar");
        assert!((rect.area() - 4.0).abs() < 1e-4, "{rect:?}");
    }

    #[test]
    fn rejects_opposite_faces() {
        let positions = [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        ];
        // Two opposite faces with outward winding: their normals cancel.
        let slab = mesh(&positions, vec![0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7]);
        assert_eq!(fit(&slab), None);
    }
}
