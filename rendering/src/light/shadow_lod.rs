//! How much shadow-map resolution each face of a local light needs from the current view: sample
//! points that each know the world size of a screen pixel there, and each face wants one shadow
//! texel per pixel for the points it covers. The points are the visible surfaces, read back from
//! the depth buffer (`pass::depth_readback`) a few frames late, or a grid through the camera
//! frustum while that's missing or the camera has moved on from it.

use std::sync::{Arc, Mutex};

use ultraviolet::Vec3;

use super::{SpotShadow, shadow_atlas::MAX_TILE};
use crate::camera::Camera;

/// Grid of view rays, and sample depths along each.
const RAYS_X: usize = 16;
const RAYS_Y: usize = 9;
const DEPTHS: usize = 16;
/// Nearest sample depth: anything closer is mostly the camera's own near-plane clutter.
const MIN_SAMPLE_DEPTH: f32 = 0.2;
const MAX_SAMPLE_DEPTH: f32 = 100.0;

/// Same order as the shader's face pick: +x, −x, +y, −y, +z, −z.
const FACE_AXES: [(Vec3, Vec3, Vec3); 6] = [
    (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
    (Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
    (Vec3::new(0.0, 1.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
    (Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
    (Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)),
    (Vec3::new(0.0, 0.0, -1.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)),
];

/// A readback's samples are used while the camera stays within this distance (m) of where they
/// were taken and its forward direction within this cosine.
const MAX_READBACK_TRAVEL: f32 = 1.0;
const MIN_READBACK_ALIGNMENT: f32 = 0.9;

/// World points, each with the world size of a screen pixel there.
pub struct ViewSamples {
    points: Vec<(Vec3, f32)>,
}

/// The latest depth readback's samples and the camera they were seen from.
pub struct SurfaceSamples {
    pub samples: Arc<ViewSamples>,
    pub eye: Vec3,
    pub forward: Vec3,
}

pub type SharedSurfaceSamples = Arc<Mutex<Option<SurfaceSamples>>>;

impl SurfaceSamples {
    /// `samples`, unless `camera` has moved or turned too far since they were seen.
    pub fn for_camera(&self, camera: &Camera) -> Option<Arc<ViewSamples>> {
        let forward = (camera.target - camera.eye).normalized();
        let close = (camera.eye - self.eye).mag() <= MAX_READBACK_TRAVEL && forward.dot(self.forward) >= MIN_READBACK_ALIGNMENT;
        close.then(|| Arc::clone(&self.samples))
    }
}

impl ViewSamples {
    pub fn from_points(points: Vec<(Vec3, f32)>) -> Self {
        Self { points }
    }

    /// A grid through the frustum: points in space rather than surfaces, so it overestimates
    /// wherever the view is open.
    pub fn frustum(camera: &Camera, (width, height): (u32, u32)) -> Self {
        let aspect = camera.aspect_or(width as f32 / height.max(1) as f32);
        let tan_y = (camera.fovy * 0.5).tan();
        let tan_x = tan_y * aspect;
        let forward = (camera.target - camera.eye).normalized();
        let right = forward.cross(camera.up).normalized();
        let up = right.cross(forward);
        let near = camera.near.max(MIN_SAMPLE_DEPTH);
        let far = camera.far.min(MAX_SAMPLE_DEPTH).max(near * 2.0);
        // A pixel's world size per metre of depth.
        let pixel_per_depth = 2.0 * tan_y / height.max(1) as f32;

        let mut points = Vec::with_capacity(RAYS_X * RAYS_Y * DEPTHS);
        for k in 0..DEPTHS {
            let depth = near * (far / near).powf(k as f32 / (DEPTHS - 1) as f32);
            for j in 0..RAYS_Y {
                let y = (j as f32 + 0.5) / RAYS_Y as f32 * 2.0 - 1.0;
                for i in 0..RAYS_X {
                    let x = (i as f32 + 0.5) / RAYS_X as f32 * 2.0 - 1.0;
                    let p = camera.eye + (forward + right * (x * tan_x) + up * (y * tan_y)) * depth;
                    points.push((p, depth * pixel_per_depth));
                }
            }
        }
        Self { points }
    }
}

/// A local light's shadow faces as `face_coverage` sees them.
pub struct FaceShape {
    pub origin: Vec3,
    pub range: f32,
    /// `Some`: one face looking along the spot's direction. `None`: a cube.
    pub spot: Option<SpotShadow>,
    /// The emitting side of a one-sided light: faces wholly behind it get no tile.
    pub front: Option<Vec3>,
}

/// Per face, the wanted tile size as a fraction of `MAX_TILE` (above 1 wants more than it gets):
/// the most texels any sample in it needs, so a texel is no larger than a pixel there. 0 marks a
/// face behind a one-sided light, which needs no map; a face no sample falls in gets a small
/// positive value, since off-screen lookups (reflection probe captures) still read it.
pub fn face_coverage(samples: &ViewSamples, shape: &FaceShape) -> [f32; 6] {
    const UNSEEN: f32 = 1e-3;
    let faces = if shape.spot.is_some() { 1 } else { 6 };
    let mut coverage = [0.0; 6];
    for (face, c) in coverage.iter_mut().enumerate().take(faces) {
        *c = if shape.spot.is_none() && shape.front.is_some_and(|n| behind(FACE_AXES[face], n)) { 0.0 } else { UNSEEN };
    }

    let spot_basis = shape.spot.map(|s| {
        let up = if s.direction.y.abs() > 0.99 { Vec3::unit_x() } else { Vec3::unit_y() };
        let right = s.direction.cross(up).normalized();
        (s, right, right.cross(s.direction))
    });
    let range_sq = shape.range * shape.range;
    for &(p, pixel) in &samples.points {
        let d = p - shape.origin;
        if d.mag_sq() > range_sq || shape.front.is_some_and(|n| d.dot(n) <= 0.0) {
            continue;
        }
        // The face holding `p`, the depth along its axis, and the tangent of its half field of view.
        let (face, depth, tan_half) = match spot_basis {
            Some((s, right, up)) => {
                let z = d.dot(s.direction);
                if z <= 0.0 || d.dot(right).abs() > z * s.tan_half_fov || d.dot(up).abs() > z * s.tan_half_fov {
                    continue;
                }
                (0, z, s.tan_half_fov)
            }
            None => {
                let a = d.abs();
                let (face, z) = if a.x >= a.y && a.x >= a.z {
                    (if d.x > 0.0 { 0 } else { 1 }, a.x)
                } else if a.y >= a.z {
                    (if d.y > 0.0 { 2 } else { 3 }, a.y)
                } else {
                    (if d.z > 0.0 { 4 } else { 5 }, a.z)
                };
                (face, z, 1.0)
            }
        };
        if coverage[face] == 0.0 {
            continue;
        }
        let texels = 2.0 * depth * tan_half / pixel;
        coverage[face] = coverage[face].max(texels / MAX_TILE as f32);
    }
    coverage
}

/// Whether a cube face's whole frustum lies behind the plane facing `front`.
fn behind((axis, a, b): (Vec3, Vec3, Vec3), front: Vec3) -> bool {
    [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)].iter().all(|&(s, t)| (axis + a * s + b * t).dot(front) <= 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera(eye: Vec3, target: Vec3) -> Camera {
        Camera { up: Vec3::unit_y(), eye, target, fovy: 1.0, aspect: None, near: 0.1, far: 100.0, exposure: 1.0 }
    }

    #[test]
    fn faces_toward_the_view_need_more() {
        let samples = ViewSamples::frustum(&camera(Vec3::new(0.0, 0.0, 5.0), Vec3::zero()), (1280, 720));
        let shape = FaceShape { origin: Vec3::zero(), range: 10.0, spot: None, front: None };
        let c = face_coverage(&samples, &shape);
        // The camera looks down −z from +z: the +z face holds the view's near samples.
        assert!(c[4] > 1.0, "{c:?}");
        assert!(c[4] > c[5]);
        assert!(c.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn one_sided_lights_skip_the_face_behind() {
        let samples = ViewSamples::frustum(&camera(Vec3::new(3.0, 1.0, 3.0), Vec3::zero()), (1280, 720));
        let shape = FaceShape { origin: Vec3::zero(), range: 10.0, spot: None, front: Some(Vec3::unit_x()) };
        let c = face_coverage(&samples, &shape);
        assert_eq!(c[1], 0.0);
        assert!(c[0] > 0.0 && c[2] > 0.0);
    }

    #[test]
    fn a_light_out_of_view_gets_the_floor() {
        let samples = ViewSamples::frustum(&camera(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0)), (1280, 720));
        let shape = FaceShape { origin: Vec3::new(0.0, 0.0, 50.0), range: 5.0, spot: None, front: None };
        assert!(face_coverage(&samples, &shape).iter().all(|&v| v > 0.0 && v < 0.01));
    }
}
