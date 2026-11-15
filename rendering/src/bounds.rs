use ultraviolet::{Mat4, Vec3, Vec4};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn empty() -> Self {
        Self { min: Vec3::broadcast(f32::MAX), max: Vec3::broadcast(f32::MIN) }
    }

    pub fn from_points(points: impl IntoIterator<Item = Vec3>) -> Self {
        let mut bounds = Self::empty();
        for p in points {
            bounds.extend(p);
        }
        bounds
    }

    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x
    }

    pub fn extend(&mut self, p: Vec3) {
        self.min = self.min.min_by_component(p);
        self.max = self.max.max_by_component(p);
    }

    pub fn union(&self, other: &Self) -> Self {
        Self { min: self.min.min_by_component(other.min), max: self.max.max_by_component(other.max) }
    }

    pub fn corners(&self) -> [Vec3; 8] {
        let (a, b) = (self.min, self.max);
        [
            Vec3::new(a.x, a.y, a.z), Vec3::new(b.x, a.y, a.z),
            Vec3::new(a.x, b.y, a.z), Vec3::new(b.x, b.y, a.z),
            Vec3::new(a.x, a.y, b.z), Vec3::new(b.x, a.y, b.z),
            Vec3::new(a.x, b.y, b.z), Vec3::new(b.x, b.y, b.z),
        ]
    }

    pub fn transformed(&self, transform: Mat4) -> Self {
        if self.is_empty() {
            return *self;
        }
        Self::from_points(self.corners().map(|c| transform.transform_point3(c)))
    }

    /// Whether the box is entirely outside one plane of the clip volume `clip_from_local` maps it
    /// into (x, y in [−w, w], z in [0, w], which covers standard and reversed Z). Conservative: a
    /// box straddling a frustum corner may be kept.
    pub fn outside_clip(&self, clip_from_local: Mat4) -> bool {
        if self.is_empty() {
            return true;
        }
        let clip = self.corners().map(|c| clip_from_local * Vec4::new(c.x, c.y, c.z, 1.0));
        let planes: [fn(Vec4) -> f32; 6] =
            [|p| p.w + p.x, |p| p.w - p.x, |p| p.w + p.y, |p| p.w - p.y, |p| p.z, |p| p.w - p.z];
        planes.iter().any(|plane| clip.iter().all(|&p| plane(p) < 0.0))
    }
}

/// The clip volume of a view-projection as 6 world-space planes, for testing many world boxes
/// against one view (each test is a few multiply-adds, where `Aabb::outside_clip` projects 8
/// corners).
pub struct Frustum {
    planes: [[f32; 4]; 6],
}

impl Frustum {
    /// `clip_from_world` maps into x, y in [−w, w], z in [0, w], as `Aabb::outside_clip` expects.
    pub fn new(clip_from_world: Mat4) -> Self {
        let m: [[f32; 4]; 4] = clip_from_world.cols.map(Into::into);
        let row = |i: usize| [m[0][i], m[1][i], m[2][i], m[3][i]];
        let (x, y, z, w) = (row(0), row(1), row(2), row(3));
        let add = |a: [f32; 4], b: [f32; 4]| std::array::from_fn(|i| a[i] + b[i]);
        let sub = |a: [f32; 4], b: [f32; 4]| std::array::from_fn(|i| a[i] - b[i]);
        Self { planes: [add(w, x), sub(w, x), add(w, y), sub(w, y), z, sub(w, z)] }
    }

    /// Whether `b` is entirely outside one plane: `Aabb::outside_clip` for a box already in world
    /// space.
    pub fn excludes(&self, b: &Aabb) -> bool {
        b.is_empty()
            || self.planes.iter().any(|p| {
                let x = if p[0] >= 0.0 { b.max.x } else { b.min.x };
                let y = if p[1] >= 0.0 { b.max.y } else { b.min.y };
                let z = if p[2] >= 0.0 { b.max.z } else { b.min.z };
                p[0] * x + p[1] * y + p[2] * z + p[3] < 0.0
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ultraviolet::projection::perspective_reversed_z_wgpu_dx_gl;

    #[test]
    fn culls_against_a_reversed_z_frustum() {
        let proj = perspective_reversed_z_wgpu_dx_gl(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
        let view = Mat4::look_at(Vec3::zero(), -Vec3::unit_z(), Vec3::unit_y());
        let clip = proj * view;
        let cube = |center: Vec3| Aabb { min: center - Vec3::broadcast(0.5), max: center + Vec3::broadcast(0.5) };

        assert!(!cube(Vec3::new(0.0, 0.0, -5.0)).outside_clip(clip));
        assert!(cube(Vec3::new(0.0, 0.0, 5.0)).outside_clip(clip), "behind the camera");
        assert!(cube(Vec3::new(20.0, 0.0, -5.0)).outside_clip(clip), "off to the side");
        assert!(cube(Vec3::new(0.0, 0.0, -200.0)).outside_clip(clip), "past the far plane");
        assert!(!cube(Vec3::new(0.0, 0.0, 0.0)).outside_clip(clip), "around the eye");
    }

    #[test]
    fn frustum_planes_agree_with_outside_clip() {
        let proj = perspective_reversed_z_wgpu_dx_gl(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
        let view = Mat4::look_at(Vec3::new(1.0, 2.0, 3.0), Vec3::new(-2.0, 0.5, -4.0), Vec3::unit_y());
        let clip = proj * view;
        let frustum = Frustum::new(clip);
        for i in 0..2000 {
            let f = |k: u32| ((i * 7919 + k * 104729) % 1000) as f32 / 1000.0;
            let center = Vec3::new(f(1) * 80.0 - 40.0, f(2) * 80.0 - 40.0, f(3) * 80.0 - 60.0);
            let half = Vec3::new(f(4), f(5), f(6)) * 3.0;
            let b = Aabb { min: center - half, max: center + half };
            assert_eq!(frustum.excludes(&b), b.outside_clip(clip), "box {b:?}");
        }
        assert!(frustum.excludes(&Aabb::empty()));
    }
}
