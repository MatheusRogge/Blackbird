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
}
