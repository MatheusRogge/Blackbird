use engine_core::entity::{Controllable, Entity};
use ultraviolet::{Mat4, Vec3, projection};

pub struct Camera {
    pub up: Vec3,
    pub eye: Vec3,
    pub target: Vec3,
    pub fovy: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn view_proj_matrix(&self) -> Mat4 {
        let view = Mat4::look_at(self.eye, self.target, self.up);

        // Reversed-Z for better depth precision — near/far swapped,
        // clip depth maps to [1, 0] instead of [0, 1].
        let proj = projection::perspective_reversed_z_wgpu_dx_gl(
            self.fovy,
            self.aspect,
            self.near,
            self.far,
        );

        proj * view
    }
}

impl Entity for Camera {}

impl Controllable for Camera {
    fn translate(&mut self, delta: Vec3) {
        self.eye += delta;
        self.target += delta;
    }

    fn forward(&self) -> Vec3 {
        (self.target - self.eye).normalized()
    }

    fn right(&self) -> Vec3 {
        let f = (self.target - self.eye).normalized();
        f.cross(self.up).normalized()
    }
}
