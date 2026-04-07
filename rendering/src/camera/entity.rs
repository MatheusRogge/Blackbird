use engine::entity::Entity;
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

// pub const OPENGL_TO_WGPU_MATRIX: Mat4 = Mat4::new(
//     Vec4::new(1.0, 0.0, 0.0, 0.0),
//     Vec4::new(0.0, 1.0, 0.0, 0.0),
//     Vec4::new(0.0, 0.0, 0.5, 0.0),
//     Vec4::new(0.0, 0.0, 0.5, 1.0),
// );

impl Camera {
    pub fn view_proj_matrix(&self) -> Mat4 {
        let view = Mat4::look_at(self.eye, self.target, self.up);

        // Reversed-Z for better depth precision — near/far swapped,
        // clip depth maps to [1, 0] instead of [0, 1].
        let proj =
            projection::perspective_reversed_z_wgpu_dx_gl(self.fovy, self.aspect, self.near, self.far);

        proj * view
    }
}

impl Entity for Camera {}
