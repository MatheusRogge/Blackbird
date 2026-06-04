use engine_core::entity::Entity;
use ultraviolet::Vec2;

pub struct Camera2D {
    pub position: Vec2,
    pub zoom: f32,
}

impl Default for Camera2D {
    fn default() -> Self {
        Self {
            position: Vec2::zero(),
            zoom: 1.0,
        }
    }
}

impl Camera2D {
    pub fn ortho_matrix(&self, width: u32, height: u32) -> [[f32; 4]; 4] {
        let w = width as f32;
        let h = height as f32;
        let z = self.zoom;
        let cx = self.position.x;
        let cy = self.position.y;

        let l = cx - w / (2.0 * z);
        let r = cx + w / (2.0 * z);
        let b = cy - h / (2.0 * z);
        let t = cy + h / (2.0 * z);

        // Column-major orthographic matrix, Y-up world space, Z in [0, 1]
        [
            [2.0 / (r - l), 0.0, 0.0, 0.0],
            [0.0, 2.0 / (t - b), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-(r + l) / (r - l), -(t + b) / (t - b), 0.0, 1.0],
        ]
    }
}

impl Entity for Camera2D {}
