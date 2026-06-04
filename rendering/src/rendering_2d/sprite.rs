use engine_core::entity::Entity;
use std::sync::Arc;
use ultraviolet::Vec2;

pub struct SpriteTexture {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

pub struct Sprite {
    pub texture: Arc<SpriteTexture>,
    pub position: Vec2,
    pub rotation: f32,
    pub scale: Vec2,
    pub color: [f32; 4],
    pub z_order: f32,
    pub uv_rect: [f32; 4],
}

impl Sprite {
    pub fn new(texture: Arc<SpriteTexture>) -> Self {
        Self {
            texture,
            position: Vec2::zero(),
            rotation: 0.0,
            scale: Vec2::one(),
            color: [1.0, 1.0, 1.0, 1.0],
            z_order: 0.0,
            uv_rect: [0.0, 0.0, 1.0, 1.0],
        }
    }
}

impl Entity for Sprite {}
