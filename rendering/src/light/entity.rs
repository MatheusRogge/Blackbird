use engine_core::entity::Entity;
use ultraviolet::Vec3;

pub struct PointLight {
    pub position: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub radius: f32,
}

impl Entity for PointLight {}

pub struct SpotLight {
    pub position: Vec3,
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub radius: f32,
    pub inner_angle: f32,
    pub outer_angle: f32,
}

impl Entity for SpotLight {}

pub struct AreaLight {
    pub position: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub color: Vec3,
    pub intensity: f32,
}

impl Entity for AreaLight {}
