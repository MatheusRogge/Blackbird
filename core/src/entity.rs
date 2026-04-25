use downcast_rs::{Downcast, impl_downcast};
use slotmap::new_key_type;
use std::any::Any;
use ultraviolet::Vec3;

new_key_type! {
    pub struct EntityKey;
}

pub trait Entity: Any + Downcast + Send + Sync + 'static {
    fn tick(&mut self, _delta: f32) {}
}

impl_downcast!(Entity);

pub trait Controllable: Entity {
    fn translate(&mut self, delta: Vec3);
    fn forward(&self) -> Vec3 {
        Vec3::new(0.0, 0.0, -1.0)
    }
    fn right(&self) -> Vec3 {
        Vec3::new(1.0, 0.0, 0.0)
    }
}
