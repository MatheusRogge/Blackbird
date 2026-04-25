use ultraviolet::Vec3;

use engine_core::{
    entity::{Controllable, EntityKey},
    input::{InputEvent, KeyCodes},
};

use crate::Engine;

pub struct PlayerController {
    pub move_speed: f32,
    controlled: Option<EntityKey>,
    moving_forward: bool,
    moving_back: bool,
    moving_left: bool,
    moving_right: bool,
}

impl PlayerController {
    pub fn new(move_speed: f32) -> Self {
        Self {
            move_speed,
            controlled: None,
            moving_forward: false,
            moving_back: false,
            moving_left: false,
            moving_right: false,
        }
    }

    pub fn attach(&mut self, key: EntityKey) {
        self.controlled = Some(key);
    }

    pub fn detach(&mut self) {
        self.controlled = None;
    }

    pub fn tick<E: Controllable>(&mut self, engine: &mut Engine, delta: f32) {
        let events: Vec<InputEvent> = std::iter::from_fn(|| engine.input().pop()).collect();

        let mut unhandled: Vec<InputEvent> = Vec::new();

        for event in events {
            match event {
                InputEvent::KeyPressed { key_code } => match key_code {
                    KeyCodes::W => self.moving_forward = true,
                    KeyCodes::S => self.moving_back = true,
                    KeyCodes::A => self.moving_left = true,
                    KeyCodes::D => self.moving_right = true,
                    _ => {}
                },
                InputEvent::KeyReleased { key_code } => match key_code {
                    KeyCodes::W => self.moving_forward = false,
                    KeyCodes::S => self.moving_back = false,
                    KeyCodes::A => self.moving_left = false,
                    KeyCodes::D => self.moving_right = false,
                    _ => {}
                },
                other => unhandled.push(other),
            }
        }

        for event in unhandled.into_iter().rev() {
            engine.input().push(event);
        }

        let Some(key) = self.controlled else { return };

        let mut dir = Vec3::zero();
        if self.moving_forward { dir.z -= 1.0; }
        if self.moving_back    { dir.z += 1.0; }
        if self.moving_left    { dir.x -= 1.0; }
        if self.moving_right   { dir.x += 1.0; }

        if dir.mag_sq() == 0.0 {
            return;
        }

        let mut world = engine.world();
        if let Some(entity) = world.get_entity_mut::<E>(key) {
            let world_dir = entity.forward() * (-dir.z) + entity.right() * dir.x;
            let movement = world_dir.normalized() * self.move_speed * delta;
            entity.translate(movement);
        }
    }
}
