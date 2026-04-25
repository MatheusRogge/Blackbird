use engine_core::{
    entity::EntityKey,
    input::{InputBuffer, InputEvent, MouseButton},
    world::World,
};
use ultraviolet::Vec3;

use crate::camera::Camera;

pub enum CameraMode {
    FirstPerson,
    ThirdPerson,
}

pub struct CameraController {
    pub camera_key: EntityKey,
    pub mode: CameraMode,
    /// Radians of rotation per pixel of cursor movement.
    pub pan_speed: f32,
    /// World units moved per scroll line.
    pub zoom_speed: f32,
    /// Minimum distance between eye and target.
    pub min_zoom_distance: f32,
    is_panning: bool,
    last_cursor_pos: Option<(f32, f32)>,
}

impl CameraController {
    pub fn new(camera_key: EntityKey, mode: CameraMode) -> Self {
        Self {
            camera_key,
            mode,
            pan_speed: 0.005,
            zoom_speed: 2.0,
            min_zoom_distance: 1.0,
            is_panning: false,
            last_cursor_pos: None,
        }
    }

    pub fn tick(&mut self, input: &mut InputBuffer, world: &mut World, _delta: f32) {
        let events: Vec<InputEvent> = std::iter::from_fn(|| input.pop()).collect();

        let mut pan_delta = (0.0f32, 0.0f32);
        let mut scroll_delta = 0.0f32;
        let mut unhandled: Vec<InputEvent> = Vec::new();

        for event in events {
            match event {
                InputEvent::MouseButtonPressed { button: MouseButton::Left } => {
                    self.is_panning = true;
                }
                InputEvent::MouseButtonReleased { button: MouseButton::Left } => {
                    self.is_panning = false;
                    self.last_cursor_pos = None;
                }
                InputEvent::MouseMoved { x, y } => {
                    if self.is_panning
                        && let Some((lx, ly)) = self.last_cursor_pos
                    {
                        pan_delta.0 += x - lx;
                        pan_delta.1 += y - ly;
                    }
                    self.last_cursor_pos = Some((x, y));
                }
                InputEvent::MouseScrolled { delta } => {
                    scroll_delta += delta;
                }
                other => unhandled.push(other),
            }
        }

        for event in unhandled.into_iter().rev() {
            input.push(event);
        }

        if pan_delta == (0.0, 0.0) && scroll_delta == 0.0 {
            return;
        }

        let Some(camera) = world.get_entity_mut::<Camera>(self.camera_key) else {
            return;
        };

        if scroll_delta != 0.0 {
            Self::apply_zoom(camera, scroll_delta, self.zoom_speed, self.min_zoom_distance);
        }

        if pan_delta != (0.0, 0.0)
            && let CameraMode::FirstPerson = self.mode
        {
            Self::apply_pan(camera, pan_delta.0, pan_delta.1, self.pan_speed);
        }
    }

    fn apply_zoom(camera: &mut Camera, scroll: f32, speed: f32, min_dist: f32) {
        let forward = (camera.target - camera.eye).normalized();
        let distance = (camera.target - camera.eye).mag();
        let new_distance = (distance - scroll * speed).max(min_dist);
        camera.eye = camera.target - forward * new_distance;
    }

    fn apply_pan(camera: &mut Camera, delta_x: f32, delta_y: f32, speed: f32) {
        let offset = camera.target - camera.eye;
        let dist = offset.mag();

        let yaw = f32::atan2(offset.x, -offset.z);
        let pitch = (offset.y / dist).asin();

        let new_yaw = yaw - delta_x * speed;
        let new_pitch = (pitch + delta_y * speed)
            .clamp(-std::f32::consts::FRAC_PI_2 + 0.01, std::f32::consts::FRAC_PI_2 - 0.01);

        let new_offset = Vec3::new(
            dist * new_pitch.cos() * new_yaw.sin(),
            dist * new_pitch.sin(),
            -dist * new_pitch.cos() * new_yaw.cos(),
        );
        camera.target = camera.eye + new_offset;
        camera.up = Vec3::unit_y();
    }
}
