use std::f32::consts::FRAC_PI_2;

use engine::EntityKey;
use engine::rendering::camera::Camera;
use engine_core::input::{InputBuffer, InputEvent, Key, MouseButton};
use engine_core::world::World;
use ultraviolet::Vec3;

use crate::state::CameraSettings;

const KEY_Q: u32 = 16;
const KEY_W: u32 = 17;
const KEY_E: u32 = 18;
const KEY_A: u32 = 30;
const KEY_S: u32 = 31;
const KEY_D: u32 = 32;
const KEY_LSHIFT: u32 = 42;
const KEY_RSHIFT: u32 = 54;

/// Radians of rotation per pixel of cursor movement.
const LOOK_SENSITIVITY: f32 = 0.004;
/// Fraction of the pivot distance panned per pixel of cursor movement.
const PAN_SENSITIVITY: f32 = 0.0015;
/// Fraction of the pivot distance covered per scroll line.
const ZOOM_STEP: f32 = 0.15;
const MIN_PIVOT_DISTANCE: f32 = 0.1;
const MAX_PITCH: f32 = FRAC_PI_2 - 0.01;
/// Speed factor applied per scroll line while flying.
const SPEED_SCROLL_FACTOR: f32 = 1.2;

#[derive(Default)]
struct Held {
    forward: bool,
    back: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    fast: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum Drag {
    Orbit,
    Look,
    Pan,
}

impl Drag {
    fn for_button(button: MouseButton) -> Option<Self> {
        match button {
            MouseButton::Left => Some(Self::Look),
            MouseButton::Right => Some(Self::Orbit),
            MouseButton::Middle => Some(Self::Pan),
            MouseButton::Other => None,
        }
    }
}

/// Editor viewport camera.
///
/// - LMB drag looks around, RMB drag orbits around the pivot, MMB drag pans.
/// - WASD moves along the view direction, Q/E move down/up, Shift is faster.
/// - Scroll dollies toward the pivot. While LMB is held it changes the move speed instead.
///
/// Orbit and look turn the view the same way the cursor moves; they only differ
/// in whether the pivot or the eye stays fixed.
pub struct EditorCameraController {
    camera_key: EntityKey,
    held: Held,
    drag: Option<Drag>,
    last_cursor: Option<(f32, f32)>,
}

impl EditorCameraController {
    pub fn new(camera_key: EntityKey) -> Self {
        Self {
            camera_key,
            held: Held::default(),
            drag: None,
            last_cursor: None,
        }
    }

    pub fn tick(
        &mut self,
        input: &mut InputBuffer,
        world: &mut World,
        settings: &mut CameraSettings,
        delta: f32,
        ui_wants_pointer: bool,
        ui_wants_keyboard: bool,
    ) {
        let mut cursor_delta = (0.0f32, 0.0f32);
        let mut scroll = 0.0f32;
        let mut unhandled = Vec::new();

        while let Some(event) = input.pop() {
            match event {
                // Typing into the UI and chorded shortcuts like Ctrl+S are not movement.
                // Releases still go through so a held key never sticks.
                InputEvent::KeyPressed { modifiers, .. }
                    if ui_wants_keyboard || modifiers.ctrl || modifiers.alt || modifiers.super_ =>
                {
                    unhandled.push(event)
                }
                InputEvent::KeyPressed { key: Key::Scancode(code), .. } if self.set_held(code, true) => {}
                InputEvent::KeyReleased { key: Key::Scancode(code), .. } if self.set_held(code, false) => {}
                InputEvent::MouseButtonPressed { button, .. } => {
                    if !ui_wants_pointer && self.drag.is_none() {
                        self.drag = Drag::for_button(button);
                    }
                }
                InputEvent::MouseButtonReleased { button, .. } => {
                    if Drag::for_button(button).is_some_and(|d| self.drag == Some(d)) {
                        self.drag = None;
                    }
                }
                InputEvent::MouseMoved { x, y } => {
                    if self.drag.is_some()
                        && let Some((lx, ly)) = self.last_cursor
                    {
                        cursor_delta.0 += x - lx;
                        cursor_delta.1 += y - ly;
                    }
                    self.last_cursor = Some((x, y));
                }
                InputEvent::MouseScrolled { y, .. } if !ui_wants_pointer || self.drag.is_some() => scroll += y,
                InputEvent::CursorLeft => self.last_cursor = None,
                InputEvent::WindowFocused(false) => {
                    self.held = Held::default();
                    self.drag = None;
                }
                other => unhandled.push(other),
            }
        }

        for event in unhandled {
            input.push(event);
        }

        if scroll != 0.0 && self.drag == Some(Drag::Look) {
            settings.move_speed = (settings.move_speed * SPEED_SCROLL_FACTOR.powf(scroll))
                .clamp(CameraSettings::MIN_SPEED, CameraSettings::MAX_SPEED);
            scroll = 0.0;
        }

        let Some(camera) = world.get_entity_mut::<Camera>(self.camera_key) else { return };
        self.apply(camera, settings, cursor_delta, scroll, delta);
    }

    /// Records a movement key's state. Returns false for keys the camera doesn't use.
    fn set_held(&mut self, code: u32, down: bool) -> bool {
        let slot = match code {
            KEY_W => &mut self.held.forward,
            KEY_S => &mut self.held.back,
            KEY_A => &mut self.held.left,
            KEY_D => &mut self.held.right,
            KEY_E => &mut self.held.up,
            KEY_Q => &mut self.held.down,
            KEY_LSHIFT | KEY_RSHIFT => &mut self.held.fast,
            _ => return false,
        };
        *slot = down;
        true
    }

    fn apply(&self, camera: &mut Camera, settings: &CameraSettings, cursor: (f32, f32), scroll: f32, delta: f32) {
        let offset = camera.target - camera.eye;
        let mut distance = offset.mag().max(MIN_PIVOT_DISTANCE);
        let forward = offset.normalized();
        let mut yaw = f32::atan2(forward.x, -forward.z);
        let mut pitch = forward.y.clamp(-1.0, 1.0).asin().clamp(-MAX_PITCH, MAX_PITCH);
        let mut eye = camera.eye;

        match self.drag {
            Some(Drag::Look) | Some(Drag::Orbit) if cursor != (0.0, 0.0) => {
                let pivot = eye + direction(yaw, pitch) * distance;
                yaw += cursor.0 * LOOK_SENSITIVITY;
                pitch = (pitch - cursor.1 * LOOK_SENSITIVITY).clamp(-MAX_PITCH, MAX_PITCH);
                if self.drag == Some(Drag::Orbit) {
                    eye = pivot - direction(yaw, pitch) * distance;
                }
            }
            Some(Drag::Pan) if cursor != (0.0, 0.0) => {
                let (right, up) = basis(yaw, pitch);
                let scale = distance * PAN_SENSITIVITY;
                eye += (up * cursor.1 - right * cursor.0) * scale;
            }
            _ => {}
        }

        let forward = direction(yaw, pitch);

        if scroll != 0.0 {
            let step = distance.max(1.0) * (1.0 - (1.0 - ZOOM_STEP).powf(scroll));
            if distance - step >= MIN_PIVOT_DISTANCE {
                distance -= step;
            }
            eye += forward * step;
        }

        let (right, _) = basis(yaw, pitch);
        let mut movement = Vec3::zero();
        if self.held.forward { movement += forward; }
        if self.held.back    { movement -= forward; }
        if self.held.right   { movement += right; }
        if self.held.left    { movement -= right; }
        if self.held.up      { movement += Vec3::unit_y(); }
        if self.held.down    { movement -= Vec3::unit_y(); }
        if movement.mag_sq() > 0.0 {
            let speed = settings.move_speed * if self.held.fast { settings.fast_multiplier } else { 1.0 };
            eye += movement.normalized() * speed * delta;
        }

        camera.eye = eye;
        camera.target = eye + forward * distance;
        camera.up = Vec3::unit_y();
    }
}

/// View direction for a yaw (0 = -Z, positive turns toward +X) and pitch (positive looks up).
fn direction(yaw: f32, pitch: f32) -> Vec3 {
    Vec3::new(pitch.cos() * yaw.sin(), pitch.sin(), -pitch.cos() * yaw.cos())
}

/// Screen-right and screen-up vectors of the view.
fn basis(yaw: f32, pitch: f32) -> (Vec3, Vec3) {
    let right = Vec3::new(yaw.cos(), 0.0, yaw.sin());
    let up = right.cross(direction(yaw, pitch));
    (right, up)
}
