//! Camera settings glTF can't express, kept in `extras`: physical exposure on a camera
//! (`blackbird_physical`) and the camera the player renders from on a scene
//! (`blackbird_main_camera`, a node index).

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const PHYSICAL_KEY: &str = "blackbird_physical";
pub const MAIN_CAMERA_KEY: &str = "blackbird_main_camera";
/// Far plane for a camera without `zfar`. The renderer needs a finite far plane for its
/// clusters, while glTF treats a missing `zfar` as infinite.
pub const INFINITE_FAR: f32 = 10_000.0;

/// Exposure settings of a real camera. Shutter speed is in seconds.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhysicalCamera {
    pub aperture: f32,
    pub shutter_speed: f32,
    pub iso: f32,
}

impl Default for PhysicalCamera {
    /// "Sunny 16": f/16, 1/125 s, ISO 100.
    fn default() -> Self {
        Self { aperture: 16.0, shutter_speed: 1.0 / 125.0, iso: 100.0 }
    }
}

impl PhysicalCamera {
    pub fn ev100(&self) -> f32 {
        (self.aperture * self.aperture / self.shutter_speed * 100.0 / self.iso).log2()
    }

    /// Linear multiplier from scene luminance (nits) to display values, saturating at the
    /// sensor's maximum luminance (Lagarde & de Rousiers, "Moving Frostbite to PBR").
    pub fn exposure(&self) -> f32 {
        1.0 / (1.2 * self.ev100().exp2())
    }

    pub fn read(camera: &gltf::Camera<'_>) -> Option<Self> {
        let value = extra(camera.extras(), PHYSICAL_KEY)?;
        serde_json::from_value(value).ok()
    }
}

/// Node index of the scene's main camera, if one is marked.
pub fn main_camera(scene: &gltf::Scene<'_>) -> Option<usize> {
    extra(scene.extras(), MAIN_CAMERA_KEY)?.as_u64().map(|idx| idx as usize)
}

fn extra(extras: &gltf::json::Extras, key: &str) -> Option<Value> {
    let raw = extras.as_ref()?;
    match serde_json::from_str::<Value>(raw.get()) {
        Ok(Value::Object(mut map)) => map.remove(key),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sunny_sixteen_is_about_ev15() {
        let camera = PhysicalCamera::default();
        assert!((camera.ev100() - 14.97).abs() < 0.01);
        assert!((camera.exposure() - 1.0 / (1.2 * 32_000.0)).abs() < 1e-7);
    }

    #[test]
    fn reads_physical_settings_and_main_camera_from_extras() {
        let json = r#"{
            "asset": { "version": "2.0" },
            "scene": 0,
            "scenes": [{ "nodes": [0, 1], "extras": { "blackbird_main_camera": 1, "other": true } }],
            "nodes": [{ "camera": 0 }, { "camera": 1 }],
            "cameras": [
                { "type": "perspective", "perspective": { "yfov": 0.8, "znear": 0.1 } },
                { "type": "perspective", "perspective": { "yfov": 0.8, "znear": 0.1 },
                  "extras": { "blackbird_physical": { "aperture": 2.8, "shutter_speed": 0.01, "iso": 400.0 } } }
            ]
        }"#;
        let document = gltf::Gltf::from_slice(json.as_bytes()).unwrap().document;

        let cameras: Vec<_> = document.cameras().map(|c| PhysicalCamera::read(&c)).collect();
        assert_eq!(cameras[0], None);
        assert_eq!(cameras[1], Some(PhysicalCamera { aperture: 2.8, shutter_speed: 0.01, iso: 400.0 }));
        assert_eq!(main_camera(&document.default_scene().unwrap()), Some(1));
    }
}
