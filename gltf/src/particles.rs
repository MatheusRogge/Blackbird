//! Particle emitters in glTF: a node's `extras.blackbird_particles` describes one emitter placed
//! at the node. `GltfScene` spawns them (`node_emitters`).

use rendering::particles::{EmitterShape, ParticleEmitter, ParticleLighting};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use ultraviolet::{Mat4, Vec2, Vec3};

pub const EXTRAS_KEY: &str = "blackbird_particles";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShapeKind {
    Box,
    Sphere,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Blend {
    Additive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Lighting {
    Lit,
    Unlit,
}

/// The `blackbird_particles` object. Unknown keys (e.g. `notes`) are ignored.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParticleData {
    pub shape: ShapeKind,
    #[serde(default = "one3")]
    pub half_extents: [f32; 3],
    #[serde(default = "one")]
    pub radius: f32,
    /// The box is sheared so its horizontal slices follow this direction (node-local).
    #[serde(default)]
    pub shear_along: Option<[f32; 3]>,
    pub count: u32,
    pub spawn_rate: f32,
    /// Seconds, min and max.
    pub lifetime: [f32; 2],
    /// Diameter in meters, min and max.
    pub size: [f32; 2],
    #[serde(default)]
    pub velocity: [f32; 3],
    #[serde(default)]
    pub turbulence: f32,
    #[serde(default = "one3")]
    pub color: [f32; 3],
    #[serde(default = "one")]
    pub opacity: f32,
    #[serde(default = "additive")]
    pub blend: Blend,
    #[serde(default = "lit")]
    pub lighting: Lighting,
    #[serde(default)]
    pub visible_in_light_only: bool,
}

fn one() -> f32 {
    1.0
}

fn one3() -> [f32; 3] {
    [1.0; 3]
}

fn additive() -> Blend {
    Blend::Additive
}

fn lit() -> Lighting {
    Lighting::Lit
}

impl ParticleData {
    /// The emitter in `node`'s `extras`, if any. A malformed one is logged and skipped.
    pub fn read(node: &gltf::Node<'_>) -> Option<Self> {
        let raw = node.extras().as_ref()?;
        let Ok(Value::Object(mut extras)) = serde_json::from_str::<Value>(raw.get()) else { return None };
        let value = extras.remove(EXTRAS_KEY)?;
        match serde_json::from_value(value) {
            Ok(data) => Some(data),
            Err(err) => {
                log::warn!("node {:?}: invalid {EXTRAS_KEY}: {err}", node.name().unwrap_or("<unnamed>"));
                None
            }
        }
    }

    /// An emitter placed at `transform`. `seed` varies the random streams between emitters.
    pub fn emitter(&self, transform: Mat4, seed: u32) -> ParticleEmitter {
        let shape = match self.shape {
            ShapeKind::Box => {
                let shear = match self.shear_along {
                    Some([x, y, z]) if y.abs() > 1e-3 => Vec2::new(x / y, z / y),
                    _ => Vec2::zero(),
                };
                EmitterShape::Box { half_extents: self.half_extents.into(), shear }
            }
            ShapeKind::Sphere => EmitterShape::Sphere { radius: self.radius },
        };
        let sorted = |[a, b]: [f32; 2]| [a.min(b).max(0.0), a.max(b).max(0.0)];
        ParticleEmitter {
            transform,
            shape,
            count: self.count,
            spawn_rate: self.spawn_rate,
            lifetime: sorted(self.lifetime),
            size: sorted(self.size),
            velocity: Vec3::from(self.velocity),
            turbulence: self.turbulence,
            color: Vec3::from(self.color),
            opacity: self.opacity,
            lighting: match self.lighting {
                Lighting::Lit => ParticleLighting::Lit,
                Lighting::Unlit => ParticleLighting::Unlit,
            },
            light_only: self.visible_in_light_only,
            seed,
            ..ParticleEmitter::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_hospital_emitter() {
        let json = r#"{
            "shape": "box", "half_extents": [4.0, 4.5, 4.0], "shear_along": [0.2, -0.9, 0.3],
            "count": 2500, "spawn_rate": 160.0, "lifetime": [10.0, 20.0], "size": [0.002, 0.008],
            "velocity": [0.0, -0.012, 0.0], "turbulence": 0.05, "color": [0.85, 0.9, 1.0],
            "opacity": 0.35, "blend": "additive", "lighting": "lit", "visible_in_light_only": true,
            "notes": "ignored"
        }"#;
        let data: ParticleData = serde_json::from_str(json).expect("valid emitter");
        let emitter = data.emitter(Mat4::identity(), 3);
        assert_eq!(emitter.count, 2500);
        assert!(emitter.light_only);
        let EmitterShape::Box { half_extents, shear } = emitter.shape else { panic!("expected a box") };
        assert_eq!(half_extents, Vec3::new(4.0, 4.5, 4.0));
        assert!((shear.x + 0.2 / 0.9).abs() < 1e-6 && (shear.y + 0.3 / 0.9).abs() < 1e-6);
    }

    #[test]
    fn sphere_defaults() {
        let json = r#"{"shape": "sphere", "radius": 1.6, "count": 300, "spawn_rate": 20.0,
            "lifetime": [14.0, 8.0], "size": [0.0015, 0.005]}"#;
        let data: ParticleData = serde_json::from_str(json).expect("valid emitter");
        let emitter = data.emitter(Mat4::identity(), 0);
        assert_eq!(emitter.shape, EmitterShape::Sphere { radius: 1.6 });
        assert_eq!(emitter.lifetime, [8.0, 14.0]);
        assert_eq!(emitter.opacity, 1.0);
        assert_eq!(emitter.lighting, ParticleLighting::Lit);
    }
}
