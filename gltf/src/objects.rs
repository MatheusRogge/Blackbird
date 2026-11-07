//! Scene objects the editor places besides plain glTF nodes, stored as `blackbird_objects` in the
//! root `extras`: built-in shapes (generated, not stored as geometry), placed models (other glTF
//! files, by URI relative to the scene) and the player start.

use std::f32::consts::TAU;

use rendering::mesh::{MaterialFactors, Mesh, Vertex};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use ultraviolet::{Mat4, Vec3, Vec4};

pub const EXTRAS_KEY: &str = "blackbird_objects";
/// Height of the player's eyes above the player start, in meters.
pub const PLAYER_EYE_HEIGHT: f32 = 1.7;
const ROUND_SEGMENTS: u32 = 32;
const SPHERE_RINGS: u32 = 16;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneObject {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub translation: [f32; 3],
    /// Quaternion, `[x, y, z, w]` as in glTF.
    #[serde(default = "identity_rotation")]
    pub rotation: [f32; 4],
    #[serde(default = "unit_scale")]
    pub scale: [f32; 3],
    #[serde(flatten)]
    pub kind: ObjectKind,
}

fn identity_rotation() -> [f32; 4] {
    [0.0, 0.0, 0.0, 1.0]
}

fn unit_scale() -> [f32; 3] {
    [1.0; 3]
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ObjectKind {
    Shape {
        shape: Shape,
        #[serde(default)]
        material: ShapeMaterial,
    },
    /// Another glTF file's default scene, placed with this object's transform.
    Model { uri: String },
    /// Where the player stands (feet) and faces (local −Z; only the heading is used).
    PlayerStart,
}

/// Unit-sized and centered on the origin: a 1 m cube, a sphere and a cylinder 1 m across (the
/// cylinder 1 m tall along Y), and a 1 m square plane in XZ facing +Y.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Shape {
    Cube,
    Sphere,
    Plane,
    Cylinder,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ShapeMaterial {
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub emissive_strength: f32,
}

impl Default for ShapeMaterial {
    fn default() -> Self {
        Self { base_color: [0.8, 0.8, 0.8, 1.0], metallic: 0.0, roughness: 0.5, emissive: [0.0; 3], emissive_strength: 1.0 }
    }
}

impl ShapeMaterial {
    pub fn factors(&self) -> MaterialFactors {
        MaterialFactors {
            base_color: self.base_color,
            metallic: self.metallic,
            roughness: self.roughness,
            emissive: self.emissive,
            emissive_strength: self.emissive_strength,
            ..MaterialFactors::default()
        }
    }
}

impl SceneObject {
    pub fn matrix(&self) -> Mat4 {
        trs(self.translation, self.rotation, self.scale)
    }

    /// The model's URI, if this places one.
    pub fn model_uri(&self) -> Option<&str> {
        match &self.kind {
            ObjectKind::Model { uri } => Some(uri),
            _ => None,
        }
    }
}

/// glTF-style translation, `[x, y, z, w]` rotation and scale to a matrix.
pub fn trs(translation: [f32; 3], rotation: [f32; 4], scale: [f32; 3]) -> Mat4 {
    let [x, y, z, w] = rotation;
    let rot = Mat4::new(
        Vec4::new(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w), 0.0),
        Vec4::new(2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w), 0.0),
        Vec4::new(2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y), 0.0),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
    );
    Mat4::from_translation(translation.into()) * rot * Mat4::from_nonuniform_scale(Vec3::from(scale))
}

/// Objects stored in `doc`'s root `extras`.
pub fn read_objects(doc: &gltf::Document) -> Result<Vec<SceneObject>, serde_json::Error> {
    let Some(raw) = &doc.as_json().extras else { return Ok(Vec::new()) };
    let Ok(Value::Object(mut extras)) = serde_json::from_str::<Value>(raw.get()) else { return Ok(Vec::new()) };
    match extras.remove(EXTRAS_KEY) {
        Some(value) => serde_json::from_value(value),
        None => Ok(Vec::new()),
    }
}

impl Shape {
    pub const ALL: [Shape; 4] = [Shape::Cube, Shape::Sphere, Shape::Plane, Shape::Cylinder];

    pub fn label(self) -> &'static str {
        match self {
            Shape::Cube => "Cube",
            Shape::Sphere => "Sphere",
            Shape::Plane => "Plane",
            Shape::Cylinder => "Cylinder",
        }
    }

    /// How far the shape extends below its origin along Y, so it can be set down on a surface.
    pub fn half_height(self) -> f32 {
        match self {
            Shape::Plane => 0.0,
            _ => 0.5,
        }
    }

    pub fn mesh(self, material: &ShapeMaterial) -> Mesh {
        let mut builder = Builder::default();
        match self {
            Shape::Cube => builder.cube(),
            Shape::Sphere => builder.sphere(),
            Shape::Plane => builder.quad(Vec3::zero(), Vec3::unit_x(), -Vec3::unit_z()),
            Shape::Cylinder => builder.cylinder(),
        }
        Mesh::new(builder.vertices, builder.indices, None, None).with_material(material.factors(), None, None)
    }
}

/// Triangles wound counter-clockwise seen from the side the normal points to, as glTF expects.
#[derive(Default)]
struct Builder {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

impl Builder {
    fn vertex(&mut self, position: Vec3, normal: Vec3, uv: [f32; 2], tangent: Vec3) -> u32 {
        self.vertices.push(Vertex {
            position: position.into(),
            normal: normal.into(),
            color: [1.0; 3],
            uv,
            tangent: [tangent.x, tangent.y, tangent.z, 1.0],
        });
        self.vertices.len() as u32 - 1
    }

    /// A 1×1 square centered at `center`, spanning `u` (left to right) and `v` (bottom to top),
    /// facing `u × v`.
    fn quad(&mut self, center: Vec3, u: Vec3, v: Vec3) {
        let normal = u.cross(v);
        let corner = |su: f32, sv: f32| center + u * (0.5 * su) + v * (0.5 * sv);
        let a = self.vertex(corner(-1.0, -1.0), normal, [0.0, 1.0], u);
        let b = self.vertex(corner(1.0, -1.0), normal, [1.0, 1.0], u);
        let c = self.vertex(corner(1.0, 1.0), normal, [1.0, 0.0], u);
        let d = self.vertex(corner(-1.0, 1.0), normal, [0.0, 0.0], u);
        self.indices.extend([a, b, c, a, c, d]);
    }

    fn cube(&mut self) {
        let (x, y, z) = (Vec3::unit_x(), Vec3::unit_y(), Vec3::unit_z());
        self.quad(z * 0.5, x, y);
        self.quad(-z * 0.5, -x, y);
        self.quad(x * 0.5, -z, y);
        self.quad(-x * 0.5, z, y);
        self.quad(y * 0.5, x, -z);
        self.quad(-y * 0.5, x, z);
    }

    fn sphere(&mut self) {
        let start = self.vertices.len() as u32;
        for ring in 0..=SPHERE_RINGS {
            let v = ring as f32 / SPHERE_RINGS as f32;
            let polar = v * std::f32::consts::PI;
            for segment in 0..=ROUND_SEGMENTS {
                let u = segment as f32 / ROUND_SEGMENTS as f32;
                let azimuth = u * TAU;
                let normal = Vec3::new(polar.sin() * azimuth.sin(), polar.cos(), polar.sin() * azimuth.cos());
                let tangent = Vec3::new(azimuth.cos(), 0.0, -azimuth.sin());
                self.vertex(normal * 0.5, normal, [u, v], tangent);
            }
        }
        let row = ROUND_SEGMENTS + 1;
        for ring in 0..SPHERE_RINGS {
            for segment in 0..ROUND_SEGMENTS {
                let a = start + ring * row + segment;
                let (b, c, d) = (a + row, a + row + 1, a + 1);
                if ring > 0 {
                    self.indices.extend([a, b, d]);
                }
                if ring + 1 < SPHERE_RINGS {
                    self.indices.extend([d, b, c]);
                }
            }
        }
    }

    fn cylinder(&mut self) {
        let start = self.vertices.len() as u32;
        for segment in 0..=ROUND_SEGMENTS {
            let u = segment as f32 / ROUND_SEGMENTS as f32;
            let azimuth = u * TAU;
            let normal = Vec3::new(azimuth.sin(), 0.0, azimuth.cos());
            let tangent = Vec3::new(azimuth.cos(), 0.0, -azimuth.sin());
            self.vertex(normal * 0.5 - Vec3::unit_y() * 0.5, normal, [u, 1.0], tangent);
            self.vertex(normal * 0.5 + Vec3::unit_y() * 0.5, normal, [u, 0.0], tangent);
        }
        for segment in 0..ROUND_SEGMENTS {
            let a = start + segment * 2;
            let (b, c, d) = (a + 1, a + 3, a + 2);
            self.indices.extend([a, d, c, a, c, b]);
        }
        for (y, normal) in [(0.5, Vec3::unit_y()), (-0.5, -Vec3::unit_y())] {
            let center = self.vertex(Vec3::unit_y() * y, normal, [0.5, 0.5], Vec3::unit_x());
            let first = self.vertices.len() as u32;
            for segment in 0..=ROUND_SEGMENTS {
                let azimuth = segment as f32 / ROUND_SEGMENTS as f32 * TAU;
                let (s, c) = azimuth.sin_cos();
                self.vertex(Vec3::new(s * 0.5, y, c * 0.5), normal, [0.5 + s * 0.5, 0.5 - c * 0.5], Vec3::unit_x());
            }
            for segment in 0..ROUND_SEGMENTS {
                let (a, b) = (first + segment, first + segment + 1);
                if y > 0.0 {
                    self.indices.extend([center, a, b]);
                } else {
                    self.indices.extend([center, b, a]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_are_wound_counter_clockwise_outward() {
        for shape in Shape::ALL {
            let mesh = shape.mesh(&ShapeMaterial::default());
            assert!(!mesh.indices.is_empty(), "{shape:?} has no triangles");
            for tri in mesh.indices.chunks(3) {
                let [a, b, c] = [tri[0], tri[1], tri[2]].map(|i| mesh.vertices[i as usize]);
                let (pa, pb, pc) = (Vec3::from(a.position), Vec3::from(b.position), Vec3::from(c.position));
                let face = (pb - pa).cross(pc - pa);
                let normal = Vec3::from(a.normal) + Vec3::from(b.normal) + Vec3::from(c.normal);
                assert!(face.dot(normal) > 0.0, "{shape:?} triangle {tri:?} faces inward");
            }
        }
    }

    #[test]
    fn objects_round_trip_through_json() {
        let objects = vec![
            SceneObject {
                name: "Crate".into(),
                translation: [1.0, 0.5, -2.0],
                rotation: [0.0, 0.38268343, 0.0, 0.9238795],
                scale: [2.0, 1.0, 1.0],
                kind: ObjectKind::Shape { shape: Shape::Cube, material: ShapeMaterial::default() },
            },
            SceneObject {
                name: "Chair".into(),
                translation: [0.0; 3],
                rotation: identity_rotation(),
                scale: unit_scale(),
                kind: ObjectKind::Model { uri: "../assets/models/chair/chair.gltf".into() },
            },
            SceneObject {
                name: String::new(),
                translation: [3.0, 0.0, 3.0],
                rotation: identity_rotation(),
                scale: unit_scale(),
                kind: ObjectKind::PlayerStart,
            },
        ];
        let json = serde_json::to_value(&objects).unwrap();
        assert_eq!(json[2]["type"], "player_start");
        let back: Vec<SceneObject> = serde_json::from_value(json).unwrap();
        assert_eq!(back, objects);
    }

    #[test]
    fn trs_matches_rotation_of_a_vector() {
        // 90° about +Y takes -Z (forward) to -X.
        let half = std::f32::consts::FRAC_PI_4;
        let m = trs([0.0; 3], [0.0, half.sin(), 0.0, half.cos()], [1.0; 3]);
        let forward = m.transform_vec3(-Vec3::unit_z());
        assert!((forward - -Vec3::unit_x()).mag() < 1e-5, "{forward:?}");
    }
}
