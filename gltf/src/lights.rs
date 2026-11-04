//! Scene lights in glTF: point, spot and sky lights as `KHR_lights_punctual` on nodes, area
//! lights (which the extension can't express) as `blackbird_lights` in the root `extras`.

use std::collections::HashSet;

use engine_core::{entity::EntityKey, world::World};
use rendering::light::{AreaLight, PointLight, SkyLight, SpotLight};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use ultraviolet::{Mat4, Vec4};

pub const EXTENSION: &str = "KHR_lights_punctual";
pub const EXTRAS_KEY: &str = "blackbird_lights";
/// Illuminance (lux) at which a light with no `range` is cut off. The renderer needs a finite
/// radius to cluster point lights, while glTF treats a missing range as infinite.
const CUTOFF_ILLUMINANCE: f32 = 0.01;

/// A light in physical units: lux for sky, candela for point/spot, nits for area.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LightData {
    Sky {
        color: [f32; 3],
        intensity: f32,
        direction: [f32; 3],
    },
    Point {
        position: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        radius: f32,
    },
    Spot {
        position: [f32; 3],
        direction: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        radius: f32,
        inner_angle: f32,
        outer_angle: f32,
    },
    Area {
        position: [f32; 3],
        right: [f32; 3],
        up: [f32; 3],
        color: [f32; 3],
        intensity: f32,
    },
}

impl LightData {
    pub fn spawn(&self, world: &mut World) -> EntityKey {
        match *self {
            LightData::Sky { color, intensity, direction } => world.add_entity(SkyLight {
                direction: direction.into(),
                color: color.into(),
                intensity,
            }),
            LightData::Point { position, color, intensity, radius } => world.add_entity(PointLight {
                position: position.into(),
                color: color.into(),
                intensity,
                radius,
            }),
            LightData::Spot { position, direction, color, intensity, radius, inner_angle, outer_angle } => {
                world.add_entity(SpotLight {
                    position: position.into(),
                    direction: direction.into(),
                    color: color.into(),
                    intensity,
                    radius,
                    inner_angle,
                    outer_angle,
                })
            }
            LightData::Area { position, right, up, color, intensity } => world.add_entity(AreaLight {
                position: position.into(),
                right: right.into(),
                up: up.into(),
                color: color.into(),
                intensity,
            }),
        }
    }

    /// Writes this data into the light entity `key` created by `spawn`.
    pub fn apply(&self, world: &mut World, key: EntityKey) {
        match *self {
            LightData::Sky { color, intensity, direction } => {
                if let Some(l) = world.get_entity_mut::<SkyLight>(key) {
                    l.color = color.into();
                    l.intensity = intensity;
                    l.direction = direction.into();
                }
            }
            LightData::Point { position, color, intensity, radius } => {
                if let Some(l) = world.get_entity_mut::<PointLight>(key) {
                    l.position = position.into();
                    l.color = color.into();
                    l.intensity = intensity;
                    l.radius = radius;
                }
            }
            LightData::Spot { position, direction, color, intensity, radius, inner_angle, outer_angle } => {
                if let Some(l) = world.get_entity_mut::<SpotLight>(key) {
                    l.position = position.into();
                    l.direction = direction.into();
                    l.color = color.into();
                    l.intensity = intensity;
                    l.radius = radius;
                    l.inner_angle = inner_angle;
                    l.outer_angle = outer_angle;
                }
            }
            LightData::Area { position, right, up, color, intensity } => {
                if let Some(l) = world.get_entity_mut::<AreaLight>(key) {
                    l.position = position.into();
                    l.right = right.into();
                    l.up = up.into();
                    l.color = color.into();
                    l.intensity = intensity;
                }
            }
        }
    }
}

/// Lights stored in `doc`, each with the node it is attached to (`None` for `extras` lights).
pub fn read_lights(doc: &gltf::Document) -> Result<Vec<(Option<usize>, LightData)>, serde_json::Error> {
    let mut lights = read_punctual(doc);
    lights.extend(read_extras(doc)?.into_iter().map(|data| (None, data)));
    Ok(lights)
}

fn read_punctual(doc: &gltf::Document) -> Vec<(Option<usize>, LightData)> {
    let mut lights = Vec::new();
    let mut visited = HashSet::new();
    let mut stack: Vec<_> = doc
        .scenes()
        .flat_map(|scene| scene.nodes())
        .map(|node| (node, Mat4::identity()))
        .collect();

    while let Some((node, parent)) = stack.pop() {
        if !visited.insert(node.index()) {
            continue;
        }
        let world = parent * Mat4::from(node.transform().matrix());
        stack.extend(node.children().map(|child| (child, world)));

        let Some(light) = node.light() else { continue };
        let position = world.cols[3].xyz().into();
        let direction = (world * Vec4::new(0.0, 0.0, -1.0, 0.0)).xyz().normalized().into();
        let color = light.color();
        let intensity = light.intensity();
        let radius = light.range().unwrap_or_else(|| (intensity / CUTOFF_ILLUMINANCE).sqrt().max(1.0));
        let data = match light.kind() {
            gltf::khr_lights_punctual::Kind::Directional => LightData::Sky { color, intensity, direction },
            gltf::khr_lights_punctual::Kind::Point => LightData::Point { position, color, intensity, radius },
            gltf::khr_lights_punctual::Kind::Spot { inner_cone_angle, outer_cone_angle } => LightData::Spot {
                position,
                direction,
                color,
                intensity,
                radius,
                inner_angle: inner_cone_angle,
                outer_angle: outer_cone_angle,
            },
        };
        lights.push((Some(node.index()), data));
    }
    lights
}

fn read_extras(doc: &gltf::Document) -> Result<Vec<LightData>, serde_json::Error> {
    let Some(raw) = &doc.as_json().extras else { return Ok(Vec::new()) };
    let Ok(Value::Object(mut extras)) = serde_json::from_str::<Value>(raw.get()) else { return Ok(Vec::new()) };
    match extras.remove(EXTRAS_KEY) {
        Some(lights) => serde_json::from_value(lights),
        None => Ok(Vec::new()),
    }
}
