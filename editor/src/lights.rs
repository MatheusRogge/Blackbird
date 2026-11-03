//! Scene lights in glTF: point, spot and sky lights as `KHR_lights_punctual` on nodes, area
//! lights (which the extension can't express) in the root `extras`.

use std::collections::HashSet;

use anyhow::{Context, Result};
use gltf_api::Document;
use gltf_api::json::extensions::scene::khr_lights_punctual::{self as khr, Type};
use gltf_api::json::scene::UnitQuaternion;
use gltf_api::json::validation::Checked;
use gltf_api::json::{Index, Root};
use serde_json::value::RawValue;
use serde_json::{Map, Value};
use ultraviolet::{Mat4, Vec3, Vec4};

use crate::math::{self, Quat};
use crate::state::{EditorState, LightData};

const EXTENSION: &str = "KHR_lights_punctual";
const EXTRAS_KEY: &str = "blackbird_lights";
const DEFAULT_RANGE: f32 = 10.0;

/// Lights stored in `doc`, each with the node it is attached to (`None` for `extras` lights).
pub fn read_lights(doc: &Document) -> Result<Vec<(Option<usize>, LightData)>> {
    let mut lights = read_punctual(doc);
    lights.extend(read_extras(doc)?.into_iter().map(|data| (None, data)));
    Ok(lights)
}

fn read_punctual(doc: &Document) -> Vec<(Option<usize>, LightData)> {
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
        let radius = light.range().unwrap_or(DEFAULT_RANGE);
        let data = match light.kind() {
            gltf_api::khr_lights_punctual::Kind::Directional => LightData::Sky { color, intensity, direction },
            gltf_api::khr_lights_punctual::Kind::Point => LightData::Point { position, color, intensity, radius },
            gltf_api::khr_lights_punctual::Kind::Spot { inner_cone_angle, outer_cone_angle } => LightData::Spot {
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

fn read_extras(doc: &Document) -> Result<Vec<LightData>> {
    let Some(raw) = &doc.as_json().extras else { return Ok(Vec::new()) };
    let Ok(Value::Object(mut extras)) = serde_json::from_str::<Value>(raw.get()) else { return Ok(Vec::new()) };
    match extras.remove(EXTRAS_KEY) {
        Some(lights) => serde_json::from_value(lights).context("invalid scene lights"),
        None => Ok(Vec::new()),
    }
}

/// Replaces every light in `root` with the editor's lights.
pub fn write_lights(root: &mut Root, state: &EditorState) -> Result<()> {
    clear_punctual(root)?;

    let mut punctual = Vec::new();
    let mut area = Vec::new();
    for light in &state.lights {
        let Some(khr_light) = to_khr(&light.data) else {
            area.push(&light.data);
            continue;
        };
        let existing = light.node.filter(|&idx| idx < root.nodes.len());
        let node_idx = existing.unwrap_or_else(|| add_root_node(root, light_name(&light.data)));
        let parent = if existing.is_some() { state.parent_world(node_idx) } else { Mat4::identity() };
        let current = parent * node_matrix(&root.nodes[node_idx]);
        set_local_transform(&mut root.nodes[node_idx], parent, light_world(&light.data, &current));

        let index = Index::new(punctual.len() as u32);
        root.nodes[node_idx].extensions.get_or_insert_with(Default::default).khr_lights_punctual =
            Some(khr::KhrLightsPunctual { light: index });
        punctual.push(khr_light);
    }

    if !punctual.is_empty() {
        root.extensions.get_or_insert_with(Default::default).khr_lights_punctual =
            Some(gltf_api::json::extensions::root::KhrLightsPunctual { lights: punctual });
        root.extensions_used.push(EXTENSION.to_string());
    }
    write_extras(root, &area)
}

fn clear_punctual(root: &mut Root) -> Result<()> {
    root.extensions_used.retain(|e| e != EXTENSION);
    root.extensions_required.retain(|e| e != EXTENSION);
    if let Some(ext) = &mut root.extensions {
        ext.khr_lights_punctual = None;
        if is_empty(&*ext)? {
            root.extensions = None;
        }
    }
    for node in &mut root.nodes {
        if let Some(ext) = &mut node.extensions {
            ext.khr_lights_punctual = None;
            if is_empty(&*ext)? {
                node.extensions = None;
            }
        }
    }
    Ok(())
}

fn is_empty(value: &impl serde::Serialize) -> Result<bool> {
    Ok(serde_json::to_value(value)?.as_object().is_some_and(Map::is_empty))
}

fn add_root_node(root: &mut Root, name: &str) -> usize {
    let idx = root.nodes.len();
    root.nodes.push(gltf_api::json::Node { name: Some(name.to_string()), ..Default::default() });
    let scene_idx = root.scene.map(|s| s.value()).unwrap_or(0);
    match root.scenes.get_mut(scene_idx) {
        Some(scene) => scene.nodes.push(Index::new(idx as u32)),
        None => root.scenes.push(gltf_api::json::Scene {
            extensions: None,
            extras: Default::default(),
            name: None,
            nodes: vec![Index::new(idx as u32)],
        }),
    }
    idx
}

fn light_name(data: &LightData) -> &'static str {
    match data {
        LightData::Sky { .. } => "Sky Light",
        LightData::Point { .. } => "Point Light",
        LightData::Spot { .. } => "Spot Light",
        LightData::Area { .. } => "Area Light",
    }
}

fn node_matrix(node: &gltf_api::json::Node) -> Mat4 {
    if let Some(m) = node.matrix {
        return Mat4::from(m);
    }
    math::trs(
        node.translation.unwrap_or([0.0; 3]),
        node.rotation.map(|r| r.0).unwrap_or([0.0, 0.0, 0.0, 1.0]),
        node.scale.unwrap_or([1.0; 3]),
    )
}

/// World transform for a light node. Sky lights have no position, so they keep the node's.
fn light_world(data: &LightData, current: &Mat4) -> Mat4 {
    let (position, direction) = match data {
        LightData::Sky { direction, .. } => (current.cols[3].xyz().into(), *direction),
        LightData::Point { position, .. } => (*position, [0.0, 0.0, -1.0]),
        LightData::Spot { position, direction, .. } => (*position, *direction),
        LightData::Area { position, .. } => (*position, [0.0, 0.0, -1.0]),
    };
    math::trs(position, rotation_to(direction.into()), [1.0; 3])
}

/// Rotation taking the local -Z axis onto `dir`.
fn rotation_to(dir: Vec3) -> Quat {
    let from = Vec3::new(0.0, 0.0, -1.0);
    let dir = dir.normalized();
    let dot = from.dot(dir);
    if dot > 0.9999 {
        [0.0, 0.0, 0.0, 1.0]
    } else if dot < -0.9999 {
        math::quat_from_axis_angle(Vec3::unit_y(), std::f32::consts::PI)
    } else {
        math::quat_from_axis_angle(from.cross(dir), dot.acos())
    }
}

fn set_local_transform(node: &mut gltf_api::json::Node, parent: Mat4, world: Mat4) {
    let local = parent.inversed() * world;
    node.matrix = None;
    node.translation = Some(local.cols[3].xyz().into());
    node.rotation = Some(UnitQuaternion(math::quat_from_matrix(&local)));
    node.scale = Some([local.cols[0].xyz().mag(), local.cols[1].xyz().mag(), local.cols[2].xyz().mag()]);
}

fn to_khr(data: &LightData) -> Option<khr::Light> {
    let (type_, color, intensity, range, spot) = match *data {
        LightData::Sky { color, intensity, .. } => (Type::Directional, color, intensity, None, None),
        LightData::Point { color, intensity, radius, .. } => (Type::Point, color, intensity, Some(radius), None),
        LightData::Spot { color, intensity, radius, inner_angle, outer_angle, .. } => (
            Type::Spot,
            color,
            intensity,
            Some(radius),
            Some(khr::Spot { inner_cone_angle: inner_angle, outer_cone_angle: outer_angle }),
        ),
        LightData::Area { .. } => return None,
    };
    Some(khr::Light {
        color,
        extensions: None,
        extras: Default::default(),
        intensity,
        name: Some(light_name(data).to_string()),
        range,
        spot,
        type_: Checked::Valid(type_),
    })
}

/// Stores area lights in the root `extras`, keeping any other keys already there.
fn write_extras(root: &mut Root, lights: &[&LightData]) -> Result<()> {
    let mut extras = match &root.extras {
        Some(raw) => match serde_json::from_str::<Value>(raw.get()) {
            Ok(Value::Object(map)) => map,
            _ => Map::new(),
        },
        None => Map::new(),
    };

    if lights.is_empty() {
        extras.remove(EXTRAS_KEY);
    } else {
        extras.insert(EXTRAS_KEY.to_string(), serde_json::to_value(lights)?);
    }

    root.extras = if extras.is_empty() {
        None
    } else {
        Some(RawValue::from_string(serde_json::to_string(&extras)?)?)
    };
    Ok(())
}
