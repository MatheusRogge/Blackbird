use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result};
use engine::gltf::camera::{MAIN_CAMERA_KEY, PHYSICAL_KEY};
use engine::gltf::particles::{self, ParticleData};
use gltf_api::json::camera::Perspective;
use gltf_api::json::extras::Extras;
use gltf_api::json::material::{EmissiveFactor, PbrBaseColorFactor, StrengthFactor};
use gltf_api::json::scene::UnitQuaternion;
use gltf_api::json::Root;
use serde_json::value::RawValue;
use serde_json::{Map, Value};

use crate::state::{CameraEdit, EditorState, ObjectKind};

pub fn write_gltf(state: &EditorState, path: &Path) -> Result<()> {
    let doc = state.document.as_ref().context("no document loaded")?;
    let mut root = doc.as_json().clone();

    for (&node_idx, edit) in &state.node_edits {
        let Some(node) = root.nodes.get_mut(node_idx) else { continue };

        node.name = (!edit.name.is_empty()).then(|| edit.name.clone());
        node.translation = Some(edit.translation);
        node.rotation = Some(UnitQuaternion(edit.rotation));
        node.scale = Some(edit.scale);
        node.matrix = None;

        if let Some(particles) = &edit.particles {
            write_particles(&mut node.extras, particles)?;
        }

        let cam_idx = node.camera.as_ref().map(|c| c.value());
        let mesh_idx = node.mesh.as_ref().map(|m| m.value());

        if let (Some(camera), Some(cam_idx)) = (&edit.camera, cam_idx) {
            write_camera(&mut root, cam_idx, camera)?;
        }
        let Some(mesh_idx) = mesh_idx else { continue };

        let mat_indices: Vec<usize> = root.meshes
            .get(mesh_idx)
            .map(|mesh| {
                mesh.primitives.iter()
                    .filter_map(|p| p.material.as_ref().map(|m| m.value()))
                    .collect()
            })
            .unwrap_or_default();

        for mat_idx in mat_indices {
            let Some(mat) = root.materials.get_mut(mat_idx) else { continue };
            let pbr = &mut mat.pbr_metallic_roughness;
            if let Some(c) = edit.base_color {
                pbr.base_color_factor = PbrBaseColorFactor(c);
            }
            if let Some(m) = edit.metallic_factor {
                pbr.metallic_factor = StrengthFactor(m);
            }
            if let Some(r) = edit.roughness_factor {
                pbr.roughness_factor = StrengthFactor(r);
            }
            if let Some(em) = edit.emissive_factor {
                mat.emissive_factor = EmissiveFactor(em);
            }
        }
    }

    detach_nodes(&mut root, &state.deleted_nodes);
    crate::lights::write_lights(&mut root, state)?;
    let clips = crate::animation::to_clip_data(&state.clips, &state.lights);
    let clips = if clips.is_empty() { None } else { Some(serde_json::to_value(clips)?) };
    set_extra(&mut root.extras, engine::gltf::animation::EXTRAS_KEY, clips)?;
    write_main_camera(&mut root, state.main_camera)?;
    write_objects(&mut root, state, path)?;

    let json = serde_json::to_string_pretty(&root).context("JSON serialization failed")?;
    std::fs::write(path, json).context("writing file")?;
    Ok(())
}

/// Writes `data` over the node's `blackbird_particles`, keeping keys the editor doesn't model
/// (e.g. `notes`). A field `data` leaves unset (`null`) is removed.
fn write_particles(extras: &mut Extras, data: &ParticleData) -> Result<()> {
    let existing = extras
        .as_ref()
        .and_then(|raw| serde_json::from_str::<Value>(raw.get()).ok())
        .and_then(|mut v| v.get_mut(particles::EXTRAS_KEY).map(Value::take));
    let mut merged = match existing {
        Some(Value::Object(map)) => map,
        _ => Map::new(),
    };
    let Value::Object(fields) = serde_json::to_value(data)? else { anyhow::bail!("particle data isn't an object") };
    for (key, value) in fields {
        if value.is_null() {
            merged.remove(&key);
        } else {
            merged.insert(key, value);
        }
    }
    set_extra(extras, particles::EXTRAS_KEY, Some(Value::Object(merged)))
}

/// Takes deleted nodes out of every scene and parent. They stay in the node list (removing them
/// would renumber every index), unreferenced.
fn detach_nodes(root: &mut Root, deleted: &HashSet<usize>) {
    for scene in &mut root.scenes {
        scene.nodes.retain(|n| !deleted.contains(&n.value()));
    }
    for node in &mut root.nodes {
        if let Some(children) = &mut node.children {
            children.retain(|n| !deleted.contains(&n.value()));
            if children.is_empty() {
                node.children = None;
            }
        }
    }
}

/// Stores the editor's objects in the root `extras`, with model paths relative to `path`.
fn write_objects(root: &mut Root, state: &EditorState, path: &Path) -> Result<()> {
    let objects: Vec<_> = state
        .objects
        .iter()
        .map(|object| {
            let mut data = object.data.clone();
            if let ObjectKind::Model { uri } = &mut data.kind {
                *uri = relative_uri(path, Path::new(uri.as_str()));
            }
            data
        })
        .collect();
    let value = if objects.is_empty() { None } else { Some(serde_json::to_value(objects)?) };
    set_extra(&mut root.extras, engine::gltf::objects::EXTRAS_KEY, value)
}

/// `target` as a `/`-separated path relative to the directory holding `file`, or as given when
/// they share no root (another drive).
fn relative_uri(file: &Path, target: &Path) -> String {
    let base: Vec<Component> = file.parent().unwrap_or(Path::new("")).components().collect();
    let target_parts: Vec<Component> = target.components().collect();
    let common = base.iter().zip(&target_parts).take_while(|(a, b)| a == b).count();
    if common == 0 && target.has_root() {
        return target.to_string_lossy().replace('\\', "/");
    }
    let mut relative = PathBuf::new();
    for _ in common..base.len() {
        relative.push("..");
    }
    for part in &target_parts[common..] {
        relative.push(part);
    }
    relative.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect::<Vec<_>>().join("/")
}

fn write_camera(root: &mut Root, cam_idx: usize, edit: &CameraEdit) -> Result<()> {
    let Some(camera) = root.cameras.get_mut(cam_idx) else { return Ok(()) };
    let Some(perspective) = &mut camera.perspective else { return Ok(()) };
    *perspective = Perspective {
        aspect_ratio: edit.aspect,
        yfov: edit.fov_y_deg.to_radians(),
        zfar: edit.far,
        znear: edit.near,
        ..perspective.clone()
    };
    let physical = edit.physical.map(serde_json::to_value).transpose()?;
    set_extra(&mut camera.extras, PHYSICAL_KEY, physical)
}

/// Marks the main camera on the default scene (or the first one, as the loader picks).
fn write_main_camera(root: &mut Root, node: Option<usize>) -> Result<()> {
    let scene_idx = root.scene.map_or(0, |s| s.value());
    let Some(scene) = root.scenes.get_mut(scene_idx) else { return Ok(()) };
    set_extra(&mut scene.extras, MAIN_CAMERA_KEY, node.map(Value::from))
}

/// Sets (or with `None` removes) one key of an `extras` object, keeping the others.
pub fn set_extra(extras: &mut Extras, key: &str, value: Option<Value>) -> Result<()> {
    let mut map = match extras {
        Some(raw) => match serde_json::from_str::<Value>(raw.get()) {
            Ok(Value::Object(map)) => map,
            _ => Map::new(),
        },
        None => Map::new(),
    };
    match value {
        Some(value) => map.insert(key.to_string(), value),
        None => map.remove(key),
    };
    *extras = if map.is_empty() { None } else { Some(RawValue::from_string(serde_json::to_string(&map)?)?) };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_detaches_deleted_nodes_and_writes_objects() {
        use crate::state::{EditorObject, ObjectData, Shape, ShapeMaterial};

        let json = r#"{
            "asset": { "version": "2.0" },
            "scene": 0,
            "scenes": [{ "nodes": [0, 1] }],
            "nodes": [{ "name": "A", "children": [2] }, { "name": "B" }, { "name": "C" }]
        }"#;
        let doc = gltf_api::Gltf::from_slice(json.as_bytes()).unwrap().document;
        let dir = std::env::temp_dir().join(format!("blackbird_export_test_{}", std::process::id()));
        let scene_path = dir.join("scenes").join("level.gltf");
        std::fs::create_dir_all(scene_path.parent().unwrap()).unwrap();

        let shared = EditorState::new();
        let mut state = shared.lock().unwrap();
        state.document = Some(doc);
        state.deleted_nodes.extend([1, 2]);
        let object = |kind| ObjectData { name: "o".into(), translation: [1.0, 0.0, 0.0], rotation: [0.0, 0.0, 0.0, 1.0], scale: [1.0; 3], kind };
        state.objects = vec![
            EditorObject { id: 1, data: object(ObjectKind::Shape { shape: Shape::Cube, material: ShapeMaterial::default() }) },
            EditorObject {
                id: 2,
                data: object(ObjectKind::Model { uri: dir.join("assets/models/chair.gltf").to_string_lossy().into_owned() }),
            },
        ];
        write_gltf(&state, &scene_path).unwrap();

        let saved = gltf_api::Gltf::open(&scene_path).unwrap().document;
        let scene = saved.default_scene().unwrap();
        assert_eq!(scene.nodes().map(|n| n.index()).collect::<Vec<_>>(), vec![0]);
        assert_eq!(saved.nodes().next().unwrap().children().count(), 0);
        let objects = engine::gltf::objects::read_objects(&saved).unwrap();
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[1].model_uri(), Some("../assets/models/chair.gltf"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn save_writes_particle_edits_and_keeps_unknown_keys() {
        let json = r#"{
            "asset": { "version": "2.0" },
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "name": "Dust", "extras": { "other": 1, "blackbird_particles": {
                "shape": "box", "shear_along": [0.2, -0.9, 0.3], "count": 10, "spawn_rate": 1.0,
                "lifetime": [1.0, 2.0], "size": [0.01, 0.02], "notes": "keep me" } } }]
        }"#;
        let doc = gltf_api::Gltf::from_slice(json.as_bytes()).unwrap().document;
        let dir = std::env::temp_dir().join(format!("blackbird_particles_test_{}", std::process::id()));
        let scene_path = dir.join("level.gltf");
        std::fs::create_dir_all(&dir).unwrap();

        let shared = EditorState::new();
        let mut state = shared.lock().unwrap();
        state.document = Some(doc);
        state.ensure_node_edit(0);
        let edit = state.node_edits.get_mut(&0).unwrap();
        let particles = edit.particles.as_mut().expect("node has an emitter");
        particles.count = 42;
        particles.shear_along = None;
        write_gltf(&state, &scene_path).unwrap();

        let saved: Value = serde_json::from_str(&std::fs::read_to_string(&scene_path).unwrap()).unwrap();
        let extras = &saved["nodes"][0]["extras"];
        assert_eq!(extras["other"], 1);
        let written = &extras[particles::EXTRAS_KEY];
        assert_eq!(written["count"], 42);
        assert_eq!(written["notes"], "keep me");
        assert!(written.get("shear_along").is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn save_keeps_animation_clips() {
        let json = r#"{
            "asset": { "version": "2.0" },
            "scene": 0,
            "scenes": [{ "nodes": [] }],
            "extras": { "blackbird_animations": [{ "name": "Flicker", "tracks": [{
                "pointer": "/materials/0/emissiveFactor", "keys": [{ "time": 0.0, "value": [1, 1, 1] }] }] }] }
        }"#;
        let doc = gltf_api::Gltf::from_slice(json.as_bytes()).unwrap().document;
        let dir = std::env::temp_dir().join(format!("blackbird_animation_test_{}", std::process::id()));
        let scene_path = dir.join("level.gltf");
        std::fs::create_dir_all(&dir).unwrap();

        let shared = EditorState::new();
        let mut state = shared.lock().unwrap();
        state.clips = crate::animation::load_clips(&doc, &[], &state.lights);
        state.document = Some(doc);
        write_gltf(&state, &scene_path).unwrap();

        let saved: Value = serde_json::from_str(&std::fs::read_to_string(&scene_path).unwrap()).unwrap();
        assert_eq!(saved["extras"]["blackbird_animations"][0]["name"], "Flicker");
        assert_eq!(saved["extras"]["blackbird_animations"][0]["tracks"][0]["keys"][0]["value"], serde_json::json!([1.0, 1.0, 1.0]));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn relative_uri_climbs_out_of_the_scene_folder() {
        let scene = Path::new("/project/scenes/level.gltf");
        assert_eq!(relative_uri(scene, Path::new("/project/assets/models/chair.gltf")), "../assets/models/chair.gltf");
        assert_eq!(relative_uri(scene, Path::new("/project/scenes/props/lamp.glb")), "props/lamp.glb");
    }
}
