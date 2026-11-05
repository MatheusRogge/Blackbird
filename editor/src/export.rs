use std::path::Path;

use anyhow::{Context, Result};
use engine::gltf::camera::{MAIN_CAMERA_KEY, PHYSICAL_KEY};
use gltf_api::json::camera::Perspective;
use gltf_api::json::extras::Extras;
use gltf_api::json::material::{EmissiveFactor, PbrBaseColorFactor, StrengthFactor};
use gltf_api::json::scene::UnitQuaternion;
use gltf_api::json::Root;
use serde_json::value::RawValue;
use serde_json::{Map, Value};

use crate::state::{CameraEdit, EditorState};

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

    crate::lights::write_lights(&mut root, state)?;
    write_main_camera(&mut root, state.main_camera)?;

    let json = serde_json::to_string_pretty(&root).context("JSON serialization failed")?;
    std::fs::write(path, json).context("writing file")?;
    Ok(())
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
