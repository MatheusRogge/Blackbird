use std::path::Path;

use anyhow::{Context, Result};
use gltf_api::json::scene::UnitQuaternion;
use gltf_api::json::material::{EmissiveFactor, PbrBaseColorFactor, StrengthFactor};

use crate::state::EditorState;

pub fn write_gltf(state: &EditorState, path: &Path) -> Result<()> {
    let doc = state.document.as_ref().context("no document loaded")?;
    let mut root = doc.as_json().clone();

    for (&node_idx, edit) in &state.node_edits {
        let Some(node) = root.nodes.get_mut(node_idx) else { continue };

        node.translation = Some(edit.translation);
        node.rotation = Some(UnitQuaternion(edit.rotation));
        node.scale = Some(edit.scale);
        node.matrix = None;

        let mesh_idx = match node.mesh {
            Some(ref m) => m.value(),
            None => continue,
        };

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

    let json = serde_json::to_string_pretty(&root).context("JSON serialization failed")?;
    std::fs::write(path, json).context("writing file")?;
    Ok(())
}
