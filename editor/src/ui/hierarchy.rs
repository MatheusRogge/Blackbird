use std::collections::{HashMap, HashSet};

use egui::Ui;

use crate::state::{EditorState, LightDataExt, LightKind, NodeKind, SceneNode, Selection};

pub fn draw(root: &mut Ui, state: &mut EditorState) {
    egui::Panel::left("hierarchy")
        .min_size(180.0)
        .default_size(220.0)
        .show(root, |ui| {
            ui.heading("Scene");
            ui.separator();

            // Right-click the panel background to add lights or make this the default scene.
            let bg_response = ui.interact(
                ui.available_rect_before_wrap(),
                ui.scope_id().with("scene_ctx"),
                egui::Sense::click(),
            );
            bg_response.context_menu(|ui| {
                if let Some(scene) = state.scene_in_project()
                    && let Some(project) = state.project.as_mut()
                {
                    let mut is_default = project.default_scene.as_ref() == Some(&scene);
                    if ui.checkbox(&mut is_default, "Default Scene").changed() {
                        project.default_scene = is_default.then_some(scene);
                        ui.close();
                    }
                    ui.separator();
                }
                ui.menu_button("Add Light", |ui| {
                    if ui.button("☀ Sky Light").clicked() {
                        state.add_light_requested = Some(LightKind::Sky);
                        ui.close();
                    }
                    if ui.button("💡 Point Light").clicked() {
                        state.add_light_requested = Some(LightKind::Point);
                        ui.close();
                    }
                    if ui.button("◎ Spot Light").clicked() {
                        state.add_light_requested = Some(LightKind::Spot);
                        ui.close();
                    }
                    if ui.button("▪ Area Light").clicked() {
                        state.add_light_requested = Some(LightKind::Area);
                        ui.close();
                    }
                });
            });

            if state.document.is_none() && state.lights.is_empty() {
                ui.add_space(8.0);
                ui.label(egui::RichText::new("Open a GLTF file to begin.").weak());
                return;
            }

            let nodes = state.scene_nodes.clone();
            let mut new_selection = state.selected.clone();
            let node_lights: HashMap<usize, (usize, &'static str)> = state.lights.iter()
                .enumerate()
                .filter_map(|(idx, light)| Some((light.node?, (idx, light.data.icon()))))
                .collect();

            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                for node in &nodes {
                    draw_node(ui, node, &node_lights, &mut new_selection);
                }
                // Lights that have no row of their own in the tree: added in the editor, area
                // lights from `extras`, or carried by a mesh/camera node or a node outside the
                // default scene.
                let mut light_rows = HashSet::new();
                for node in &nodes {
                    collect_light_nodes(node, &mut light_rows);
                }
                let unlisted = state.lights.iter().enumerate()
                    .filter(|(_, l)| l.node.is_none_or(|n| !light_rows.contains(&n)));
                for (idx, light) in unlisted {
                    let target = Selection::Light(idx);
                    if ui.selectable_label(new_selection.as_ref() == Some(&target), light.data.label()).clicked() {
                        new_selection = Some(target);
                    }
                }
            });

            if new_selection != state.selected {
                if let Some(Selection::GltfNode(idx)) = &new_selection {
                    state.ensure_node_edit(*idx);
                }
                state.selected = new_selection;
            }
        });
}

fn collect_light_nodes(node: &SceneNode, out: &mut HashSet<usize>) {
    if matches!(node.kind, NodeKind::Light) {
        out.insert(node.index);
    }
    for child in &node.children {
        collect_light_nodes(child, out);
    }
}

/// `node_lights` maps a light-carrying node to its `state.lights` index, so clicking the node
/// selects the light rather than a node whose transform the light would overwrite on save.
fn draw_node(
    ui: &mut egui::Ui,
    node: &SceneNode,
    node_lights: &HashMap<usize, (usize, &'static str)>,
    selection: &mut Option<Selection>,
) {
    let light = node_lights.get(&node.index).filter(|_| matches!(node.kind, NodeKind::Light));
    let label = match (&node.kind, light) {
        (NodeKind::Light, Some((_, icon))) => format!("{icon} {}", node.name),
        (NodeKind::Mesh(mesh), _) if !mesh.is_empty() && *mesh != node.name =>
            format!("⏶ {} [{mesh}]", node.name),
        (NodeKind::Mesh(_), _) => format!("⏶ {}", node.name),
        (NodeKind::Camera, _)  => format!("📷 {}", node.name),
        (NodeKind::Light | NodeKind::Empty, _) => format!("○ {}", node.name),
    };
    let target = match light {
        Some(&(idx, _)) => Selection::Light(idx),
        None => Selection::GltfNode(node.index),
    };
    let selected = selection.as_ref() == Some(&target);

    if node.children.is_empty() {
        if ui.selectable_label(selected, &label).clicked() {
            *selection = Some(target);
        }
    } else {
        egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            ui.make_persistent_id(node.index),
            true,
        )
        .show_header(ui, |ui| {
            if ui.selectable_label(selected, &label).clicked() {
                *selection = Some(target.clone());
            }
        })
        .body(|ui| {
            ui.indent(node.index, |ui| {
                for child in &node.children {
                    draw_node(ui, child, node_lights, selection);
                }
            });
        });
    }
}
