#![allow(deprecated)]
use egui::Context;

use crate::state::{EditorState, LightKind, NodeKind, SceneNode, Selection};

pub fn draw(ctx: &Context, state: &mut EditorState) {
    egui::Panel::left("hierarchy")
        .min_size(180.0)
        .default_size(220.0)
        .show(ctx, |ui| {
            ui.heading("Scene");
            ui.separator();

            // Right-click the panel background to add lights.
            let bg_response = ui.interact(
                ui.available_rect_before_wrap(),
                ui.id().with("scene_ctx"),
                egui::Sense::click(),
            );
            bg_response.context_menu(|ui| {
                ui.menu_button("Add Light", |ui| {
                    if ui.button("☀ Sky Light").clicked() {
                        state.add_light_requested = Some(LightKind::Sky);
                        ui.close_menu();
                    }
                    if ui.button("● Point Light").clicked() {
                        state.add_light_requested = Some(LightKind::Point);
                        ui.close_menu();
                    }
                    if ui.button("◎ Spot Light").clicked() {
                        state.add_light_requested = Some(LightKind::Spot);
                        ui.close_menu();
                    }
                    if ui.button("▪ Area Light").clicked() {
                        state.add_light_requested = Some(LightKind::Area);
                        ui.close_menu();
                    }
                });
            });

            if !state.lights.is_empty() {
                egui::collapsing_header::CollapsingState::load_with_default_open(
                    ctx,
                    ui.make_persistent_id("lights_section"),
                    true,
                )
                .show_header(ui, |ui| { ui.label("Lights"); })
                .body(|ui| {
                    for (idx, light) in state.lights.iter().enumerate() {
                        let selected = state.selected == Some(Selection::Light(idx));
                        if ui.selectable_label(selected, light.data.label()).clicked() {
                            state.selected = Some(Selection::Light(idx));
                        }
                    }
                });
            }

            if state.scene_nodes.is_empty() && state.document.is_none() {
                ui.add_space(8.0);
                ui.label(egui::RichText::new("Open a GLTF file to begin.").weak());
                return;
            }

            let mesh_entries: Vec<(usize, String)> = state.document.as_ref()
                .map(|doc| doc.meshes()
                    .map(|m| (m.index(), m.name().unwrap_or("(unnamed)").to_string()))
                    .collect())
                .unwrap_or_default();

            if !mesh_entries.is_empty() {
                egui::collapsing_header::CollapsingState::load_with_default_open(
                    ctx,
                    ui.make_persistent_id("meshes_section"),
                    true,
                )
                .show_header(ui, |ui| { ui.label("Meshes"); })
                .body(|ui| {
                    for (idx, name) in &mesh_entries {
                        let selected = state.selected == Some(Selection::Mesh(*idx));
                        if ui.selectable_label(selected, format!("▲ {name}")).clicked() {
                            state.selected = Some(Selection::Mesh(*idx));
                        }
                    }
                });
            }

            let nodes = state.scene_nodes.clone();
            let mut new_selection = state.selected.clone();

            if !nodes.is_empty() {
                egui::collapsing_header::CollapsingState::load_with_default_open(
                    ctx,
                    ui.make_persistent_id("nodes_section"),
                    true,
                )
                .show_header(ui, |ui| { ui.label("Nodes"); })
                .body(|ui| {
                    for node in &nodes {
                        draw_node(ui, node, &mut new_selection);
                    }
                });
            }

            if new_selection != state.selected {
                if let Some(Selection::GltfNode(idx)) = &new_selection {
                    state.ensure_node_edit(*idx);
                }
                state.selected = new_selection;
            }
        });
}

fn draw_node(ui: &mut egui::Ui, node: &SceneNode, selection: &mut Option<Selection>) {
    let label = match &node.kind {
        NodeKind::Mesh(mesh) if !mesh.is_empty() && *mesh != node.name =>
            format!("▲ {} [{mesh}]", node.name),
        NodeKind::Mesh(_) => format!("▲ {}", node.name),
        NodeKind::Camera  => format!("📷 {}", node.name),
        NodeKind::Empty   => format!("○ {}", node.name),
    };
    let selected = *selection == Some(Selection::GltfNode(node.index));

    if node.children.is_empty() {
        if ui.selectable_label(selected, &label).clicked() {
            *selection = Some(Selection::GltfNode(node.index));
        }
    } else {
        egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            ui.make_persistent_id(node.index),
            true,
        )
        .show_header(ui, |ui| {
            if ui.selectable_label(selected, &label).clicked() {
                *selection = Some(Selection::GltfNode(node.index));
            }
        })
        .body(|ui| {
            ui.indent(node.index, |ui| {
                for child in &node.children {
                    draw_node(ui, child, selection);
                }
            });
        });
    }
}
