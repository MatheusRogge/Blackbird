#![allow(deprecated)]
use egui::Context;

use crate::state::{EditorState, SceneNode, Selection};

pub fn draw(ctx: &Context, state: &mut EditorState) {
    egui::Panel::left("hierarchy")
        .min_size(180.0)
        .default_size(220.0)
        .show(ctx, |ui| {
            ui.heading("Scene");
            ui.separator();

            let sky_selected = state.selected == Some(Selection::SkyLight);
            if ui.selectable_label(sky_selected, "☀ Sky Light").clicked() {
                state.selected = Some(Selection::SkyLight);
            }

            if state.scene_nodes.is_empty() && state.document.is_none() {
                ui.add_space(8.0);
                ui.label(egui::RichText::new("Open a GLTF file to begin.").weak());
                return;
            }

            ui.separator();

            let nodes = state.scene_nodes.clone();
            let mut new_selection = state.selected.clone();

            for node in &nodes {
                draw_node(ui, node, &mut new_selection);
            }

            if new_selection != state.selected {
                if let Some(Selection::GltfNode(idx)) = &new_selection {
                    state.ensure_node_edit(*idx);
                }
                state.selected = new_selection;
            }
        });
}

fn draw_node(
    ui: &mut egui::Ui,
    node: &SceneNode,
    selection: &mut Option<Selection>,
) {
    let (icon, suffix) = match &node.kind {
        crate::state::NodeKind::Mesh(mesh) if !mesh.is_empty() && *mesh != node.name =>
            ("▲", format!(" [{mesh}]")),
        crate::state::NodeKind::Mesh(_) => ("▲", String::new()),
        crate::state::NodeKind::Camera => ("📷", String::new()),
        crate::state::NodeKind::Empty => ("○", String::new()),
    };
    let label = format!("{icon} {}{suffix}", node.name);
    let selected = *selection == Some(Selection::GltfNode(node.index));

    if node.children.is_empty() {
        if ui.selectable_label(selected, &label).clicked() {
            *selection = Some(Selection::GltfNode(node.index));
        }
    } else {
        let header = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            ui.make_persistent_id(node.index),
            true,
        );
        header.show_header(ui, |ui| {
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
