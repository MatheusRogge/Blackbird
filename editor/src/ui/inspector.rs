#![allow(deprecated)]
use egui::Context;

use crate::state::{EditorState, LightData, NodeEdit, NodeKind, Selection};

pub fn draw(ctx: &Context, state: &mut EditorState) {
    egui::Panel::right("inspector")
        .min_size(240.0)
        .default_size(280.0)
        .show(ctx, |ui| {
            ui.heading("Inspector");
            ui.separator();

            match state.selected.clone() {
                None => {
                    ui.label(egui::RichText::new("Nothing selected.").weak());
                }
                Some(Selection::Light(idx)) => {
                    if let Some(light) = state.lights.get_mut(idx) {
                        draw_light(ui, &mut light.data);
                    }
                }
                Some(Selection::GltfNode(idx)) => {
                    let node_name = state.scene_nodes.iter()
                        .find_map(|n| find_node(n, idx))
                        .map(|n| n.name.clone())
                        .unwrap_or_default();
                    let node_kind = state.scene_nodes.iter()
                        .find_map(|n| find_node(n, idx))
                        .map(|n| n.kind.clone());

                    ui.label(egui::RichText::new(&node_name).strong());
                    ui.separator();

                    if let Some(edit) = state.node_edits.get_mut(&idx) {
                        ui.push_id(idx, |ui| {
                            draw_transform(ui, edit);
                            if let Some(NodeKind::Mesh(_)) = &node_kind {
                                draw_material(ui, edit);
                            }
                            if let Some(NodeKind::Camera) = &node_kind {
                                draw_camera(ui, edit);
                            }
                        });
                    }

                    if let Some(NodeKind::Camera) = &node_kind {
                        ui.separator();
                        let previewing = state.camera_preview_node == Some(idx);
                        let label = if previewing { "Exit Camera View" } else { "Assume Camera View" };
                        if ui.button(label).clicked() {
                            state.camera_preview_node = if previewing { None } else { Some(idx) };
                        }
                    }
                }
                Some(Selection::Mesh(idx)) => {
                    if let Some(doc) = &state.document
                        && let Some(mesh) = doc.meshes().find(|m| m.index() == idx)
                    {
                        ui.label(egui::RichText::new(mesh.name().unwrap_or("(unnamed)")).strong());
                        ui.separator();
                        let prim_count = mesh.primitives().count();
                        ui.label(format!("Primitives: {prim_count}"));
                    }
                }
            }
        });
}

fn draw_light(ui: &mut egui::Ui, data: &mut LightData) {
    ui.label(egui::RichText::new(data.label()).strong());
    ui.separator();

    match data {
        LightData::Sky { color, intensity, direction } => {
            egui::Grid::new("light_grid").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
                ui.label("Color");
                ui.color_edit_button_rgb(color);
                ui.end_row();

                ui.label("Intensity");
                ui.add(egui::DragValue::new(intensity).speed(0.05).range(0.0..=20.0));
                ui.end_row();

                ui.label("Direction");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut direction[0]).speed(0.01).prefix("X "));
                    ui.add(egui::DragValue::new(&mut direction[1]).speed(0.01).prefix("Y "));
                    ui.add(egui::DragValue::new(&mut direction[2]).speed(0.01).prefix("Z "));
                });
                ui.end_row();
            });
        }

        LightData::Point { position, color, intensity, radius } => {
            egui::Grid::new("light_grid").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
                ui.label("Position");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut position[0]).speed(0.1).prefix("X "));
                    ui.add(egui::DragValue::new(&mut position[1]).speed(0.1).prefix("Y "));
                    ui.add(egui::DragValue::new(&mut position[2]).speed(0.1).prefix("Z "));
                });
                ui.end_row();

                ui.label("Color");
                ui.color_edit_button_rgb(color);
                ui.end_row();

                ui.label("Intensity");
                ui.add(egui::DragValue::new(intensity).speed(0.1).range(0.0..=10000.0));
                ui.end_row();

                ui.label("Radius");
                ui.add(egui::DragValue::new(radius).speed(0.1).range(0.0..=1000.0));
                ui.end_row();
            });
        }

        LightData::Spot { position, direction, color, intensity, radius, inner_angle, outer_angle } => {
            egui::Grid::new("light_grid").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
                ui.label("Position");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut position[0]).speed(0.1).prefix("X "));
                    ui.add(egui::DragValue::new(&mut position[1]).speed(0.1).prefix("Y "));
                    ui.add(egui::DragValue::new(&mut position[2]).speed(0.1).prefix("Z "));
                });
                ui.end_row();

                ui.label("Direction");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut direction[0]).speed(0.01).prefix("X "));
                    ui.add(egui::DragValue::new(&mut direction[1]).speed(0.01).prefix("Y "));
                    ui.add(egui::DragValue::new(&mut direction[2]).speed(0.01).prefix("Z "));
                });
                ui.end_row();

                ui.label("Color");
                ui.color_edit_button_rgb(color);
                ui.end_row();

                ui.label("Intensity");
                ui.add(egui::DragValue::new(intensity).speed(0.1).range(0.0..=10000.0));
                ui.end_row();

                ui.label("Radius");
                ui.add(egui::DragValue::new(radius).speed(0.1).range(0.0..=1000.0));
                ui.end_row();

                let mut inner_deg = inner_angle.to_degrees();
                let mut outer_deg = outer_angle.to_degrees();

                ui.label("Inner Angle");
                if ui.add(egui::DragValue::new(&mut inner_deg).speed(0.5).range(0.0..=89.0).suffix("°")).changed() {
                    *inner_angle = inner_deg.to_radians();
                }
                ui.end_row();

                ui.label("Outer Angle");
                if ui.add(egui::DragValue::new(&mut outer_deg).speed(0.5).range(0.0..=90.0).suffix("°")).changed() {
                    *outer_angle = outer_deg.to_radians();
                }
                ui.end_row();
            });
        }

        LightData::Area { position, right, up, color, intensity } => {
            egui::Grid::new("light_grid").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
                ui.label("Position");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut position[0]).speed(0.1).prefix("X "));
                    ui.add(egui::DragValue::new(&mut position[1]).speed(0.1).prefix("Y "));
                    ui.add(egui::DragValue::new(&mut position[2]).speed(0.1).prefix("Z "));
                });
                ui.end_row();

                ui.label("Right");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut right[0]).speed(0.01).prefix("X "));
                    ui.add(egui::DragValue::new(&mut right[1]).speed(0.01).prefix("Y "));
                    ui.add(egui::DragValue::new(&mut right[2]).speed(0.01).prefix("Z "));
                });
                ui.end_row();

                ui.label("Up");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut up[0]).speed(0.01).prefix("X "));
                    ui.add(egui::DragValue::new(&mut up[1]).speed(0.01).prefix("Y "));
                    ui.add(egui::DragValue::new(&mut up[2]).speed(0.01).prefix("Z "));
                });
                ui.end_row();

                ui.label("Color");
                ui.color_edit_button_rgb(color);
                ui.end_row();

                ui.label("Intensity");
                ui.add(egui::DragValue::new(intensity).speed(0.1).range(0.0..=10000.0));
                ui.end_row();
            });
        }
    }
}

fn draw_transform(ui: &mut egui::Ui, edit: &mut NodeEdit) {
    ui.collapsing("Transform", |ui| {
        let euler = quat_to_euler(edit.rotation);
        let mut euler_edit = euler;

        egui::Grid::new("transform_grid")
            .num_columns(2)
            .spacing([12.0, 4.0])
            .show(ui, |ui| {
                ui.label("Translation");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut edit.translation[0]).speed(0.1).prefix("X "));
                    ui.add(egui::DragValue::new(&mut edit.translation[1]).speed(0.1).prefix("Y "));
                    ui.add(egui::DragValue::new(&mut edit.translation[2]).speed(0.1).prefix("Z "));
                });
                ui.end_row();

                ui.label("Rotation");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut euler_edit[0]).speed(0.5).prefix("X ").suffix("°"));
                    ui.add(egui::DragValue::new(&mut euler_edit[1]).speed(0.5).prefix("Y ").suffix("°"));
                    ui.add(egui::DragValue::new(&mut euler_edit[2]).speed(0.5).prefix("Z ").suffix("°"));
                });
                ui.end_row();

                ui.label("Scale");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut edit.scale[0]).speed(0.01).prefix("X "));
                    ui.add(egui::DragValue::new(&mut edit.scale[1]).speed(0.01).prefix("Y "));
                    ui.add(egui::DragValue::new(&mut edit.scale[2]).speed(0.01).prefix("Z "));
                });
                ui.end_row();
            });

        if euler_edit != euler {
            edit.rotation = euler_to_quat(euler_edit);
        }
    });
}

fn draw_material(ui: &mut egui::Ui, edit: &mut NodeEdit) {
    ui.collapsing("Material", |ui| {
        egui::Grid::new("material_grid")
            .num_columns(2)
            .spacing([12.0, 4.0])
            .show(ui, |ui| {
                let color = edit.base_color.get_or_insert([1.0, 1.0, 1.0, 1.0]);
                ui.label("Base Color");
                ui.horizontal(|ui| {
                    let mut rgb = [color[0], color[1], color[2]];
                    ui.color_edit_button_rgb(&mut rgb);
                    color[0] = rgb[0];
                    color[1] = rgb[1];
                    color[2] = rgb[2];
                    ui.add(egui::DragValue::new(&mut color[3]).speed(0.01).range(0.0..=1.0).prefix("A "));
                });
                ui.end_row();

                let metallic = edit.metallic_factor.get_or_insert(0.0);
                ui.label("Metallic");
                ui.add(egui::Slider::new(metallic, 0.0..=1.0));
                ui.end_row();

                let roughness = edit.roughness_factor.get_or_insert(1.0);
                ui.label("Roughness");
                ui.add(egui::Slider::new(roughness, 0.0..=1.0));
                ui.end_row();

                let emissive = edit.emissive_factor.get_or_insert([0.0; 3]);
                ui.label("Emissive");
                ui.color_edit_button_rgb(emissive);
                ui.end_row();
            });
    });
}

fn draw_camera(ui: &mut egui::Ui, edit: &mut NodeEdit) {
    ui.collapsing("Camera", |ui| {
        egui::Grid::new("camera_grid")
            .num_columns(2)
            .spacing([12.0, 4.0])
            .show(ui, |ui| {
                let fov = edit.fov_y_deg.get_or_insert(45.0);
                ui.label("FOV Y");
                ui.add(egui::DragValue::new(fov).speed(0.5).range(1.0..=170.0).suffix("°"));
                ui.end_row();

                let near = edit.near_plane.get_or_insert(0.01);
                ui.label("Near");
                ui.add(egui::DragValue::new(near).speed(0.001).range(0.0001..=1000.0));
                ui.end_row();

                let far = edit.far_plane.get_or_insert(1000.0);
                ui.label("Far");
                ui.add(egui::DragValue::new(far).speed(1.0).range(1.0..=1_000_000.0));
                ui.end_row();
            });
    });
}

fn find_node(
    node: &crate::state::SceneNode,
    idx: usize,
) -> Option<&crate::state::SceneNode> {
    if node.index == idx {
        return Some(node);
    }
    node.children.iter().find_map(|c| find_node(c, idx))
}

fn quat_to_euler(q: [f32; 4]) -> [f32; 3] {
    let [x, y, z, w] = q;
    let roll = f32::atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
    let sin_pitch = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0);
    let pitch = f32::asin(sin_pitch);
    let yaw = f32::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
    [roll.to_degrees(), pitch.to_degrees(), yaw.to_degrees()]
}

fn euler_to_quat(euler: [f32; 3]) -> [f32; 4] {
    let [roll, pitch, yaw] = euler.map(f32::to_radians);
    let (cr, sr) = (f32::cos(roll * 0.5), f32::sin(roll * 0.5));
    let (cp, sp) = (f32::cos(pitch * 0.5), f32::sin(pitch * 0.5));
    let (cy, sy) = (f32::cos(yaw * 0.5), f32::sin(yaw * 0.5));
    [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]
}
