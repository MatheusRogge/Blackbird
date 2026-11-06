use egui::Ui;

use crate::state::{CameraEdit, EditorState, LightData, LightDataExt, NodeEdit, NodeKind, PhysicalCamera, SceneNode, Selection};

const UNNAMED: &str = "(unnamed)";
const DEFAULT_FAR: f32 = 1000.0;
const ASPECT_PRESETS: &[(&str, f32)] = &[("16:9", 16.0 / 9.0), ("16:10", 1.6), ("4:3", 4.0 / 3.0), ("21:9", 21.0 / 9.0), ("1:1", 1.0)];

pub fn draw(root: &mut Ui, state: &mut EditorState) {
    egui::Panel::right("inspector")
        .min_size(240.0)
        .default_size(280.0)
        .show(root, |ui| {
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
                    let node_kind = state.scene_nodes.iter()
                        .find_map(|n| find_node(n, idx))
                        .map(|n| n.kind.clone());

                    if let Some(edit) = state.node_edits.get_mut(&idx) {
                        let name = egui::TextEdit::singleline(&mut edit.name).hint_text(UNNAMED);
                        if ui.add(name).changed() {
                            rename_node(&mut state.scene_nodes, idx, &edit.name);
                        }
                    }
                    if let Some(mesh) = state.document.as_ref()
                        .and_then(|doc| doc.nodes().nth(idx))
                        .and_then(|node| node.mesh())
                    {
                        let primitives = mesh.primitives().count();
                        ui.label(egui::RichText::new(format!(
                            "Mesh {} · {primitives} primitive{}",
                            mesh.name().unwrap_or("(unnamed)"),
                            if primitives == 1 { "" } else { "s" },
                        )).weak());
                    }
                    ui.separator();

                    if let Some(edit) = state.node_edits.get_mut(&idx) {
                        ui.push_id(idx, |ui| {
                            draw_transform(ui, edit);
                            if let Some(NodeKind::Mesh(_)) = &node_kind {
                                draw_material(ui, edit);
                            }
                            if let Some(camera) = &mut edit.camera {
                                let mut is_main = state.main_camera == Some(idx);
                                draw_camera(ui, camera, &mut is_main, state.camera_settings.exposure_stops);
                                if is_main {
                                    state.main_camera = Some(idx);
                                } else if state.main_camera == Some(idx) {
                                    state.main_camera = None;
                                }
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
                ui.add(egui::DragValue::new(intensity).speed(0.05).range(0.0..=200_000.0).suffix(" lx"));
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

        LightData::Point { position, color, intensity, radius, source_radius } => {
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
                ui.add(egui::DragValue::new(intensity).speed(0.1).range(0.0..=1_000_000.0).suffix(" cd"));
                ui.end_row();

                ui.label("Radius");
                ui.add(egui::DragValue::new(radius).speed(0.1).range(0.0..=1000.0));
                ui.end_row();

                ui.label("Source radius");
                ui.add(egui::DragValue::new(source_radius).speed(0.005).range(0.0..=10.0).suffix(" m"));
                ui.end_row();
            });
        }

        LightData::Spot { position, direction, color, intensity, radius, source_radius, inner_angle, outer_angle } => {
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
                ui.add(egui::DragValue::new(intensity).speed(0.1).range(0.0..=1_000_000.0).suffix(" cd"));
                ui.end_row();

                ui.label("Radius");
                ui.add(egui::DragValue::new(radius).speed(0.1).range(0.0..=1000.0));
                ui.end_row();

                ui.label("Source radius");
                ui.add(egui::DragValue::new(source_radius).speed(0.005).range(0.0..=10.0).suffix(" m"));
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
                ui.add(egui::DragValue::new(intensity).speed(0.1).range(0.0..=1_000_000.0).suffix(" nt"));
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

fn draw_camera(ui: &mut egui::Ui, camera: &mut CameraEdit, is_main: &mut bool, project_stops: f32) {
    ui.collapsing("Camera", |ui| {
        ui.checkbox(is_main, "Player camera")
            .on_hover_text("The camera the player renders this scene from. Other cameras are removed when it plays.");

        egui::Grid::new("camera_grid")
            .num_columns(2)
            .spacing([12.0, 4.0])
            .show(ui, |ui| {
                ui.label("FOV Y");
                ui.add(egui::DragValue::new(&mut camera.fov_y_deg).speed(0.5).range(1.0..=170.0).suffix("°"));
                ui.end_row();

                ui.label("Near");
                ui.add(egui::DragValue::new(&mut camera.near).speed(0.001).range(0.0001..=1000.0));
                ui.end_row();

                ui.label("Far");
                ui.horizontal(|ui| {
                    let mut infinite = camera.far.is_none();
                    if let Some(far) = &mut camera.far {
                        ui.add(egui::DragValue::new(far).speed(1.0).range(camera.near..=1_000_000.0));
                    }
                    if ui.checkbox(&mut infinite, "Infinite").changed() {
                        camera.far = (!infinite).then_some(DEFAULT_FAR.max(camera.near * 2.0));
                    }
                });
                ui.end_row();

                ui.label("Aspect");
                ui.horizontal(|ui| {
                    let selected = match camera.aspect {
                        None => "Viewport".to_string(),
                        Some(a) => ASPECT_PRESETS
                            .iter()
                            .find(|(_, r)| (r - a).abs() < 1e-3)
                            .map_or_else(|| "Custom".to_string(), |(name, _)| name.to_string()),
                    };
                    egui::ComboBox::from_id_salt("camera_aspect").selected_text(selected).show_ui(ui, |ui| {
                        ui.selectable_value(&mut camera.aspect, None, "Viewport");
                        for &(name, ratio) in ASPECT_PRESETS {
                            ui.selectable_value(&mut camera.aspect, Some(ratio), name);
                        }
                    });
                    if let Some(aspect) = &mut camera.aspect {
                        ui.add(egui::DragValue::new(aspect).speed(0.01).range(0.1..=10.0).max_decimals(3));
                    }
                });
                ui.end_row();
            });

        ui.add_space(4.0);
        let mut physical = camera.physical.is_some();
        if ui.checkbox(&mut physical, "Physical exposure").changed() {
            camera.physical = physical.then(PhysicalCamera::default);
        }
        match &mut camera.physical {
            Some(p) => {
                egui::Grid::new("camera_exposure_grid")
                    .num_columns(2)
                    .spacing([12.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Aperture");
                        ui.add(egui::DragValue::new(&mut p.aperture).speed(0.05).range(0.7..=64.0).prefix("f/"));
                        ui.end_row();

                        ui.label("Shutter");
                        let mut denominator = 1.0 / p.shutter_speed;
                        let shutter = egui::DragValue::new(&mut denominator).speed(1.0).range(1.0..=8000.0).prefix("1/").suffix(" s");
                        if ui.add(shutter).changed() {
                            p.shutter_speed = 1.0 / denominator;
                        }
                        ui.end_row();

                        ui.label("ISO");
                        ui.add(egui::DragValue::new(&mut p.iso).speed(10.0).range(25.0..=102_400.0));
                        ui.end_row();
                    });
                ui.label(egui::RichText::new(format!("EV₁₀₀ {:.1}", p.ev100())).weak());
                ui.label(egui::RichText::new("Visible through Assume Camera View and in the player.").small().weak());
            }
            None => {
                ui.label(egui::RichText::new(format!("Uses the project exposure ({project_stops:+.1} EV, Camera menu)")).weak());
            }
        }
    });
}

fn rename_node(nodes: &mut [SceneNode], idx: usize, name: &str) {
    for node in nodes {
        if node.index == idx {
            node.name = if name.is_empty() { UNNAMED.to_string() } else { name.to_string() };
            return;
        }
        rename_node(&mut node.children, idx, name);
    }
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
