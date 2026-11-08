use egui::Ui;

use std::path::Path;

use engine::gltf::objects::PLAYER_EYE_HEIGHT;
use engine::gltf::particles::{Collision, Lighting, ParticleData, ShapeKind};
use engine::rendering::pass::particle_sim::MAX_PARTICLES;

use engine::gltf::animation::{LightProperty, ParticleProperty, Target};

use super::animation::KeyButtons;
use crate::animation::{self, EditorTarget};
use crate::math::{euler_to_quat, quat_to_euler};
use crate::state::{
    CameraEdit, EditorObject, EditorState, LightData, LightDataExt, NodeEdit, NodeKind, ObjectKind, PhysicalCamera,
    SceneNode, Selection, Shape, ShapeMaterial,
};

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
            let mut keys = KeyButtons::new(state);

            match state.selected.clone() {
                None => {
                    ui.label(egui::RichText::new("Nothing selected.").weak());
                }
                Some(Selection::Light(idx)) => {
                    if let Some(light) = state.lights.get_mut(idx) {
                        let id = light.id;
                        draw_light(ui, &mut light.data, &mut keys, id);
                    }
                }
                Some(Selection::Object(id)) => {
                    let project_dir = state.project_dir.clone();
                    if let Some(object) = state.objects.iter_mut().find(|o| o.id == id) {
                        ui.push_id(id, |ui| draw_object(ui, object, project_dir.as_deref()));
                    }
                }
                Some(Selection::GltfNode(idx)) => {
                    let emissive: Vec<EditorTarget> = animation::node_materials(state, idx)
                        .into_iter()
                        .map(|m| EditorTarget::Scene(Target::EmissiveFactor(m)))
                        .collect();
                    let fov = animation::node_camera(state, idx).map(|c| EditorTarget::Scene(Target::CameraFov(c)));
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
                            draw_transform(ui, &mut edit.translation, &mut edit.rotation, &mut edit.scale, Some((&mut keys, idx)));
                            if let Some(NodeKind::Mesh(_)) = &node_kind {
                                draw_material(ui, edit, &mut keys, &emissive);
                            }
                            if let Some(particles) = &mut edit.particles {
                                draw_particles(ui, particles, &mut keys, idx);
                                ui.horizontal(|ui| {
                                    if ui.button("Burst").on_hover_text("Spawn this many particles at once (not saved).").clicked() {
                                        state.particle_bursts.push((idx, state.burst_size));
                                    }
                                    ui.add(egui::DragValue::new(&mut state.burst_size).speed(10.0).range(1..=MAX_PARTICLES));
                                });
                            }
                            if let Some(camera) = &mut edit.camera {
                                let mut is_main = state.main_camera == Some(idx);
                                let fov: Vec<EditorTarget> = fov.into_iter().collect();
                                draw_camera(ui, camera, &mut is_main, state.camera_settings.exposure_stops, &mut keys, &fov);
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
            keys.apply(state);
        });
}

fn draw_light(ui: &mut egui::Ui, data: &mut LightData, keys: &mut KeyButtons, id: u64) {
    let color_key = [EditorTarget::Light(id, LightProperty::Color)];
    let intensity_key = [EditorTarget::Light(id, LightProperty::Intensity)];
    ui.label(egui::RichText::new(data.label()).strong());
    ui.separator();

    match data {
        LightData::Sky { color, intensity, direction } => {
            egui::Grid::new("light_grid").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
                keys.label(ui, "Color", &color_key);
                ui.color_edit_button_rgb(color);
                ui.end_row();

                keys.label(ui, "Intensity", &intensity_key);
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

                keys.label(ui, "Color", &color_key);
                ui.color_edit_button_rgb(color);
                ui.end_row();

                keys.label(ui, "Intensity", &intensity_key);
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

                keys.label(ui, "Color", &color_key);
                ui.color_edit_button_rgb(color);
                ui.end_row();

                keys.label(ui, "Intensity", &intensity_key);
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

                keys.label(ui, "Color", &color_key);
                ui.color_edit_button_rgb(color);
                ui.end_row();

                keys.label(ui, "Intensity", &intensity_key);
                ui.add(egui::DragValue::new(intensity).speed(0.1).range(0.0..=1_000_000.0).suffix(" nt"));
                ui.end_row();
            });
        }
    }
}

fn draw_object(ui: &mut egui::Ui, object: &mut EditorObject, project_dir: Option<&Path>) {
    ui.add(egui::TextEdit::singleline(&mut object.data.name).hint_text(UNNAMED));
    let data = &mut object.data;
    match &mut data.kind {
        ObjectKind::Shape { shape, .. } => {
            ui.horizontal(|ui| {
                ui.label("Shape");
                egui::ComboBox::from_id_salt("shape").selected_text(shape.label()).show_ui(ui, |ui| {
                    for option in Shape::ALL {
                        ui.selectable_value(shape, option, option.label());
                    }
                });
            });
        }
        ObjectKind::Model { uri } => {
            let path = Path::new(uri.as_str());
            let shown = project_dir.and_then(|dir| path.strip_prefix(dir).ok()).unwrap_or(path);
            ui.label(egui::RichText::new(format!("Model {}", shown.display())).weak());
        }
        ObjectKind::PlayerStart => {
            ui.label(
                egui::RichText::new(format!(
                    "The player spawns here with its eyes {PLAYER_EYE_HEIGHT} m up, facing the arrow. Only the heading of the rotation is used."
                ))
                .weak(),
            );
        }
    }
    ui.separator();
    draw_transform(ui, &mut data.translation, &mut data.rotation, &mut data.scale, None);
    if let ObjectKind::Shape { material, .. } = &mut data.kind {
        draw_shape_material(ui, material);
    }
}

fn draw_particles(ui: &mut egui::Ui, data: &mut ParticleData, keys: &mut KeyButtons, node: usize) {
    let key = |p| [EditorTarget::Scene(Target::Particles(node, p))];
    egui::CollapsingHeader::new("Particles").default_open(true).show(ui, |ui| {
        egui::Grid::new("particles_grid").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
            ui.label("Shape");
            egui::ComboBox::from_id_salt("particle_shape")
                .selected_text(match data.shape {
                    ShapeKind::Box => "Box",
                    ShapeKind::Sphere => "Sphere",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut data.shape, ShapeKind::Box, "Box");
                    ui.selectable_value(&mut data.shape, ShapeKind::Sphere, "Sphere");
                });
            ui.end_row();

            match data.shape {
                ShapeKind::Box => {
                    ui.label("Half extents").on_hover_text("Half the box's size along the node's X, Y and Z, in meters.");
                    vec3_drag(ui, &mut data.half_extents, 0.01, 0.0..=f32::MAX, " m");
                    ui.end_row();

                    let mut slanted = data.shear_along.is_some();
                    ui.checkbox(&mut slanted, "Slant along").on_hover_text(
                        "Shears the box so its horizontal slices follow this direction, e.g. a light shaft.",
                    );
                    if slanted {
                        let direction = data.shear_along.get_or_insert([0.0, -1.0, 0.0]);
                        vec3_drag(ui, direction, 0.01, -1.0..=1.0, "");
                    } else {
                        data.shear_along = None;
                    }
                    ui.end_row();
                }
                ShapeKind::Sphere => {
                    ui.label("Radius");
                    ui.add(egui::DragValue::new(&mut data.radius).speed(0.01).range(0.0..=f32::MAX).suffix(" m"));
                    ui.end_row();
                }
            }

            ui.label("Max particles").on_hover_text("Particles alive at once. Shared budget across all emitters.");
            ui.add(egui::DragValue::new(&mut data.count).speed(10.0).range(1..=MAX_PARTICLES));
            ui.end_row();

            keys.label(ui, "Spawn rate", &key(ParticleProperty::SpawnRate));
            ui.add(egui::DragValue::new(&mut data.spawn_rate).speed(1.0).range(0.0..=f32::MAX).suffix(" /s"));
            ui.end_row();

            keys.label(ui, "Lifetime", &key(ParticleProperty::Lifetime));
            range_drag(ui, &mut data.lifetime, 0.1, 1.0, " s");
            ui.end_row();

            keys.label(ui, "Size", &key(ParticleProperty::Size)).on_hover_text("Diameter, min and max.");
            range_drag(ui, &mut data.size, 0.1, 1000.0, " mm");
            ui.end_row();

            keys.label(ui, "Velocity", &key(ParticleProperty::Velocity)).on_hover_text("World-space drift, m/s.");
            vec3_drag(ui, &mut data.velocity, 0.001, f32::MIN..=f32::MAX, "");
            ui.end_row();

            keys.label(ui, "Turbulence", &key(ParticleProperty::Turbulence)).on_hover_text("Speed of the swirling motion added to the drift.");
            ui.add(egui::DragValue::new(&mut data.turbulence).speed(0.001).range(0.0..=f32::MAX).suffix(" m/s"));
            ui.end_row();

            keys.label(ui, "Color", &key(ParticleProperty::Color)).on_hover_text("Scattering albedo when lit, emitted radiance when unlit.");
            ui.color_edit_button_rgb(&mut data.color);
            ui.end_row();

            keys.label(ui, "Opacity", &key(ParticleProperty::Opacity));
            ui.add(egui::Slider::new(&mut data.opacity, 0.0..=1.0));
            ui.end_row();

            ui.label("Lighting");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut data.lighting, Lighting::Lit, "Lit");
                ui.selectable_value(&mut data.lighting, Lighting::Unlit, "Unlit");
            });
            ui.end_row();

            ui.label("");
            ui.add_enabled(data.lighting == Lighting::Lit, egui::Checkbox::new(&mut data.visible_in_light_only, "Direct light only"))
                .on_hover_text("Ignore bounced light, so particles only show where a light reaches them (e.g. a light shaft).");
            ui.end_row();

            ui.label("Collision");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut data.collision, Collision::None, "None");
                ui.selectable_value(&mut data.collision, Collision::Settle, "Settle")
                    .on_hover_text("Particles come to rest on surfaces they fall onto. Only surfaces visible on screen count.");
            });
            ui.end_row();
        });

        let mean_life = (data.lifetime[0] + data.lifetime[1]) * 0.5;
        let steady = data.spawn_rate * mean_life;
        let text = if steady > data.count as f32 * 1.05 {
            format!("Spawning {steady:.0} per lifetime: capped at {}, so particles are recycled early.", data.count)
        } else {
            format!("≈ {steady:.0} alive once settled.")
        };
        ui.label(egui::RichText::new(text).weak());
    });
}

fn vec3_drag(ui: &mut egui::Ui, v: &mut [f32; 3], speed: f64, range: std::ops::RangeInclusive<f32>, suffix: &str) {
    ui.horizontal(|ui| {
        for (value, axis) in v.iter_mut().zip(["X ", "Y ", "Z "]) {
            ui.add(egui::DragValue::new(value).speed(speed).range(range.clone()).prefix(axis).suffix(suffix));
        }
    });
}

/// A min–max pair shown × `scale` (e.g. meters as millimeters). Min stays at or below max.
fn range_drag(ui: &mut egui::Ui, pair: &mut [f32; 2], speed: f64, scale: f32, suffix: &str) {
    let shown = pair.map(|v| v * scale);
    let [mut lo, mut hi] = shown;
    ui.horizontal(|ui| {
        ui.add(egui::DragValue::new(&mut lo).speed(speed).range(0.0..=hi).suffix(suffix));
        ui.label("–");
        ui.add(egui::DragValue::new(&mut hi).speed(speed).range(lo..=f32::MAX).suffix(suffix));
    });
    // Only write back on an edit: dividing the shown values back out can change the last bit,
    // which would read as an unsaved change.
    if [lo, hi] != shown {
        *pair = [lo / scale, hi / scale];
    }
}

fn draw_shape_material(ui: &mut egui::Ui, material: &mut ShapeMaterial) {
    ui.collapsing("Material", |ui| {
        egui::Grid::new("shape_material_grid").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
            ui.label("Base Color");
            ui.horizontal(|ui| {
                let color = &mut material.base_color;
                let mut rgb = [color[0], color[1], color[2]];
                ui.color_edit_button_rgb(&mut rgb);
                color[..3].copy_from_slice(&rgb);
                ui.add(egui::DragValue::new(&mut color[3]).speed(0.01).range(0.0..=1.0).prefix("A "));
            });
            ui.end_row();

            ui.label("Metallic");
            ui.add(egui::Slider::new(&mut material.metallic, 0.0..=1.0));
            ui.end_row();

            ui.label("Roughness");
            ui.add(egui::Slider::new(&mut material.roughness, 0.0..=1.0));
            ui.end_row();

            ui.label("Emissive");
            ui.horizontal(|ui| {
                ui.color_edit_button_rgb(&mut material.emissive);
                ui.add(egui::DragValue::new(&mut material.emissive_strength).speed(0.1).range(0.0..=f32::MAX).prefix("× "));
            });
            ui.end_row();
        });
    });
}

/// `keys`: the node whose fields get ◆ buttons, if the transform is a glTF node's.
fn draw_transform(
    ui: &mut egui::Ui,
    translation: &mut [f32; 3],
    rotation: &mut [f32; 4],
    scale: &mut [f32; 3],
    mut keys: Option<(&mut KeyButtons, usize)>,
) {
    let mut label = |ui: &mut egui::Ui, text: &str, target: fn(usize) -> Target| match &mut keys {
        Some((keys, node)) => {
            keys.label(ui, text, &[EditorTarget::Scene(target(*node))]);
        }
        None => {
            ui.label(text);
        }
    };
    ui.collapsing("Transform", |ui| {
        let euler = quat_to_euler(*rotation);
        let mut euler_edit = euler;

        egui::Grid::new("transform_grid")
            .num_columns(2)
            .spacing([12.0, 4.0])
            .show(ui, |ui| {
                label(ui, "Translation", Target::Translation);
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut translation[0]).speed(0.1).prefix("X "));
                    ui.add(egui::DragValue::new(&mut translation[1]).speed(0.1).prefix("Y "));
                    ui.add(egui::DragValue::new(&mut translation[2]).speed(0.1).prefix("Z "));
                });
                ui.end_row();

                label(ui, "Rotation", Target::Rotation);
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut euler_edit[0]).speed(0.5).prefix("X ").suffix("°"));
                    ui.add(egui::DragValue::new(&mut euler_edit[1]).speed(0.5).prefix("Y ").suffix("°"));
                    ui.add(egui::DragValue::new(&mut euler_edit[2]).speed(0.5).prefix("Z ").suffix("°"));
                });
                ui.end_row();

                label(ui, "Scale", Target::Scale);
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut scale[0]).speed(0.01).prefix("X "));
                    ui.add(egui::DragValue::new(&mut scale[1]).speed(0.01).prefix("Y "));
                    ui.add(egui::DragValue::new(&mut scale[2]).speed(0.01).prefix("Z "));
                });
                ui.end_row();
            });

        if euler_edit != euler {
            *rotation = euler_to_quat(euler_edit);
        }
    });
}

fn draw_material(ui: &mut egui::Ui, edit: &mut NodeEdit, keys: &mut KeyButtons, emissive_targets: &[EditorTarget]) {
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
                keys.label(ui, "Emissive", emissive_targets);
                ui.color_edit_button_rgb(emissive);
                ui.end_row();
            });
    });
}

fn draw_camera(
    ui: &mut egui::Ui,
    camera: &mut CameraEdit,
    is_main: &mut bool,
    project_stops: f32,
    keys: &mut KeyButtons,
    fov: &[EditorTarget],
) {
    ui.collapsing("Camera", |ui| {
        ui.checkbox(is_main, "Player camera")
            .on_hover_text("The camera the player renders this scene from. Other cameras are removed when it plays.");

        egui::Grid::new("camera_grid")
            .num_columns(2)
            .spacing([12.0, 4.0])
            .show(ui, |ui| {
                keys.label(ui, "FOV Y", fov);
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
