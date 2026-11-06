use std::sync::RwLock;

use egui::Ui;
use engine::rendering::debug_flags::{GBufferView, RenderDebugFlags};

use crate::state::{CameraSettings, EditorState};

const SAVE_SHORTCUT: egui::KeyboardShortcut = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::S);
const GRID_SHORTCUT: egui::KeyboardShortcut = egui::KeyboardShortcut::new(egui::Modifiers::NONE, egui::Key::G);

pub fn draw(root: &mut Ui, state: &mut EditorState, debug_flags: &RwLock<RenderDebugFlags>) {
    let can_save = state.document.is_some();
    if can_save && root.ctx().input_mut(|i| i.consume_shortcut(&SAVE_SHORTCUT)) {
        state.save_requested = true;
    }
    let typing = root.ctx().egui_wants_keyboard_input();
    if !typing && root.ctx().input_mut(|i| i.consume_shortcut(&GRID_SHORTCUT)) {
        let mut flags = debug_flags.write().expect("debug_flags poisoned");
        flags.show_grid = !flags.show_grid;
    }

    egui::Panel::top("toolbar").show(root, |ui| {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("New Project…").clicked() {
                    state.new_project_requested = true;
                    ui.close();
                }
                if ui.button("Open Project…").clicked() {
                    state.open_project_requested = true;
                    ui.close();
                }

                let recent = state.recent_projects.entries.clone();
                ui.add_enabled_ui(!recent.is_empty(), |ui| {
                    ui.menu_button("Open Recent", |ui| {
                        for entry in &recent {
                            let label = format!("{}\n{}", entry.name, entry.path.display());
                            if ui.button(&label).clicked() {
                                state.open_recent_path = Some(entry.path.clone());
                                ui.close();
                            }
                        }
                    });
                });

                ui.separator();
                let has_project = state.project.is_some();
                if ui.add_enabled(has_project, egui::Button::new("Open Scene…")).clicked() {
                    state.open_requested = true;
                    ui.close();
                }
                if ui.add_enabled(has_project, egui::Button::new("Import Scene…")).clicked() {
                    state.import_requested = true;
                    ui.close();
                }
                let save = egui::Button::new("Save Scene").shortcut_text(ui.ctx().format_shortcut(&SAVE_SHORTCUT));
                if ui.add_enabled(can_save, save).clicked() {
                    state.save_requested = true;
                    ui.close();
                }
                if ui.add_enabled(can_save, egui::Button::new("Export Scene…")).clicked() {
                    state.export_requested = true;
                    ui.close();
                }
            });

            ui.add_enabled_ui(state.project.is_some(), |ui| {
                ui.menu_button("Project", |ui| draw_project_menu(ui, state));
            });
            ui.add_enabled_ui(state.project.is_some(), |ui| {
                ui.menu_button("Build", |ui| super::build::draw_menu(ui, state));
            });
            ui.menu_button("View", |ui| draw_view_menu(ui, state, debug_flags));
            ui.menu_button("Camera", |ui| draw_camera_settings(ui, &mut state.camera_settings));
        });
    });

    egui::Panel::bottom("statusbar").show(root, |ui| {
        ui.horizontal(|ui| {
            if state.build.running.is_some() {
                ui.spinner();
            }
            ui.label(&state.status);
        });
    });
}

fn draw_project_menu(ui: &mut Ui, state: &mut EditorState) {
    let scenes = state.project_scenes.clone();
    let Some(project) = state.project.as_mut() else { return };
    ui.menu_button("Default Scene", |ui| {
        ui.radio_value(&mut project.default_scene, None, "None");
        for scene in scenes {
            let label = scene.strip_prefix("scenes/").unwrap_or(&scene).to_string();
            ui.radio_value(&mut project.default_scene, Some(scene), label);
        }
    });
}

fn draw_view_menu(ui: &mut Ui, state: &mut EditorState, debug_flags: &RwLock<RenderDebugFlags>) {
    let mut flags = debug_flags.write().expect("debug_flags poisoned");
    let grid = egui::Button::selectable(flags.show_grid, "Grid").shortcut_text(ui.ctx().format_shortcut(&GRID_SHORTCUT));
    if ui.add(grid).clicked() {
        flags.show_grid = !flags.show_grid;
    }
    if ui.add(egui::Button::selectable(state.show_gizmos, "Gizmos")).clicked() {
        state.show_gizmos = !state.show_gizmos;
    }
    ui.separator();
    if ui.add(egui::Button::selectable(flags.reflections, "Reflections")).clicked() {
        flags.reflections = !flags.reflections;
    }
    let ssr = egui::Button::selectable(flags.ssr, "Screen-space reflections");
    if ui.add_enabled(flags.reflections, ssr).clicked() {
        flags.ssr = !flags.ssr;
    }
    let probe = egui::Button::selectable(flags.reflection_probe, "Reflection probe");
    if ui.add_enabled(flags.reflections, probe).clicked() {
        flags.reflection_probe = !flags.reflection_probe;
    }
    ui.menu_button("G-buffer view", |ui| {
        for view in GBufferView::ALL {
            ui.radio_value(&mut flags.view, view, view.label());
        }
    });
}

fn draw_camera_settings(ui: &mut Ui, settings: &mut CameraSettings) {
    egui::Grid::new("camera_settings_grid")
        .num_columns(2)
        .spacing([12.0, 6.0])
        .show(ui, |ui| {
            ui.label("Move speed");
            ui.add(
                egui::DragValue::new(&mut settings.move_speed)
                    .speed(0.1)
                    .range(CameraSettings::MIN_SPEED..=CameraSettings::MAX_SPEED)
                    .suffix(" u/s"),
            );
            ui.end_row();

            ui.label("Shift multiplier");
            ui.add(egui::DragValue::new(&mut settings.fast_multiplier).speed(0.05).range(1.0..=20.0).prefix("×"));
            ui.end_row();

            ui.label("Exposure");
            ui.add(egui::DragValue::new(&mut settings.exposure_stops).speed(0.05).range(-24.0..=24.0).suffix(" EV"));
            ui.end_row();
        });
    ui.label(egui::RichText::new("Scroll while holding LMB to adjust speed").small().weak());
    if ui.button("Reset to defaults").clicked() {
        *settings = CameraSettings::default();
    }
}
