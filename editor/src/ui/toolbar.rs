use egui::Ui;

use crate::state::{CameraSettings, EditorState};

const SAVE_SHORTCUT: egui::KeyboardShortcut = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::S);

pub fn draw(root: &mut Ui, state: &mut EditorState) {
    let can_save = state.document.is_some();
    if can_save && root.ctx().input_mut(|i| i.consume_shortcut(&SAVE_SHORTCUT)) {
        state.save_requested = true;
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

            ui.menu_button("Camera", |ui| draw_camera_settings(ui, &mut state.camera_settings));
        });
    });

    egui::Panel::bottom("statusbar").show(root, |ui| {
        ui.label(&state.status);
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
        });
    ui.label(egui::RichText::new("Scroll while holding LMB to adjust speed").small().weak());
    if ui.button("Reset to defaults").clicked() {
        *settings = CameraSettings::default();
    }
}
