use egui::Ui;

use crate::state::EditorState;

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
        });
    });

    egui::Panel::bottom("statusbar").show(root, |ui| {
        ui.label(&state.status);
    });
}
