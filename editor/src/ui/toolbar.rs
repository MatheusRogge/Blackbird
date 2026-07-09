#![allow(deprecated)]
use egui::Context;

use crate::state::EditorState;

pub fn draw(ctx: &Context, state: &mut EditorState) {
    egui::Panel::top("toolbar").show(ctx, |ui| {
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
                let can_export = state.document.is_some();
                if ui.add_enabled(can_export, egui::Button::new("Export Scene…")).clicked() {
                    state.export_requested = true;
                    ui.close();
                }
            });
        });
    });

    egui::Panel::bottom("statusbar").show(ctx, |ui| {
        ui.label(&state.status);
    });
}
