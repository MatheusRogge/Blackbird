use build_tool::BuildKind;
use egui::Ui;

use crate::state::EditorState;

pub fn draw_menu(ui: &mut Ui, state: &mut EditorState) {
    let idle = state.build.running.is_none();
    if let Some(kind) = state.build.running {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(format!("Building {kind}…"));
        });
        ui.separator();
    }

    let dev = ui
        .add_enabled(idle, egui::Button::new("Development Build"))
        .on_hover_text("Loose assets next to the binary, dev tools (F1), free camera (F2), log file. Quick to compile.");
    if dev.clicked() {
        state.build.requested = Some(BuildKind::Development);
        ui.close();
    }
    let prod = ui
        .add_enabled(idle, egui::Button::new("Production Build"))
        .on_hover_text("Assets compressed into data.bbpak, no dev tools, fully optimised and stripped. Slow to compile.");
    if prod.clicked() {
        state.build.requested = Some(BuildKind::Production);
        ui.close();
    }

    ui.separator();
    ui.checkbox(&mut state.build.show_log, "Show Build Log");
    ui.label(egui::RichText::new("Builds use the saved scene files.").small().weak());
}

pub fn draw_log_window(ctx: &egui::Context, state: &mut EditorState) {
    let mut open = state.build.show_log;
    egui::Window::new("Build")
        .open(&mut open)
        .default_size([640.0, 360.0])
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                match state.build.running {
                    Some(kind) => {
                        ui.spinner();
                        ui.label(format!("Building {kind}…"));
                    }
                    None => match &state.build.last_output {
                        Some(path) => {
                            ui.label("Output:");
                            ui.monospace(path.display().to_string());
                            if ui.small_button("Copy").clicked() {
                                ui.ctx().copy_text(path.display().to_string());
                            }
                        }
                        None => {
                            ui.label(egui::RichText::new("No build output yet.").weak());
                        }
                    },
                }
            });
            ui.separator();
            egui::ScrollArea::both()
                .auto_shrink([false, false])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for line in &state.build.log {
                        let text = egui::RichText::new(line).monospace().small();
                        let text = if line.starts_with("error") || line.contains("build failed") {
                            text.color(ui.visuals().error_fg_color)
                        } else if line.starts_with("warning") {
                            text.color(ui.visuals().warn_fg_color)
                        } else {
                            text
                        };
                        ui.label(text);
                    }
                });
        });
    state.build.show_log = open;
}
