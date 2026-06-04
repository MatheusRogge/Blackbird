#![allow(deprecated)]
use egui::Context;

use crate::state::EditorState;

pub fn draw(ctx: &Context, state: &mut EditorState) {
    egui::Panel::top("toolbar").show(ctx, |ui| {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Open…").clicked() {
                    state.open_requested = true;
                    ui.close();
                }
                let can_export = state.document.is_some();
                if ui.add_enabled(can_export, egui::Button::new("Export…")).clicked() {
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
