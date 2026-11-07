//! Items of the Add menu, shared by the viewport's right-click menu and the hierarchy's.

use egui::Ui;

use crate::asset_browser::AssetKind;
use crate::state::{AddKind, EditorState, LightKind, Shape};

const MODEL_LIST_HEIGHT: f32 = 320.0;

/// Draws the Add submenus. Returns what was picked; the caller decides where it goes.
pub fn items(ui: &mut Ui, state: &EditorState) -> Option<AddKind> {
    let mut picked = None;
    ui.menu_button("💡 Light", |ui| {
        for (kind, label) in [
            (LightKind::Point, "💡 Point Light"),
            (LightKind::Spot, "◎ Spot Light"),
            (LightKind::Area, "▪ Area Light"),
            (LightKind::Sky, "☀ Sky Light"),
        ] {
            if ui.button(label).clicked() {
                picked = Some(AddKind::Light(kind));
                ui.close();
            }
        }
    });
    ui.menu_button("⬛ Shape", |ui| {
        for shape in Shape::ALL {
            if ui.button(shape.label()).clicked() {
                picked = Some(AddKind::Shape(shape));
                ui.close();
            }
        }
    });
    ui.menu_button("📦 Model", |ui| {
        let Some(project_dir) = &state.project_dir else {
            ui.label(egui::RichText::new("Open a project to place its models.").weak());
            return;
        };
        let models: Vec<&str> = state
            .assets
            .entries
            .iter()
            .filter(|e| e.kind == AssetKind::Model)
            .map(|e| e.path.as_str())
            .collect();
        if models.is_empty() {
            ui.label(egui::RichText::new("No .gltf / .glb models in the project.").weak());
            return;
        }
        egui::ScrollArea::vertical().max_height(MODEL_LIST_HEIGHT).show(ui, |ui| {
            for path in models {
                let name = path.rsplit('/').next().unwrap_or(path);
                if ui.button(name).on_hover_text(path).clicked() {
                    picked = Some(AddKind::Model(project_dir.join(path)));
                    ui.close();
                }
            }
        });
    });
    if ui.button("🚶 Player Start").clicked() {
        picked = Some(AddKind::PlayerStart);
        ui.close();
    }
    picked
}
