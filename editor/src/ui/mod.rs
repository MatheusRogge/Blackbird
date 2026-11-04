mod assets;
mod build;
mod gizmo;
mod hierarchy;
mod inspector;
mod toolbar;

use engine_core::world::World;
use engine::ui::UiLayer;

use crate::state::SharedState;

pub struct EditorUiLayer {
    state: SharedState,
    assets: assets::AssetPanel,
}

impl EditorUiLayer {
    pub fn new(state: SharedState) -> Self {
        Self { state, assets: assets::AssetPanel::default() }
    }
}

impl UiLayer for EditorUiLayer {
    fn ui(&mut self, ctx: &egui::Context, world: &World) {
        let Ok(mut state) = self.state.try_lock() else { return };
        let viewport = ctx.root_ui(|root| {
            toolbar::draw(root, &mut state);
            hierarchy::draw(root, &mut state);
            inspector::draw(root, &mut state);
            self.assets.draw(root, &mut state);
            root.available_rect_before_wrap()
        });
        gizmo::draw(ctx, &mut state, world, viewport);
        build::draw_log_window(ctx, &mut state);
        draw_camera_hints(ctx);
    }
}

fn draw_camera_hints(ctx: &egui::Context) {
    let screen = ctx.content_rect();
    let margin = 12.0;

    egui::Area::new(egui::Id::unique("camera_hints"))
        .fixed_pos(egui::pos2(screen.right() - margin, screen.bottom() - margin))
        .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(0.0, 0.0))
        .interactable(false)
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(egui::Color32::from_black_alpha(140))
                .corner_radius(4.0)
                .inner_margin(egui::Margin::symmetric(8, 6))
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 2.0;
                    let hint_color = egui::Color32::from_gray(200);
                    let label_color = egui::Color32::from_gray(140);
                    let font = egui::TextStyle::Small.resolve(ui.style());

                    for (key, action) in [
                        ("WASD / QE", "Move / Down, Up"),
                        ("Shift",    "Move faster"),
                        ("LMB drag", "Look"),
                        ("RMB drag", "Orbit"),
                        ("MMB drag", "Pan"),
                        ("Scroll",   "Zoom"),
                        ("LMB + Scroll", "Move speed"),
                        ("Click",    "Select"),
                        ("1 / 2 / 3", "Move / Rotate / Scale"),
                    ] {
                        ui.horizontal(|ui| {
                            ui.add(egui::Label::new(
                                egui::RichText::new(key).font(font.clone()).color(hint_color),
                            ));
                            ui.add(egui::Label::new(
                                egui::RichText::new(action).font(font.clone()).color(label_color),
                            ));
                        });
                    }
                });
        });
}
