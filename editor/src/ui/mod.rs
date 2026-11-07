mod add_menu;
mod assets;
mod build;
mod gizmo;
mod hierarchy;
mod inspector;
mod toolbar;

use std::sync::{Arc, RwLock};

use engine_core::world::World;
use engine::rendering::debug_flags::RenderDebugFlags;
use engine::ui::UiLayer;

use crate::state::{Prompt, PromptAnswer, SharedState};

pub struct EditorUiLayer {
    state: SharedState,
    assets: assets::AssetPanel,
    debug_flags: Arc<RwLock<RenderDebugFlags>>,
}

impl EditorUiLayer {
    pub fn new(state: SharedState, debug_flags: Arc<RwLock<RenderDebugFlags>>) -> Self {
        Self { state, assets: assets::AssetPanel::default(), debug_flags }
    }
}

impl UiLayer for EditorUiLayer {
    fn ui(&mut self, ctx: &egui::Context, world: &World) {
        let Ok(mut state) = self.state.try_lock() else { return };
        let (viewport, add) = ctx.root_ui(|root| {
            toolbar::draw(root, &mut state, &self.debug_flags);
            let add = hierarchy::draw(root, &mut state);
            inspector::draw(root, &mut state);
            self.assets.draw(root, &mut state);
            (root.available_rect_before_wrap(), add)
        });
        if let Some(kind) = add {
            gizmo::place(ctx, &mut state, world, kind, Some(viewport.center()));
        }
        gizmo::draw(ctx, &mut state, world, viewport);
        build::draw_log_window(ctx, &mut state);
        draw_camera_hints(ctx);
        draw_prompt(ctx, &mut state);

        state.close_guard.set_veto(state.is_dirty());
        if state.close_guard.take_close_request() {
            state.prompt = Some(Prompt::Exit);
        }

        // One undo step per finished interaction: nothing is committed while a button is held
        // (gizmo or slider drags) or text is being typed (a name, a value typed into a field).
        let interacting = ctx.input(|i| i.pointer.any_down()) || ctx.egui_wants_keyboard_input();
        if !interacting {
            let now = state.snapshot();
            state.history.commit(now);
        }
    }
}

/// The question in `state.prompt`, as a modal dialog. The answer goes to `prompt_answer`.
fn draw_prompt(ctx: &egui::Context, state: &mut crate::state::EditorState) {
    let Some(prompt) = state.prompt.clone() else { return };
    if state.prompt_answer.is_some() {
        return;
    }
    let scene = state
        .file_path
        .as_ref()
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "the scene".to_string());
    let (title, body, yes, no, cancel) = match &prompt {
        Prompt::Exit => (
            "Unsaved changes",
            format!("Save the changes to {scene} before closing?"),
            "Save",
            "Don't Save",
            Some("Cancel"),
        ),
        Prompt::Reload => (
            "Scene changed on disk",
            format!("{scene} was changed outside the editor, but you have unsaved changes. Reloading discards them."),
            "Reload",
            "Keep My Changes",
            None,
        ),
        Prompt::Recover(_) => (
            "Recover unsaved changes",
            format!("{scene} has an auto-saved copy with changes that were never saved. Restore them?"),
            "Restore",
            "Discard",
            None,
        ),
    };
    let mut answer = None;
    egui::Modal::new(egui::Id::unique("editor_prompt")).show(ctx, |ui| {
        ui.set_max_width(360.0);
        ui.heading(title);
        ui.label(body);
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui.button(yes).clicked() {
                answer = Some(PromptAnswer::Yes);
            }
            if ui.button(no).clicked() {
                answer = Some(PromptAnswer::No);
            }
            if let Some(cancel) = cancel
                && ui.button(cancel).clicked()
            {
                answer = Some(PromptAnswer::Cancel);
            }
        });
    });
    if answer.is_none() && cancel_pressed(ctx) && matches!(prompt, Prompt::Exit) {
        answer = Some(PromptAnswer::Cancel);
    }
    state.prompt_answer = answer;
}

fn cancel_pressed(ctx: &egui::Context) -> bool {
    ctx.input(|i| i.key_pressed(egui::Key::Escape))
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
                        ("RMB click", "Add here"),
                        ("Del",      "Delete selected"),
                        ("Ctrl+Z / Ctrl+Y", "Undo / Redo"),
                        ("1 / 2 / 3", "Move / Rotate / Scale"),
                        ("G",        "Toggle grid"),
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
