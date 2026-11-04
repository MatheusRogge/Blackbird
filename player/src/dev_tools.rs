//! Development-build only: dev tools window (F1), free camera (F2) and an in-game log console.
//! The rendering stats overlay comes from the engine's `debug` feature.

use std::sync::{Arc, Mutex};

use engine::logging::LogHistory;
use engine::player_controller::PlayerController;
use engine::rendering::camera::Camera;
use engine::rendering::camera_controller::{CameraController, CameraMode};
use engine::rendering::light::{AreaLight, PointLight, SkyLight, SpotLight};
use engine::rendering::mesh::Mesh;
use engine::ui::{UiLayer, egui};
use engine::world::World;
use engine::{Engine, EntityKey};
use project::Project;

use crate::Content;

pub type SharedDevState = Arc<Mutex<DevState>>;

pub struct DevState {
    pub show_tools: bool,
    pub free_camera: bool,
    pub scene_status: String,
    project_name: String,
    asset_source: String,
    manifest_len: Option<usize>,
}

impl DevState {
    pub fn shared(project: &Project, asset_source: String, manifest_len: Option<usize>) -> SharedDevState {
        Arc::new(Mutex::new(Self {
            show_tools: true,
            free_camera: false,
            scene_status: String::new(),
            project_name: project.name.clone(),
            asset_source,
            manifest_len,
        }))
    }
}

/// WASD to move, LMB drag to look, scroll to zoom.
pub struct FreeCamera {
    look: CameraController,
    movement: PlayerController,
}

impl FreeCamera {
    pub fn new(camera: EntityKey, move_speed: f32) -> Self {
        let mut movement = PlayerController::new(move_speed);
        movement.attach(camera);
        Self { look: CameraController::new(camera, CameraMode::FirstPerson), movement }
    }

    pub fn tick(&mut self, engine: &mut Engine, delta: f32) {
        self.movement.tick::<Camera>(engine, delta);
        let ui_wants_pointer = engine.ui_wants_pointer();
        let (input, mut world) = engine.input_and_world();
        self.look.tick(input, &mut world, delta, ui_wants_pointer);
    }
}

pub struct DevToolsLayer {
    state: SharedDevState,
    log: LogHistory,
    filter: String,
    problems_only: bool,
}

impl DevToolsLayer {
    pub fn new(state: SharedDevState, log: LogHistory) -> Self {
        Self { state, log, filter: String::new(), problems_only: false }
    }

    fn draw_log(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(&mut self.filter).hint_text("filter").desired_width(160.0));
            ui.checkbox(&mut self.problems_only, "warnings & errors");
        });

        let filter = self.filter.to_lowercase();
        let lines = self.log.snapshot();
        egui::ScrollArea::vertical()
            .max_height(260.0)
            .auto_shrink([false, true])
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for line in &lines {
                    let color = if line.contains("ERROR") {
                        egui::Color32::from_rgb(240, 110, 100)
                    } else if line.contains("WARN") {
                        egui::Color32::from_rgb(230, 190, 90)
                    } else if self.problems_only {
                        continue;
                    } else {
                        ui.visuals().text_color()
                    };
                    if !filter.is_empty() && !line.to_lowercase().contains(&filter) {
                        continue;
                    }
                    ui.label(egui::RichText::new(line).monospace().small().color(color));
                }
            });
    }
}

impl UiLayer for DevToolsLayer {
    fn ui(&mut self, ctx: &egui::Context, world: &World) {
        let shared = Arc::clone(&self.state);
        let Ok(mut state) = shared.try_lock() else { return };
        if !state.show_tools {
            return;
        }

        let lights = world.get_entities::<SkyLight>().len()
            + world.get_entities::<PointLight>().len()
            + world.get_entities::<SpotLight>().len()
            + world.get_entities::<AreaLight>().len();
        let meshes = world.get_entities::<Mesh>().len();

        egui::Window::new("Dev Tools")
            .default_width(460.0)
            .default_pos(ctx.content_rect().right_top() + egui::vec2(-480.0, 20.0))
            .show(ctx, |ui| {
                egui::Grid::new("dev_build_info").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
                    ui.label("Project");
                    ui.label(&state.project_name);
                    ui.end_row();
                    ui.label("Build");
                    ui.label(format!("development, {}", Content::CURRENT.label()));
                    ui.end_row();
                    ui.label("Assets");
                    ui.label(&state.asset_source);
                    ui.end_row();
                    ui.label("Manifest");
                    ui.label(match state.manifest_len {
                        Some(n) => format!("{n} assets baked in"),
                        None => "none, any file can load".to_string(),
                    });
                    ui.end_row();
                    ui.label("Scene");
                    ui.label(&state.scene_status);
                    ui.end_row();
                    ui.label("Entities");
                    ui.label(format!("{meshes} meshes, {lights} lights"));
                    ui.end_row();
                });

                ui.separator();
                ui.checkbox(&mut state.free_camera, "Free camera (F2)");
                ui.label(egui::RichText::new("F1 hides this window").small().weak());

                ui.separator();
                egui::CollapsingHeader::new("Log").default_open(true).show(ui, |ui| self.draw_log(ui));
            });
    }
}
