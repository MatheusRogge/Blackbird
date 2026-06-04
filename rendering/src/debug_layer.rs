use std::sync::{Arc, RwLock};

use engine_core::world::World;
use ui::UiLayer;

use crate::profiler::FrameStats;

pub struct DebugLayer {
    stats: Arc<RwLock<FrameStats>>,
}

impl DebugLayer {
    pub fn new(stats: Arc<RwLock<FrameStats>>) -> Self {
        Self { stats }
    }
}

impl UiLayer for DebugLayer {
    fn ui(&mut self, ctx: &egui::Context, _world: &World) {
        let stats = self.stats.read().expect("frame_stats RwLock poisoned").clone();

        egui::Window::new("Rendering Stats")
            .resizable(false)
            .show(ctx, |ui| {
                ui.label(format!(
                    "frame {:5}   fps {:.1}   cpu {:.2}ms (avg {:.2}ms)",
                    stats.frame, stats.fps, stats.last_ms, stats.avg_ms
                ));

                if !stats.passes.is_empty() {
                    ui.separator();
                    egui::Grid::new("passes")
                        .striped(true)
                        .min_col_width(120.0)
                        .show(ui, |ui| {
                            ui.label("pass");
                            ui.label("cpu (last)");
                            ui.label("cpu (avg)");
                            ui.label("gpu");
                            ui.end_row();

                            for pass in &stats.passes {
                                ui.label(pass.name);
                                ui.label(format!("{:.2}ms", pass.last_ms));
                                ui.label(format!("{:.2}ms", pass.avg_ms));
                                if let Some(gpu) = pass.gpu_ms {
                                    ui.label(format!("{:.2}ms", gpu));
                                } else {
                                    ui.label("—");
                                }
                                ui.end_row();
                            }
                        });
                }
            });
    }
}
