use std::sync::{Arc, RwLock};

use engine_core::world::World;
use ui::UiLayer;

use crate::debug_flags::RenderDebugFlags;
use crate::profiler::FrameStats;

pub struct DebugLayer {
    stats: Arc<RwLock<FrameStats>>,
    debug_flags: Arc<RwLock<RenderDebugFlags>>,
}

impl DebugLayer {
    pub fn new(stats: Arc<RwLock<FrameStats>>, debug_flags: Arc<RwLock<RenderDebugFlags>>) -> Self {
        Self { stats, debug_flags }
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

                ui.separator();
                let mem = &stats.memory;
                let mib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);
                egui::Grid::new("arenas").striped(true).show(ui, |ui| {
                    ui.label("upload arena");
                    ui.label(format!("{:.2} MiB in {} page(s)", mib(mem.upload_bytes), mem.upload_pages));
                    ui.end_row();
                    ui.label("geometry");
                    ui.label(format!(
                        "{}/{} verts  {}/{} indices  (gen {})",
                        mem.geometry_vertices.0,
                        mem.geometry_vertices.1,
                        mem.geometry_indices.0,
                        mem.geometry_indices.1,
                        mem.geometry_generation
                    ));
                    ui.end_row();
                    ui.label("frame arena");
                    ui.label(format!(
                        "{:.1}/{:.1} KiB  (gen {})",
                        mem.frame_bytes.0 as f64 / 1024.0,
                        mem.frame_bytes.1 as f64 / 1024.0,
                        mem.frame_generation
                    ));
                    ui.end_row();
                    ui.label("screen textures");
                    ui.label(format!("{} logical → {} physical", mem.screen_textures.0, mem.screen_textures.1));
                    ui.end_row();
                });

                ui.separator();
                ui.label("Layers");
                if let Ok(mut flags) = self.debug_flags.write() {
                    ui.checkbox(&mut flags.albedo, "Albedo");
                    ui.checkbox(&mut flags.indirect, "Indirect (GI)");
                    ui.checkbox(&mut flags.direct, "Direct lights");
                    ui.checkbox(&mut flags.shadows, "Shadows");

                    ui.separator();
                    ui.label("Visualization");
                    ui.checkbox(&mut flags.show_bvh, "BVH AABBs");
                    if flags.show_bvh {
                        ui.indent("bvh_indent", |ui| {
                            ui.checkbox(&mut flags.bvh_leaves_only, "Leaves only");
                        });
                    }
                    ui.checkbox(&mut flags.show_probes, "DDGI Probes");

                    ui.separator();
                    ui.label("DDGI");
                    ui.add(
                        egui::Slider::new(&mut flags.ddgi_hysteresis, 0.0..=1.0)
                            .text("Hysteresis"),
                    );
                    ui.add(
                        egui::Slider::new(&mut flags.ddgi_normal_bias, 0.0..=1.0)
                            .text("Normal Bias"),
                    );
                    ui.add(
                        egui::Slider::new(&mut flags.ddgi_view_bias, 0.0..=1.0)
                            .text("View Bias"),
                    );
                    ui.add(
                        egui::Slider::new(&mut flags.ddgi_probe_spacing, 0.5..=50.0)
                            .text("Probe Spacing"),
                    );
                    ui.label("Grid Anchor");
                    egui::Grid::new("grid_anchor").num_columns(2).show(ui, |ui| {
                        ui.label("X");
                        ui.add(egui::DragValue::new(&mut flags.ddgi_grid_anchor[0]).speed(0.5));
                        ui.end_row();
                        ui.label("Y");
                        ui.add(egui::DragValue::new(&mut flags.ddgi_grid_anchor[1]).speed(0.5));
                        ui.end_row();
                        ui.label("Z");
                        ui.add(egui::DragValue::new(&mut flags.ddgi_grid_anchor[2]).speed(0.5));
                        ui.end_row();
                    });
                }
            });
    }
}
