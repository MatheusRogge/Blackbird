use std::sync::{Arc, RwLock};

use engine_core::world::World;
use ui::UiLayer;

use crate::arena::texture_pool::AliasingReport;
use crate::debug_flags::RenderDebugFlags;
use crate::profiler::FrameStats;

const LABEL_WIDTH: f32 = 64.0;
const HEADER_HEIGHT: f32 = 20.0;
const ROW_HEIGHT: f32 = 26.0;

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
                    ui.label(format!(
                        "{} logical → {} physical",
                        mem.screen_textures.logical(),
                        mem.screen_textures.physical()
                    ));
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
                    ui.checkbox(&mut flags.show_texture_aliasing, "Texture aliasing");

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

        let mut open = self
            .debug_flags
            .read()
            .expect("debug_flags RwLock poisoned")
            .show_texture_aliasing;
        if open {
            egui::Window::new("Texture aliasing")
                .open(&mut open)
                .resizable(false)
                .show(ctx, |ui| aliasing_timeline(ui, &stats.memory.screen_textures));
            if !open {
                self.debug_flags
                    .write()
                    .expect("debug_flags RwLock poisoned")
                    .show_texture_aliasing = false;
            }
        }
    }
}

/// Passes run left to right, each row is one physical texture, and each bar is
/// a logical texture spanning the passes it is live for.
fn aliasing_timeline(ui: &mut egui::Ui, report: &AliasingReport) {
    let mib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);
    let logical = report.logical_bytes();
    let physical = report.physical_bytes();
    ui.label(format!(
        "{} logical → {} physical    {:.1} MiB → {:.1} MiB (saved {:.1} MiB)",
        report.logical(),
        report.physical(),
        mib(logical),
        mib(physical),
        mib(logical.saturating_sub(physical))
    ));
    if report.unaliased > 0 {
        ui.label(format!("{} screen texture(s) not aliased (read before written or unused)", report.unaliased));
    }
    ui.separator();

    if report.passes.is_empty() {
        ui.label("No passes compiled yet");
        return;
    }

    let font = egui::FontId::proportional(11.0);
    let longest = report.passes.iter().map(|p| p.len()).max().unwrap_or(0);
    let col_width = (longest as f32 * 6.5).max(56.0);
    let size = egui::vec2(
        LABEL_WIDTH + col_width * report.passes.len() as f32,
        HEADER_HEIGHT + ROW_HEIGHT * report.slots as f32,
    );

    egui::ScrollArea::horizontal().show(ui, |ui| {
        let (response, painter) = ui.allocate_painter(size, egui::Sense::hover());
        let origin = response.rect.min;
        let visuals = ui.visuals();
        let text_color = visuals.text_color();
        let grid_color = visuals.widgets.noninteractive.bg_stroke.color;

        for (i, pass) in report.passes.iter().enumerate() {
            let x = origin.x + LABEL_WIDTH + col_width * i as f32;
            painter.line_segment(
                [egui::pos2(x, origin.y), egui::pos2(x, origin.y + size.y)],
                egui::Stroke::new(1.0, grid_color),
            );
            painter.text(
                egui::pos2(x + col_width * 0.5, origin.y + HEADER_HEIGHT * 0.5),
                egui::Align2::CENTER_CENTER,
                pass,
                font.clone(),
                text_color,
            );
        }

        for slot in 0..report.slots {
            painter.text(
                egui::pos2(origin.x + 4.0, origin.y + HEADER_HEIGHT + ROW_HEIGHT * (slot as f32 + 0.5)),
                egui::Align2::LEFT_CENTER,
                format!("phys {slot}"),
                font.clone(),
                text_color,
            );
        }

        let mut hovered = None;
        for texture in &report.textures {
            let bar = egui::Rect::from_min_max(
                egui::pos2(
                    origin.x + LABEL_WIDTH + col_width * texture.first as f32,
                    origin.y + HEADER_HEIGHT + ROW_HEIGHT * texture.slot as f32,
                ),
                egui::pos2(
                    origin.x + LABEL_WIDTH + col_width * (texture.last + 1) as f32,
                    origin.y + HEADER_HEIGHT + ROW_HEIGHT * (texture.slot + 1) as f32,
                ),
            )
            .shrink(3.0);

            let hue = (texture.id as f32 * 0.618_034).fract();
            painter.rect_filled(bar, 4.0, egui::ecolor::Hsva::new(hue, 0.45, 0.65, 1.0));
            painter.with_clip_rect(bar).text(
                bar.left_center() + egui::vec2(6.0, 0.0),
                egui::Align2::LEFT_CENTER,
                format!("{} #{} {:?}", texture.writer, texture.id, texture.format),
                font.clone(),
                egui::Color32::BLACK,
            );

            if response.hover_pos().is_some_and(|p| bar.contains(p)) {
                hovered = Some(texture);
                painter.rect_stroke(bar, 4.0, egui::Stroke::new(2.0, text_color), egui::StrokeKind::Outside);
            }
        }

        if let Some(texture) = hovered {
            response.on_hover_ui_at_pointer(|ui| {
                ui.label(format!("resource #{} written by {}", texture.id, texture.writer));
                ui.label(format!("{:?}, {:.2} MiB", texture.format, mib(texture.bytes)));
                ui.label(format!(
                    "live: {} → {}",
                    report.passes[texture.first], report.passes[texture.last]
                ));
                ui.label(format!("physical texture {}", texture.slot));
            });
        }
    });
}
