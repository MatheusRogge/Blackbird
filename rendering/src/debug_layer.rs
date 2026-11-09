//! Rendering dev tools: an overlay (F1) with two tabs, per-pass CPU and GPU times and memory
//! stats, and folded away below them the render layer toggles, debug visualisations and GI settings.
//! `gi_controls` and `layer_toggles` are shared with hosts that put these settings in their own UI.
//! The overlay sits in the top-right of the host's scene viewport (`ui::SCENE_VIEWPORT_ID`), read a
//! frame late since this layer runs first; the whole window without one.

use std::sync::{Arc, RwLock};

use engine_core::world::World;
use ui::UiLayer;

use crate::arena::texture_pool::AliasingReport;
use crate::debug_flags::{GBufferView, RenderDebugFlags};
use crate::profiler::FrameStats;

const LABEL_WIDTH: f32 = 64.0;
const HEADER_HEIGHT: f32 = 20.0;
const ROW_HEIGHT: f32 = 26.0;

const TOGGLE_KEY: egui::Key = egui::Key::F1;
/// Width of a pass's GPU time bar at the frame's slowest pass.
const BAR_WIDTH: f32 = 70.0;
/// Room below the settings for the header and the frame's margins.
const SETTINGS_MARGIN: f32 = 40.0;

#[derive(Copy, Clone, PartialEq, Eq)]
enum Tab {
    Timings,
    Memory,
}

pub struct DebugLayer {
    stats: Arc<RwLock<FrameStats>>,
    debug_flags: Arc<RwLock<RenderDebugFlags>>,
    visible: bool,
    tab: Tab,
}

impl DebugLayer {
    pub fn new(stats: Arc<RwLock<FrameStats>>, debug_flags: Arc<RwLock<RenderDebugFlags>>) -> Self {
        Self { stats, debug_flags, visible: false, tab: Tab::Timings }
    }
}

impl UiLayer for DebugLayer {
    fn ui(&mut self, ctx: &egui::Context, _world: &World) {
        if !ctx.egui_wants_keyboard_input() && ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, TOGGLE_KEY)) {
            self.visible = !self.visible;
        }
        let stats = self.stats.read().expect("frame_stats RwLock poisoned").clone();

        if self.visible {
            let viewport = ctx
                .data(|d| d.get_temp::<egui::Rect>(egui::Id::unique(ui::SCENE_VIEWPORT_ID)))
                .unwrap_or_else(|| ctx.content_rect());
            egui::Area::new(egui::Id::unique("frame_timings"))
                .order(egui::Order::Foreground)
                .pivot(egui::Align2::RIGHT_TOP)
                .fixed_pos(viewport.right_top() + egui::vec2(-10.0, 10.0))
                .show(ctx, |ui| {
                    egui::Frame::new()
                        .fill(egui::Color32::from_black_alpha(200))
                        .corner_radius(6.0)
                        .inner_margin(10.0)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.selectable_value(&mut self.tab, Tab::Timings, "Timings");
                                ui.selectable_value(&mut self.tab, Tab::Memory, "Memory");
                            });
                            ui.separator();
                            ui.style_mut().override_text_style = Some(egui::TextStyle::Monospace);
                            match self.tab {
                                Tab::Timings => frame_timings(ui, &stats),
                                Tab::Memory => {
                                    memory(ui, &stats);
                                    let mut flags = self.debug_flags.write().expect("debug_flags RwLock poisoned");
                                    ui.checkbox(&mut flags.show_texture_aliasing, "Texture aliasing timeline");
                                }
                            }
                            ui.add_space(4.0);
                            // The settings scroll within what's left of the viewport: an area taller
                            // than the window is clipped, leaving its lower controls out of reach.
                            let left = viewport.bottom() - 10.0 - ui.min_rect().bottom() - SETTINGS_MARGIN;
                            egui::CollapsingHeader::new("Render settings").default_open(false).show(ui, |ui| {
                                ui.style_mut().override_text_style = None;
                                egui::ScrollArea::vertical().max_height(left.max(80.0)).show(ui, |ui| {
                                    let mut flags = self.debug_flags.write().expect("debug_flags RwLock poisoned");
                                    layer_toggles(ui, &mut flags);
                                    ui.separator();
                                    egui::CollapsingHeader::new("Global illumination").show(ui, |ui| {
                                        gi_controls(ui, &mut flags);
                                    });
                                });
                            });
                        });
                });
        }

        let mut open = self.debug_flags.read().expect("debug_flags RwLock poisoned").show_texture_aliasing;
        if open {
            egui::Window::new("Texture aliasing")
                .open(&mut open)
                .resizable(false)
                .show(ctx, |ui| aliasing_timeline(ui, &stats.memory.screen_textures));
            if !open {
                self.debug_flags.write().expect("debug_flags RwLock poisoned").show_texture_aliasing = false;
            }
        }
    }
}

/// Frame rate and time (between frames), CPU recording time, GPU total, and each pass's CPU (60-frame average) and GPU
/// time (a few frames old) with a bar scaled to the slowest pass.
fn frame_timings(ui: &mut egui::Ui, stats: &FrameStats) {
    let gpu_times: Vec<f32> = stats.passes.iter().filter_map(|p| p.gpu_ms).collect();
    let gpu = (!gpu_times.is_empty()).then(|| gpu_times.iter().sum::<f32>());
    ui.label(egui::RichText::new(format!("{:5.1} fps   {:6.2} ms frame", stats.fps, stats.frame_ms)).strong());
    ui.horizontal(|ui| {
        ui.label(format!("cpu {:6.2} ms", stats.cpu_ms));
        match gpu {
            Some(gpu) => {
                ui.label(format!("gpu {gpu:6.2} ms"));
                // The GPU work alone fills (most of) the frame: the frame rate is the GPU's.
                if gpu > 0.85 * stats.frame_ms {
                    ui.label(egui::RichText::new("GPU-bound").color(ui.visuals().warn_fg_color));
                }
            }
            None => {
                ui.label("gpu    n/a");
            }
        }
    });
    if stats.passes.is_empty() {
        return;
    }
    ui.separator();
    let slowest = gpu_times.iter().copied().fold(0.0f32, f32::max).max(1e-3);
    let warn = ui.visuals().warn_fg_color;
    egui::Grid::new("pass_timings").num_columns(4).spacing([10.0, 2.0]).show(ui, |ui| {
        ui.label(egui::RichText::new("pass").weak());
        ui.label(egui::RichText::new("   cpu").weak());
        ui.label(egui::RichText::new("   gpu").weak());
        ui.label("");
        ui.end_row();
        for pass in &stats.passes {
            ui.label(pass.name);
            ui.label(format!("{:6.2}", pass.avg_ms));
            match pass.gpu_ms {
                Some(gpu) => {
                    ui.label(format!("{gpu:6.2}"));
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(BAR_WIDTH, 8.0), egui::Sense::hover());
                    let width = BAR_WIDTH * (gpu / slowest).min(1.0);
                    let color = if gpu >= 0.5 * slowest { warn } else { ui.visuals().widgets.inactive.fg_stroke.color };
                    ui.painter().rect_filled(
                        egui::Rect::from_min_size(rect.left_center() - egui::vec2(0.0, 3.0), egui::vec2(width, 6.0)),
                        2.0,
                        color,
                    );
                }
                None => {
                    ui.label("     -");
                    ui.label("");
                }
            }
            ui.end_row();
        }
    });
}

fn memory(ui: &mut egui::Ui, stats: &FrameStats) {
    let mem = &stats.memory;
    let mib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);
    egui::Grid::new("arenas").striped(true).show(ui, |ui| {
        ui.label("upload arena");
        ui.label(format!("{:.2} MiB in {} page(s)", mib(mem.upload_bytes), mem.upload_pages));
        ui.end_row();
        ui.label("geometry");
        ui.label(format!(
            "{}/{} verts  {}/{} indices  (gen {})",
            mem.geometry_vertices.0, mem.geometry_vertices.1, mem.geometry_indices.0, mem.geometry_indices.1, mem.geometry_generation
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
        ui.label(format!("{} logical ➡ {} physical", mem.screen_textures.logical(), mem.screen_textures.physical()));
        ui.end_row();
    });
}

/// Toggles for each part of the lighting, the G-buffer view, and debug visualisations.
pub fn layer_toggles(ui: &mut egui::Ui, flags: &mut RenderDebugFlags) {
    ui.checkbox(&mut flags.albedo, "Albedo");
    ui.checkbox(&mut flags.indirect, "Indirect (GI)");
    ui.checkbox(&mut flags.direct, "Direct lights");
    ui.checkbox(&mut flags.shadows, "Shadows");
    ui.checkbox(&mut flags.ambient_occlusion, "Ambient occlusion");
    ui.checkbox(&mut flags.contact_shadows, "Contact shadows");
    ui.checkbox(&mut flags.shadow_mask, "Half-resolution shadows");
    ui.checkbox(&mut flags.emissive_lights, "Emissive lights");
    ui.checkbox(&mut flags.specular, "Specular");
    ui.checkbox(&mut flags.reflections, "Reflections");
    ui.add_enabled_ui(flags.reflections, |ui| {
        ui.indent("reflections_indent", |ui| {
            ui.checkbox(&mut flags.ssr, "Screen-space");
            ui.checkbox(&mut flags.reflection_probe, "Reflection probe");
        });
    });
    ui.checkbox(&mut flags.particles, "Particles");
    egui::ComboBox::from_label("View").selected_text(flags.view.label()).show_ui(ui, |ui| {
        for view in GBufferView::ALL {
            ui.selectable_value(&mut flags.view, view, view.label());
        }
    });
    ui.separator();
    ui.checkbox(&mut flags.show_bvh, "BVH AABBs");
    if flags.show_bvh {
        ui.indent("bvh_indent", |ui| {
            ui.checkbox(&mut flags.bvh_leaves_only, "Leaves only");
        });
    }
}

/// DDGI probe settings and the probe visualisation.
pub fn gi_controls(ui: &mut egui::Ui, flags: &mut RenderDebugFlags) {
    egui::Grid::new("gi_controls").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
        ui.label("Hysteresis").on_hover_text("Share of the previous probe value kept per update: higher is smoother but reacts more slowly to changes.");
        ui.add(egui::Slider::new(&mut flags.ddgi_hysteresis, 0.0..=0.995));
        ui.end_row();
        ui.label("Normal bias").on_hover_text("How far shading points move along their normal before sampling the probes, against leaks through walls.");
        ui.add(egui::Slider::new(&mut flags.ddgi_normal_bias, 0.0..=1.0).suffix(" m"));
        ui.end_row();
        ui.label("View bias").on_hover_text("How far shading points move toward the camera before sampling the probes.");
        ui.add(egui::Slider::new(&mut flags.ddgi_view_bias, 0.0..=1.0).suffix(" m"));
        ui.end_row();
        ui.label("Probe spacing").on_hover_text("Distance between probes. The grid has 12 × 6 × 12 probes, so it covers 11 × 5 × 11 spacings.");
        ui.add(egui::Slider::new(&mut flags.ddgi_probe_spacing, 0.5..=20.0).suffix(" m").logarithmic(true));
        ui.end_row();
        ui.label("Grid anchor").on_hover_text("A world point the probes are aligned to.");
        ui.horizontal(|ui| {
            for (axis, value) in ["x", "y", "z"].iter().zip(flags.ddgi_grid_anchor.iter_mut()) {
                ui.add(egui::DragValue::new(value).speed(0.05).prefix(format!("{axis} ")));
            }
        });
        ui.end_row();
    });
    ui.checkbox(&mut flags.show_probes, "Show probes");
}

/// Passes run left to right, each row is one physical texture, and each bar is
/// a logical texture spanning the passes it is live for.
fn aliasing_timeline(ui: &mut egui::Ui, report: &AliasingReport) {
    let mib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);
    let logical = report.logical_bytes();
    let physical = report.physical_bytes();
    ui.label(format!(
        "{} logical ➡ {} physical    {:.1} MiB ➡ {:.1} MiB (saved {:.1} MiB)",
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
                    "live: {} ➡ {}",
                    report.passes[texture.first], report.passes[texture.last]
                ));
                ui.label(format!("physical texture {}", texture.slot));
            });
        }
    });
}
