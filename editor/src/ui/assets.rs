use std::collections::{BTreeSet, HashMap};

use egui::{Ui, load::SizedTexture};

use crate::asset_browser::{AssetDetails, AssetEntry, AssetInfo, AssetKind, BuildSet};
use crate::state::EditorState;

const TILE: egui::Vec2 = egui::vec2(96.0, 112.0);
const THUMB: f32 = 72.0;

/// Bottom "Assets" panel: folder list, tile grid with thumbnails, and details of the selection.
/// Thumbnail textures live here, on the render thread, keyed by asset path.
#[derive(Default)]
pub struct AssetPanel {
    textures: HashMap<String, egui::TextureHandle>,
    generation: u64,
}

impl AssetPanel {
    pub fn draw(&mut self, root: &mut Ui, state: &mut EditorState) {
        if state.project_dir.is_none() {
            return;
        }
        if self.generation != state.assets.generation {
            self.textures.clear();
            self.generation = state.assets.generation;
        }

        egui::Panel::bottom("assets")
            .resizable(true)
            .min_size(120.0)
            .default_size(240.0)
            .show(root, |ui| {
                self.draw_header(ui, state);
                ui.separator();
                egui::Panel::left("asset_folders")
                    .resizable(true)
                    .default_size(170.0)
                    .min_size(110.0)
                    .show(ui, |ui| draw_folders(ui, state));
                egui::Panel::right("asset_details")
                    .resizable(true)
                    .default_size(280.0)
                    .min_size(200.0)
                    .show(ui, |ui| self.draw_details(ui, state));
                egui::CentralPanel::default().show(ui, |ui| self.draw_grid(ui, state));
            });
    }

    fn draw_header(&mut self, ui: &mut Ui, state: &mut EditorState) {
        let assets = &mut state.assets;
        ui.horizontal(|ui| {
            ui.strong("Assets");
            if ui.button("⟳").on_hover_text("Rescan project files").clicked() {
                assets.refresh_requested = true;
            }
            ui.add(egui::TextEdit::singleline(&mut assets.search).hint_text("🔍 search").desired_width(160.0));
            ui.separator();
            for kind in AssetKind::ALL {
                let mut shown = !assets.hidden_kinds.contains(&kind);
                let toggle = ui.toggle_value(&mut shown, kind.icon()).on_hover_text(kind.label());
                if toggle.changed() {
                    if shown {
                        assets.hidden_kinds.remove(&kind);
                    } else {
                        assets.hidden_kinds.insert(kind);
                    }
                }
            }
            ui.separator();
            let summary = match &assets.build_set {
                BuildSet::Pending => format!("{} files · working out build contents…", assets.entries.len()),
                BuildSet::Ready(set) => format!("{} files · {} ship in a build", assets.entries.len(), set.len()),
                BuildSet::Error(e) => format!("{} files · build: {e}", assets.entries.len()),
            };
            ui.label(egui::RichText::new(summary).weak());
        });
    }

    fn texture_for(&mut self, ctx: &egui::Context, path: &str, info: Option<&AssetInfo>) -> Option<egui::TextureHandle> {
        if let Some(tex) = self.textures.get(path) {
            return Some(tex.clone());
        }
        let thumb = info?.thumbnail.as_ref()?;
        let image = egui::ColorImage::from_rgba_unmultiplied([thumb.width as usize, thumb.height as usize], &thumb.rgba);
        let tex = ctx.load_texture(format!("asset-thumb:{path}"), image, egui::TextureOptions::LINEAR);
        self.textures.insert(path.to_string(), tex.clone());
        Some(tex)
    }

    fn draw_grid(&mut self, ui: &mut Ui, state: &mut EditorState) {
        let search = state.assets.search.to_lowercase();
        let folder = state.assets.folder.clone();
        let visible: Vec<AssetEntry> = state
            .assets
            .entries
            .iter()
            .filter(|e| folder.is_empty() || e.folder() == folder || e.folder().starts_with(&format!("{folder}/")))
            .filter(|e| !state.assets.hidden_kinds.contains(&e.kind))
            .filter(|e| search.is_empty() || e.path.to_lowercase().contains(&search))
            .cloned()
            .collect();

        if visible.is_empty() {
            ui.label(egui::RichText::new("No matching files.").weak());
            return;
        }

        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = egui::vec2(6.0, 6.0);
                for entry in &visible {
                    self.draw_tile(ui, state, entry);
                }
            });
        });
    }

    fn draw_tile(&mut self, ui: &mut Ui, state: &mut EditorState, entry: &AssetEntry) {
        let (rect, response) = ui.allocate_exact_size(TILE, egui::Sense::click());
        if !ui.is_rect_visible(rect) {
            return;
        }

        let info = state.assets.info.get(&entry.path).cloned();
        if entry.kind.inspectable() && info.is_none() && !state.assets.pending.contains(&entry.path) {
            state.assets.wanted.push(entry.path.clone());
        }

        let selected = state.assets.selected.as_deref() == Some(entry.path.as_str());
        let in_build = state.assets.in_build(&entry.path);
        let visuals = ui.visuals().clone();
        let painter = ui.painter_at(rect);

        if selected {
            painter.rect_filled(rect, 4.0, visuals.selection.bg_fill.linear_multiply(0.6));
        } else if response.hovered() {
            painter.rect_filled(rect, 4.0, visuals.widgets.hovered.weak_bg_fill);
        }

        let thumb_rect = egui::Rect::from_center_size(
            egui::pos2(rect.center().x, rect.top() + 6.0 + THUMB / 2.0),
            egui::vec2(THUMB, THUMB),
        );
        let dim = in_build == Some(false);
        let tint = if dim { egui::Color32::from_white_alpha(110) } else { egui::Color32::WHITE };
        match self.texture_for(ui.ctx(), &entry.path, info.as_deref()) {
            Some(tex) => {
                let size = fit(tex.size_vec2(), thumb_rect.size());
                egui::Image::from_texture(SizedTexture::new(tex.id(), size))
                    .tint(tint)
                    .paint_at(ui, egui::Rect::from_center_size(thumb_rect.center(), size));
            }
            None => {
                painter.rect_filled(thumb_rect, 6.0, visuals.extreme_bg_color);
                painter.text(
                    thumb_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    entry.kind.icon(),
                    egui::FontId::proportional(30.0),
                    visuals.text_color().gamma_multiply(if dim { 0.4 } else { 0.9 }),
                );
            }
        }

        let name_color = if dim { visuals.weak_text_color() } else { visuals.text_color() };
        painter.text(
            egui::pos2(rect.center().x, thumb_rect.bottom() + 6.0),
            egui::Align2::CENTER_TOP,
            ellipsize(entry.file_name(), 15),
            egui::FontId::proportional(11.5),
            name_color,
        );
        if in_build == Some(true) {
            painter.circle_filled(thumb_rect.right_top() + egui::vec2(-2.0, 2.0), 3.5, egui::Color32::from_rgb(90, 190, 120));
        }

        let hover = format!(
            "{}\n{} · {}\n{}",
            entry.path,
            entry.kind.label(),
            build_tool::format_bytes(entry.size),
            build_label(in_build)
        );
        let response = response.on_hover_text(hover);
        if response.clicked() {
            state.assets.selected = Some(entry.path.clone());
        }
        if response.double_clicked() {
            open_entry(state, entry);
        }
        response.context_menu(|ui| entry_actions(ui, state, entry));
    }

    fn draw_details(&mut self, ui: &mut Ui, state: &mut EditorState) {
        let Some(entry) = state.assets.selected.as_deref().and_then(|p| state.assets.entry(p)).cloned() else {
            ui.label(egui::RichText::new("Select an asset to see its details.").weak());
            return;
        };
        let info = state.assets.info.get(&entry.path).cloned();
        let in_build = state.assets.in_build(&entry.path);

        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading(entry.kind.icon());
                ui.strong(entry.file_name());
            });
            ui.label(egui::RichText::new(&entry.path).small().weak());

            if let Some(tex) = self.texture_for(ui.ctx(), &entry.path, info.as_deref()) {
                let max = egui::vec2(ui.available_width().min(256.0), 160.0);
                ui.add(egui::Image::from_texture(SizedTexture::new(tex.id(), fit(tex.size_vec2(), max))));
            }

            ui.add_space(4.0);
            egui::Grid::new("asset_details_grid").num_columns(2).spacing([12.0, 3.0]).show(ui, |ui| {
                row(ui, "Type", entry.kind.label());
                row(ui, "Size", &build_tool::format_bytes(entry.size));
                row(ui, "Build", build_label(in_build));
                match info.as_deref().map(|i| &i.details) {
                    Some(AssetDetails::Texture { width, height, color }) => {
                        row(ui, "Dimensions", &format!("{width} × {height}"));
                        row(ui, "Format", color);
                    }
                    Some(AssetDetails::Model(stats)) => {
                        row(ui, "Format", if stats.binary { "glTF binary (.glb)" } else { "glTF (.gltf)" });
                        row(ui, "Nodes", &stats.nodes.to_string());
                        row(ui, "Meshes", &format!("{} ({} primitives)", stats.meshes, stats.primitives));
                        row(ui, "Vertices", &stats.vertices.to_string());
                        row(ui, "Triangles", &stats.triangles.to_string());
                        row(ui, "Materials", &stats.materials.to_string());
                        row(ui, "Textures", &stats.textures.to_string());
                        if stats.cameras > 0 {
                            row(ui, "Cameras", &stats.cameras.to_string());
                        }
                        if stats.lights > 0 {
                            row(ui, "Lights", &stats.lights.to_string());
                        }
                        if stats.animations > 0 {
                            row(ui, "Animations", &format!("{} (not played yet)", stats.animations));
                        }
                    }
                    Some(AssetDetails::Error(e)) => row(ui, "Error", e),
                    Some(AssetDetails::None) => {}
                    None if entry.kind.inspectable() => row(ui, "", "inspecting…"),
                    None => {}
                }
            });

            if let Some(AssetDetails::Model(stats)) = info.as_deref().map(|i| &i.details)
                && !stats.dependencies.is_empty()
            {
                ui.add_space(6.0);
                egui::CollapsingHeader::new(format!("Files used ({})", stats.dependencies.len()))
                    .default_open(true)
                    .show(ui, |ui| {
                        for (uri, resolved) in &stats.dependencies {
                            match resolved {
                                Some(path) if state.assets.entry(path).is_some() => {
                                    if ui.link(egui::RichText::new(path).small()).clicked() {
                                        state.assets.selected = Some(path.clone());
                                    }
                                }
                                Some(path) => {
                                    ui.colored_label(ui.visuals().error_fg_color, format!("✖ missing: {path}"));
                                }
                                None => {
                                    ui.colored_label(ui.visuals().error_fg_color, format!("✖ outside project: {uri}"));
                                }
                            }
                        }
                    });
            }

            ui.add_space(8.0);
            ui.horizontal_wrapped(|ui| entry_actions(ui, state, &entry));
        });
    }
}

fn draw_folders(ui: &mut Ui, state: &mut EditorState) {
    let mut folders = BTreeSet::new();
    for entry in &state.assets.entries {
        let mut folder = entry.folder();
        while !folder.is_empty() {
            folders.insert(folder.to_string());
            folder = folder.rsplit_once('/').map(|(parent, _)| parent).unwrap_or("");
        }
    }

    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        if ui.selectable_label(state.assets.folder.is_empty(), "🗀 All files").clicked() {
            state.assets.folder.clear();
        }
        for folder in folders {
            let depth = folder.matches('/').count();
            let name = folder.rsplit('/').next().unwrap_or(&folder);
            ui.horizontal(|ui| {
                ui.add_space(12.0 * (depth + 1) as f32);
                if ui.selectable_label(state.assets.folder == folder, format!("🗀 {name}")).clicked() {
                    state.assets.folder = folder.clone();
                }
            });
        }
    });
}

/// Buttons shared by the context menu and the details pane.
fn entry_actions(ui: &mut Ui, state: &mut EditorState, entry: &AssetEntry) {
    if matches!(entry.kind, AssetKind::Scene | AssetKind::Model) && ui.button("Open").clicked() {
        open_entry(state, entry);
        ui.close();
    }

    let is_scene_file = entry.kind == AssetKind::Scene;
    if let Some(project) = state.project.as_mut() {
        if is_scene_file {
            let mut is_default = project.default_scene.as_deref() == Some(entry.path.as_str());
            if ui.checkbox(&mut is_default, "Default scene").on_hover_text("Opened on launch and the root of every build").changed() {
                project.default_scene = is_default.then(|| entry.path.clone());
                ui.close();
            }
        }

        let mut included = project.build.include.contains(&entry.path);
        let toggle = ui
            .checkbox(&mut included, "Always ship")
            .on_hover_text("Include in builds even if no scene references it, for assets loaded by path at runtime");
        if toggle.changed() {
            if included {
                project.build.include.push(entry.path.clone());
            } else {
                project.build.include.retain(|p| p != &entry.path);
            }
            ui.close();
        }
    }

    if ui.button("Copy path").clicked() {
        ui.ctx().copy_text(entry.path.clone());
        ui.close();
    }
}

fn open_entry(state: &mut EditorState, entry: &AssetEntry) {
    if !matches!(entry.kind, AssetKind::Scene | AssetKind::Model) {
        return;
    }
    if let Some(dir) = &state.project_dir {
        state.load_path = Some(dir.join(&entry.path));
    }
}

fn row(ui: &mut Ui, label: &str, value: &str) {
    ui.label(egui::RichText::new(label).weak());
    ui.label(value);
    ui.end_row();
}

fn build_label(in_build: Option<bool>) -> &'static str {
    match in_build {
        Some(true) => "✔ ships in builds",
        Some(false) => "not referenced, left out of builds",
        None => "unknown",
    }
}

fn fit(size: egui::Vec2, max: egui::Vec2) -> egui::Vec2 {
    let scale = (max.x / size.x.max(1.0)).min(max.y / size.y.max(1.0)).min(1.0);
    size * scale
}

fn ellipsize(name: &str, max_chars: usize) -> String {
    if name.chars().count() <= max_chars {
        return name.to_string();
    }
    let head: String = name.chars().take(max_chars - 1).collect();
    format!("{head}…")
}
