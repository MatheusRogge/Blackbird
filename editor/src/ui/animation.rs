//! The Animation panel: clips, a track per animated property, a timeline to scrub and play, and a
//! curve view of the selected track where keys and cubic tangents are dragged.

use egui::{Color32, Pos2, Rect, Sense, Stroke, Ui, pos2, vec2};
use engine_core::animation::{Interpolation, Key};

use crate::animation::{self, EditorClip, EditorTarget, components};
use crate::math::{euler_to_quat, quat_to_euler};
use crate::state::EditorState;
use engine::gltf::animation::Target;

const PRESETS: [(Interpolation, &str); 7] = [
    (Interpolation::Step, "Step"),
    (Interpolation::Linear, "Linear"),
    (Interpolation::Smooth, "Smooth"),
    (Interpolation::EaseIn, "Ease in"),
    (Interpolation::EaseOut, "Ease out"),
    (Interpolation::EaseInOut, "Ease in-out"),
    (Interpolation::Cubic, "Cubic (tangents)"),
];
const COMPONENT_COLORS: [Color32; 4] = [
    Color32::from_rgb(230, 90, 90),
    Color32::from_rgb(110, 200, 110),
    Color32::from_rgb(100, 150, 240),
    Color32::from_rgb(220, 220, 220),
];
const RULER_HEIGHT: f32 = 18.0;
const KEY_RADIUS: f32 = 5.0;
/// Length of a drawn tangent handle, in points.
const TANGENT_HANDLE: f32 = 50.0;

pub fn preset_name(interpolation: Interpolation) -> &'static str {
    PRESETS.iter().find(|(i, _)| *i == interpolation).map_or("?", |(_, n)| n)
}

pub fn draw(root: &mut Ui, state: &mut EditorState) {
    if !state.timeline.show || state.document.is_none() {
        return;
    }
    egui::Panel::bottom("animation").resizable(true).min_size(160.0).default_size(260.0).show(root, |ui| {
        draw_header(ui, state);
        ui.separator();
        let Some(clip_index) = state.timeline.clip.filter(|&c| c < state.clips.len()) else {
            ui.label(egui::RichText::new("No clip. Add one with + Clip, or key a property with ◆ in the inspector.").weak());
            return;
        };
        egui::Panel::left("animation_tracks")
            .resizable(true)
            .default_size(230.0)
            .min_size(160.0)
            .show(ui, |ui| draw_tracks(ui, state, clip_index));
        egui::Panel::right("animation_key")
            .resizable(true)
            .default_size(250.0)
            .min_size(200.0)
            .show(ui, |ui| draw_key_inspector(ui, state, clip_index));
        egui::CentralPanel::default().show(ui, |ui| draw_curves(ui, state, clip_index));
    });
}

fn draw_header(ui: &mut Ui, state: &mut EditorState) {
    ui.horizontal_wrapped(|ui| {
        ui.strong("Animation");
        let selected = state.timeline.clip.and_then(|c| state.clips.get(c)).map_or("—".to_string(), |c| c.name.clone());
        let mut chosen = None;
        egui::ComboBox::from_id_salt("animation_clip").selected_text(selected).show_ui(ui, |ui| {
            for (i, clip) in state.clips.iter().enumerate() {
                if ui.selectable_label(state.timeline.clip == Some(i), &clip.name).clicked() {
                    chosen = Some(i);
                }
            }
        });
        if chosen.is_some() {
            select_clip(state, chosen);
        }
        if ui.button("+ Clip").clicked() {
            let name = format!("Clip {}", state.clips.len() + 1);
            state.clips.push(EditorClip::new(name));
            select_clip(state, Some(state.clips.len() - 1));
        }
        let Some(index) = state.timeline.clip.filter(|&c| c < state.clips.len()) else { return };

        ui.separator();
        let timeline = &mut state.timeline;
        let play = if timeline.playing { "⏸" } else { "▶" };
        if ui.button(play).on_hover_text("Play / pause (previews in the viewport)").clicked() {
            timeline.playing = !timeline.playing;
            timeline.preview = true;
        }
        if ui.button("⏹").on_hover_text("Stop: back to the start, preview off").clicked() {
            timeline.playing = false;
            timeline.time = 0.0;
            timeline.preview = false;
        }
        ui.toggle_value(&mut timeline.preview, "👁 Preview")
            .on_hover_text("Show the clip at the playhead in the viewport. Off shows the scene as edited.");
        let duration = state.clips[index].duration();
        ui.add(egui::DragValue::new(&mut state.timeline.time).speed(0.01).range(0.0..=duration.max(0.0)).suffix(" s"));
        ui.label(egui::RichText::new(format!("/ {duration:.2} s")).weak());

        ui.separator();
        let clip = &mut state.clips[index];
        ui.add(egui::TextEdit::singleline(&mut clip.name).desired_width(120.0));
        ui.checkbox(&mut clip.looping, "Loop");
        ui.checkbox(&mut clip.autoplay, "Autoplay").on_hover_text("Starts playing when the game loads the scene.");
        ui.add(egui::DragValue::new(&mut clip.speed).speed(0.01).range(0.0..=10.0).prefix("Speed ×"));
        if let Some(standard) = clip.animation {
            ui.label(egui::RichText::new(format!("+ glTF animation {standard} (read-only)")).weak());
        }
        if ui.button("🗑").on_hover_text("Delete clip").clicked() {
            state.clips.remove(index);
            let next = if state.clips.is_empty() { None } else { Some(index.min(state.clips.len() - 1)) };
            select_clip(state, next);
        }
    });
}

fn select_clip(state: &mut EditorState, clip: Option<usize>) {
    let timeline = &mut state.timeline;
    timeline.clip = clip;
    timeline.track = None;
    timeline.key = None;
    timeline.time = 0.0;
    timeline.playing = false;
}

fn draw_tracks(ui: &mut Ui, state: &mut EditorState, clip_index: usize) {
    let tracked: Vec<EditorTarget> = state.clips[clip_index].tracks.iter().map(|t| t.target).collect();
    let addable: Vec<EditorTarget> = animation::animatable(state).into_iter().filter(|t| !tracked.contains(t)).collect();
    ui.horizontal(|ui| {
        ui.strong("Tracks");
        ui.add_enabled_ui(!addable.is_empty(), |ui| {
            ui.menu_button("+ Track", |ui| {
                for target in &addable {
                    if ui.button(animation::label(state, *target)).clicked() {
                        key_at_playhead(state, *target);
                        ui.close();
                    }
                }
            })
            .response
            .on_disabled_hover_text("Select a node or light to animate its properties.");
        });
    });
    egui::ScrollArea::vertical().show(ui, |ui| {
        let mut delete = None;
        for (i, target) in tracked.iter().enumerate() {
            ui.horizontal(|ui| {
                let label = animation::label(state, *target);
                let keys = state.clips[clip_index].tracks[i].curve.keys.len();
                if ui.selectable_label(state.timeline.track == Some(i), format!("{label} ({keys})")).clicked() {
                    state.timeline.track = Some(i);
                    state.timeline.key = None;
                }
                if ui.small_button("🗑").on_hover_text("Delete track").clicked() {
                    delete = Some(i);
                }
            });
        }
        if let Some(i) = delete {
            state.clips[clip_index].tracks.remove(i);
            state.timeline.track = None;
            state.timeline.key = None;
        }
    });
}

/// Keys `target`'s current (inspector) value at the playhead of the selected clip, creating a clip
/// if there is none, and selects the key.
pub fn key_at_playhead(state: &mut EditorState, target: EditorTarget) {
    let Some(value) = animation::current_value(state, target) else { return };
    let clip = match state.timeline.clip.filter(|&c| c < state.clips.len()) {
        Some(clip) => clip,
        None => {
            state.clips.push(EditorClip::new(format!("Clip {}", state.clips.len() + 1)));
            state.clips.len() - 1
        }
    };
    state.timeline.clip = Some(clip);
    state.timeline.show = true;
    let time = state.timeline.time;
    let track = state.clips[clip].set_key(target, time, value);
    state.timeline.track = Some(track);
    state.timeline.key = state.clips[clip].tracks[track].curve.keys.iter().position(|k| (k.time - time).abs() < 1e-3);
}

fn draw_key_inspector(ui: &mut Ui, state: &mut EditorState, clip_index: usize) {
    let Some(track_index) = state.timeline.track.filter(|&t| t < state.clips[clip_index].tracks.len()) else {
        ui.label(egui::RichText::new("Select a track.").weak());
        return;
    };
    let time = state.timeline.time;
    let track = &mut state.clips[clip_index].tracks[track_index];
    let is_rotation = matches!(track.target, EditorTarget::Scene(Target::Rotation(_)));
    let n = components(track.target);
    if ui.button("◆ Key at playhead").on_hover_text("Adds a key with the curve's value at the playhead.").clicked() {
        let value = track.curve.sample(time);
        let interpolation = track.curve.keys.iter().rev().find(|k| k.time <= time).map_or(Interpolation::Linear, |k| k.interpolation);
        match track.curve.keys.iter().position(|k| (k.time - time).abs() < 1e-3) {
            Some(i) => state.timeline.key = Some(i),
            None => {
                track.curve.keys.push(Key::new(time, value, interpolation));
                track.curve.keys.sort_by(|a, b| a.time.total_cmp(&b.time));
                state.timeline.key = track.curve.keys.iter().position(|k| k.time == time);
            }
        }
    }
    ui.separator();
    let Some(key_index) = state.timeline.key.filter(|&k| k < track.curve.keys.len()) else {
        ui.label(egui::RichText::new("Click a key in the curve view.").weak());
        return;
    };
    let (before, after) = neighbor_times(&track.curve.keys, key_index);
    let key = &mut track.curve.keys[key_index];
    egui::Grid::new("animation_key_grid").num_columns(2).spacing([10.0, 4.0]).show(ui, |ui| {
        ui.label("Time");
        ui.add(egui::DragValue::new(&mut key.time).speed(0.01).range(before..=after).suffix(" s"));
        ui.end_row();

        ui.label("Value");
        ui.horizontal_wrapped(|ui| {
            if is_rotation {
                let euler = quat_to_euler(key.value);
                let mut edited = euler;
                for (c, axis) in ["X", "Y", "Z"].iter().enumerate() {
                    ui.add(egui::DragValue::new(&mut edited[c]).speed(0.5).prefix(format!("{axis} ")).suffix("°"));
                }
                if edited != euler {
                    key.value = euler_to_quat(edited);
                }
            } else {
                for c in 0..n {
                    let speed = drag_speed(key.value[c]);
                    ui.add(egui::DragValue::new(&mut key.value[c]).speed(speed));
                }
            }
        });
        ui.end_row();

        ui.label("To next key");
        egui::ComboBox::from_id_salt("animation_interpolation").selected_text(preset_name(key.interpolation)).show_ui(ui, |ui| {
            for (preset, name) in PRESETS {
                ui.selectable_value(&mut key.interpolation, preset, name);
            }
        });
        ui.end_row();

        if !is_rotation {
            for (label, tangent) in [("In slope", &mut key.in_tangent), ("Out slope", &mut key.out_tangent)] {
                ui.label(label).on_hover_text("Value per second, used by Cubic segments.");
                ui.horizontal_wrapped(|ui| {
                    for value in tangent.iter_mut().take(n) {
                        ui.add(egui::DragValue::new(value).speed(0.05));
                    }
                });
                ui.end_row();
            }
        }
    });
    if ui.button("Delete key").clicked() {
        track.curve.keys.remove(key_index);
        state.timeline.key = None;
    }
}

/// The times a key can move between without passing its neighbours.
fn neighbor_times(keys: &[Key], i: usize) -> (f32, f32) {
    let before = if i > 0 { keys[i - 1].time + 1e-3 } else { 0.0 };
    let after = keys.get(i + 1).map_or(f32::MAX, |k| k.time - 1e-3);
    (before, after.max(before))
}

fn drag_speed(value: f32) -> f64 {
    (f64::from(value.abs()) * 0.01).max(0.001)
}

/// Maps time / value to the curve view and back.
struct View {
    rect: Rect,
    t_max: f32,
    v_min: f32,
    v_max: f32,
}

impl View {
    fn x(&self, t: f32) -> f32 {
        self.rect.left() + t / self.t_max * self.rect.width()
    }

    fn y(&self, v: f32) -> f32 {
        self.rect.bottom() - (v - self.v_min) / (self.v_max - self.v_min) * self.rect.height()
    }

    fn pos(&self, t: f32, v: f32) -> Pos2 {
        pos2(self.x(t), self.y(v))
    }

    fn time_at(&self, x: f32) -> f32 {
        ((x - self.rect.left()) / self.rect.width() * self.t_max).max(0.0)
    }

    fn dt(&self, dx: f32) -> f32 {
        dx / self.rect.width() * self.t_max
    }

    fn dv(&self, dy: f32) -> f32 {
        -dy / self.rect.height() * (self.v_max - self.v_min)
    }
}

fn draw_curves(ui: &mut Ui, state: &mut EditorState, clip_index: usize) {
    let full = ui.available_rect_before_wrap();
    let ruler = Rect::from_min_size(full.min, vec2(full.width(), RULER_HEIGHT));
    let area = Rect::from_min_max(pos2(full.left(), ruler.bottom()), full.max).shrink2(vec2(8.0, 10.0));
    let painter = ui.painter_at(full);
    painter.rect_filled(full, 2.0, ui.visuals().extreme_bg_color);

    let clip = &state.clips[clip_index];
    let t_max = (clip.duration().max(state.timeline.time) * 1.05).max(1.0);
    let track_index = state.timeline.track.filter(|&t| t < clip.tracks.len());

    let (v_min, v_max) = track_index.map_or((0.0, 1.0), |t| value_range(&clip.tracks[t]));
    let view = View { rect: area, t_max, v_min, v_max };

    // Ruler: ticks, and scrubbing.
    let step = nice_step(t_max / (full.width() / 70.0).max(1.0));
    let text = ui.visuals().weak_text_color();
    let mut tick = 0.0;
    while tick <= t_max {
        let x = view.x(tick);
        painter.line_segment([pos2(x, ruler.top()), pos2(x, full.bottom())], Stroke::new(1.0, ui.visuals().faint_bg_color));
        painter.text(pos2(x + 2.0, ruler.top() + 2.0), egui::Align2::LEFT_TOP, format!("{tick:.2}"), egui::FontId::monospace(10.0), text);
        tick += step;
    }
    let scrub = ui.interact(ruler, ui.scope_id().with("animation_ruler"), Sense::click_and_drag());
    if (scrub.dragged() || scrub.clicked())
        && let Some(pointer) = scrub.interact_pointer_pos()
    {
        state.timeline.time = view.time_at(pointer.x).min(clip.duration().max(0.0));
        state.timeline.preview = true;
        state.timeline.playing = false;
    }

    // Every track's keys, faintly, so the timing of the whole clip is visible.
    for (i, track) in clip.tracks.iter().enumerate() {
        if Some(i) == track_index {
            continue;
        }
        for key in &track.curve.keys {
            painter.circle_filled(pos2(view.x(key.time), ruler.bottom() + 4.0), 2.5, text);
        }
    }

    let Some(track_index) = track_index else {
        painter.text(area.center(), egui::Align2::CENTER_CENTER, "Select a track to edit its curve", egui::FontId::proportional(13.0), text);
        draw_playhead(&painter, &view, full, state.timeline.time);
        return;
    };

    let track = &state.clips[clip_index].tracks[track_index];
    let n = components(track.target);
    let is_rotation = matches!(track.target, EditorTarget::Scene(Target::Rotation(_)));
    let samples = (area.width() as usize).clamp(32, 600);
    for (c, &color) in COMPONENT_COLORS.iter().enumerate().take(n) {
        let points: Vec<Pos2> = (0..=samples)
            .map(|i| {
                let t = t_max * i as f32 / samples as f32;
                view.pos(t, track.curve.sample(t)[c])
            })
            .collect();
        painter.add(egui::Shape::line(points, Stroke::new(1.5, color)));
    }
    if v_min < 0.0 && v_max > 0.0 {
        painter.line_segment([view.pos(0.0, 0.0), view.pos(t_max, 0.0)], Stroke::new(1.0, ui.visuals().weak_text_color()));
    }

    // Registered before the keys and handles so they, drawn on top, get the pointer first.
    let background = ui.interact(area, ui.scope_id().with("animation_curve_bg"), Sense::click());

    // Keys: click to select, drag to move (time, and the grabbed component's value).
    let selected = state.timeline.key;
    let mut select = None;
    let keys = state.clips[clip_index].tracks[track_index].curve.keys.clone();
    for (k, key) in keys.iter().enumerate() {
        for (c, &color) in COMPONENT_COLORS.iter().enumerate().take(n) {
            let pos = view.pos(key.time, key.value[c]);
            let id = ui.scope_id().with(("animation_key", clip_index, track_index, k, c));
            let response = ui.interact(Rect::from_center_size(pos, vec2(KEY_RADIUS * 3.0, KEY_RADIUS * 3.0)), id, Sense::click_and_drag());
            let is_selected = selected == Some(k);
            let fill = if is_selected { Color32::WHITE } else { color };
            painter.add(egui::Shape::convex_polygon(
                diamond(pos, if response.hovered() || is_selected { KEY_RADIUS + 1.5 } else { KEY_RADIUS }),
                fill,
                Stroke::new(1.0, Color32::BLACK),
            ));
            if response.clicked() || response.drag_started() {
                select = Some(k);
            }
            if response.dragged() {
                let delta = response.drag_delta();
                let (before, after) = neighbor_times(&keys, k);
                let key = &mut state.clips[clip_index].tracks[track_index].curve.keys[k];
                key.time = (key.time + view.dt(delta.x)).clamp(before, after);
                if !is_rotation {
                    key.value[c] += view.dv(delta.y);
                }
            }
        }
    }
    if let Some(k) = select {
        state.timeline.key = Some(k);
    }

    // Tangent handles of the selected key, where a cubic segment uses them.
    if let Some(k) = state.timeline.key.filter(|&k| k < keys.len())
        && !is_rotation
    {
        let incoming = k > 0 && keys[k - 1].interpolation == Interpolation::Cubic;
        let outgoing = keys[k].interpolation == Interpolation::Cubic && k + 1 < keys.len();
        for (side, active) in [(-1.0f32, incoming), (1.0, outgoing)] {
            if !active {
                continue;
            }
            for (c, &color) in COMPONENT_COLORS.iter().enumerate().take(n) {
                let key = state.clips[clip_index].tracks[track_index].curve.keys[k];
                let slope = if side < 0.0 { key.in_tangent[c] } else { key.out_tangent[c] };
                let origin = view.pos(key.time, key.value[c]);
                // A handle TANGENT_HANDLE points long along the slope, as drawn on screen.
                let direction = vec2(view.dt(1.0), slope * view.dt(1.0));
                let screen = vec2(1.0, view.y(key.value[c] + direction.y) - origin.y).normalized() * TANGENT_HANDLE * side;
                let handle = origin + screen;
                painter.line_segment([origin, handle], Stroke::new(1.0, color));
                let id = ui.scope_id().with(("animation_tangent", clip_index, track_index, k, c, side as i8));
                let response = ui.interact(Rect::from_center_size(handle, vec2(10.0, 10.0)), id, Sense::drag());
                painter.circle(handle, 4.0, ui.visuals().extreme_bg_color, Stroke::new(1.5, color));
                if response.dragged()
                    && let Some(pointer) = response.interact_pointer_pos()
                {
                    let dt = view.dt(pointer.x - origin.x);
                    if dt.abs() > 1e-5 {
                        let slope = view.dv(pointer.y - origin.y) / dt;
                        let key = &mut state.clips[clip_index].tracks[track_index].curve.keys[k];
                        if side < 0.0 { key.in_tangent[c] = slope } else { key.out_tangent[c] = slope }
                    }
                }
            }
        }
    }

    // Double-click empty space: a key on the curve there.
    if background.double_clicked()
        && let Some(pointer) = background.interact_pointer_pos()
    {
        let time = view.time_at(pointer.x);
        let curve = &mut state.clips[clip_index].tracks[track_index].curve;
        if !curve.keys.iter().any(|k| (k.time - time).abs() < 1e-3) {
            let value = curve.sample(time);
            let interpolation = curve.keys.iter().rev().find(|k| k.time <= time).map_or(Interpolation::Linear, |k| k.interpolation);
            curve.keys.push(Key::new(time, value, interpolation));
            curve.keys.sort_by(|a, b| a.time.total_cmp(&b.time));
            state.timeline.key = curve.keys.iter().position(|k| k.time == time);
        }
    }

    draw_playhead(&painter, &view, full, state.timeline.time);
}

fn draw_playhead(painter: &egui::Painter, view: &View, full: Rect, time: f32) {
    let x = view.x(time);
    painter.line_segment([pos2(x, full.top()), pos2(x, full.bottom())], Stroke::new(1.5, Color32::from_rgb(255, 170, 40)));
}

fn diamond(center: Pos2, r: f32) -> Vec<Pos2> {
    vec![center + vec2(0.0, -r), center + vec2(r, 0.0), center + vec2(0.0, r), center + vec2(-r, 0.0)]
}

/// The value range the curve view shows for `track`: its keys and the curve between them, padded.
fn value_range(track: &animation::EditorTrack) -> (f32, f32) {
    let n = components(track.target);
    let duration = track.curve.duration();
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for i in 0..=64 {
        let v = track.curve.sample(duration * i as f32 / 64.0);
        for &c in &v[..n] {
            lo = lo.min(c);
            hi = hi.max(c);
        }
    }
    if lo > hi {
        return (0.0, 1.0);
    }
    let pad = ((hi - lo) * 0.1).max(hi.abs().max(lo.abs()) * 0.05).max(0.05);
    (lo - pad, hi + pad)
}

/// A round step near `raw`: 1, 2 or 5 × a power of ten.
fn nice_step(raw: f32) -> f32 {
    let magnitude = 10f32.powf(raw.max(1e-3).log10().floor());
    let normalized = raw / magnitude;
    let nice = if normalized <= 1.0 { 1.0 } else if normalized <= 2.0 { 2.0 } else if normalized <= 5.0 { 5.0 } else { 10.0 };
    nice * magnitude
}
