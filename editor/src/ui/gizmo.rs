//! Viewport transform gizmo for the selected glTF node, light or object, light and player-start
//! markers, click-to-select, and the right-click Add menu.
//!
//! Translate and rotate act along world axes. Scale acts along the target's own
//! axes, and the white centre square scales uniformly. Node edits are written to
//! `EditorState::node_edits` (the plugin pushes them into the mesh entities); light
//! edits go straight into the light's `LightData`.

use egui::{Color32, Id, LayerId, Order, Pos2, Rect, Sense, Shape, Stroke};
use engine::gltf::objects::PLAYER_EYE_HEIGHT;
use engine::rendering::camera::Camera;
use engine::rendering::mesh::Mesh;
use engine_core::world::World;
use ultraviolet::{Mat4, Vec3, Vec4};

use crate::math::{self, Quat};
use crate::state::{AddKind, EditorState, GizmoMode, LightData, ObjectKind, Placement, Selection};

const GIZMO_PX: f32 = 90.0;
const PICK_PX: f32 = 8.0;
const MIN_AXIS_PX: f32 = 10.0;
const RING_SEGMENTS: usize = 64;
const MIN_SCALE: f32 = 1e-3;
const UNIFORM_SCALE_PER_PX: f32 = 0.01;
const MARKER_RADIUS_PX: f32 = 6.0;
const SPOT_LINE_PX: f32 = 40.0;
/// Distance of the sky light's sun from the scene origin, just outside the rotate rings.
const SUN_DISTANCE_PX: f32 = 120.0;
const SUN_RADIUS_PX: f32 = 8.0;
const SUN_BELOW_HORIZON: Color32 = Color32::from_gray(110);
const MARK_STROKE_PX: f32 = 2.0;
/// Dark outline under every marker so it reads against bright lights and materials.
const HALO: Color32 = Color32::from_black_alpha(200);
const HALO_PX: f32 = 2.0;
const ALL_AXES: [bool; 3] = [true; 3];
const MODES: [GizmoMode; 3] = [GizmoMode::Translate, GizmoMode::Rotate, GizmoMode::Scale];

const AXIS_COLORS: [Color32; 3] = [
    Color32::from_rgb(230, 70, 70),
    Color32::from_rgb(90, 200, 90),
    Color32::from_rgb(70, 120, 240),
];
const ACTIVE_COLOR: Color32 = Color32::from_rgb(255, 210, 60);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Handle {
    Axis(usize),
    Uniform,
}

/// What the gizmo edits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Target {
    Node(usize),
    Light(usize),
    Object(u64),
}

/// Open right-click Add menu: where it was opened and what it places at.
#[derive(Clone, Copy)]
struct AddMenu {
    pos: Pos2,
    placement: Placement,
}

/// Placement when nothing is hit and the ray misses the ground: this far along it.
const FALLBACK_DISTANCE: f32 = 5.0;
const PLAYER_START_COLOR: Color32 = Color32::from_rgb(80, 220, 120);

/// A target's transform relative to its parent (identity for lights).
#[derive(Clone, Copy)]
struct Pose {
    translation: [f32; 3],
    rotation: Quat,
    scale: [f32; 3],
}

/// State of an in-progress drag, kept in egui temp memory between frames.
#[derive(Clone)]
struct Drag {
    target: Target,
    mode: GizmoMode,
    handle: Handle,
    axis: Vec3,
    start_pose: Pose,
    start_center: Vec3,
    parent_world: Mat4,
    start_pointer: Pos2,
    start_param: f32,
    last_angle: f32,
    accumulated_angle: f32,
}

struct View {
    view_proj: Mat4,
    inv_view_proj: Mat4,
    eye: Vec3,
    fovy: f32,
    screen: Rect,
}

impl View {
    fn new(camera: &Camera, screen: Rect) -> Self {
        let view_proj = camera.view_proj_matrix(screen.width() / screen.height());
        Self {
            inv_view_proj: view_proj.inversed(),
            view_proj,
            eye: camera.eye,
            fovy: camera.fovy,
            screen,
        }
    }

    fn project(&self, p: Vec3) -> Option<Pos2> {
        let clip = self.view_proj * Vec4::new(p.x, p.y, p.z, 1.0);
        if clip.w <= 1e-5 {
            return None;
        }
        let (x, y) = (clip.x / clip.w, clip.y / clip.w);
        Some(Pos2::new(
            self.screen.left() + (x * 0.5 + 0.5) * self.screen.width(),
            self.screen.top() + (0.5 - y * 0.5) * self.screen.height(),
        ))
    }

    /// World-space ray through a screen point. Depth is reversed-Z, so NDC
    /// z = 1 is the near plane.
    fn ray(&self, pos: Pos2) -> (Vec3, Vec3) {
        let x = (pos.x - self.screen.left()) / self.screen.width() * 2.0 - 1.0;
        let y = 1.0 - (pos.y - self.screen.top()) / self.screen.height() * 2.0;
        let unproject = |z: f32| {
            let v = self.inv_view_proj * Vec4::new(x, y, z, 1.0);
            Vec3::new(v.x, v.y, v.z) / v.w
        };
        let near = unproject(1.0);
        let far = unproject(0.5);
        (near, (far - near).normalized())
    }

    /// World length that covers `px` screen pixels at `at`.
    fn world_size(&self, at: Vec3, px: f32) -> f32 {
        let dist = (at - self.eye).mag();
        px * dist * 2.0 * (self.fovy * 0.5).tan() / self.screen.height()
    }
}

pub fn draw(ctx: &egui::Context, state: &mut EditorState, world: &World, viewport: Rect) {
    if state.document.is_none() || state.camera_preview_node.is_some() {
        return;
    }
    let Some(camera) = state
        .editor_camera_key
        .and_then(|key| world.get_entity::<Camera>(key))
    else {
        return;
    };
    let view = View::new(camera, ctx.content_rect());

    let target = selected_target(state);
    if let Some(target) = target
        && editable_axes(state, target, state.gizmo_mode).is_none()
        && let Some(&mode) = MODES.iter().find(|&&m| editable_axes(state, target, m).is_some())
    {
        state.gizmo_mode = mode;
    }
    draw_mode_toolbar(ctx, state, target, viewport);

    let drag_id = Id::unique("gizmo_drag");
    let pressed_id = Id::unique("gizmo_pressed_handle");
    let mut drag: Option<Drag> = ctx.data(|d| d.get_temp(drag_id));

    if drag.as_ref().is_some_and(|d| Some(d.target) != target || d.mode != state.gizmo_mode) {
        drag = None;
    }

    let pointer = ctx.input(|i| i.pointer.hover_pos());
    let gizmo_layer = LayerId::new(Order::Middle, Id::unique("gizmo_input"));
    let pointer_free = pointer.is_some_and(|p| {
        viewport.contains(p) && ctx.layer_id_at(p).is_none_or(|layer| layer == gizmo_layer)
    });

    let painter = ctx
        .layer_painter(LayerId::new(Order::Background, Id::unique("gizmo_paint")))
        .with_clip_rect(viewport);
    let mut markers = paint_light_markers(&painter, &view, state);
    markers.extend(paint_player_starts(&painter, &view, state));

    let mut hovered = None;
    if let Some(target) = target
        && let Some(mask) = editable_axes(state, target, state.gizmo_mode)
        && let Some(world_matrix) = target_world(state, target)
    {
        let center = world_matrix.extract_translation();
        let len = view.world_size(center, GIZMO_PX);
        let axes = gizmo_axes(state.gizmo_mode, &world_matrix, mask);

        if drag.is_none() && pointer_free {
            hovered = pointer.and_then(|p| hit_test(&view, state.gizmo_mode, center, axes, len, p));
        }

        let active = drag.as_ref().map(|d| d.handle).or(hovered);
        paint(&painter, &view, state.gizmo_mode, center, axes, len, active);

        let response = egui::Area::new(gizmo_layer.id)
            .order(Order::Middle)
            .fixed_pos(viewport.min)
            .interactable(hovered.is_some() || drag.is_some())
            .show(ctx, |ui| ui.allocate_exact_size(viewport.size(), Sense::drag()).1)
            .inner;

        if ctx.input(|i| i.pointer.primary_pressed()) {
            ctx.data_mut(|d| match hovered {
                Some(handle) => {
                    d.insert_temp(pressed_id, handle);
                }
                None => d.remove::<Handle>(pressed_id),
            });
        }

        if response.drag_started()
            && let Some(handle) = ctx.data(|d| d.get_temp::<Handle>(pressed_id))
        {
            let start_pointer = ctx.input(|i| i.pointer.press_origin()).or(pointer);
            if let Some(start_pointer) = start_pointer {
                if let Target::Node(node) = target {
                    state.ensure_node_edit(node);
                }
                drag = begin_drag(state, &view, target, handle, axes, len, start_pointer);
            }
        }

        if let (Some(d), Some(p)) = (drag.as_mut(), pointer)
            && response.dragged()
        {
            update_drag(state, &view, d, p);
        }

        if response.drag_stopped() {
            drag = None;
        }
    }

    ctx.data_mut(|d| match &drag {
        Some(drag) => {
            d.insert_temp(drag_id, drag.clone());
        }
        None => d.remove::<Drag>(drag_id),
    });

    if drag.is_none()
        && hovered.is_none()
        && pointer_free
        && !ctx.is_pointer_over_egui()
        && ctx.input(|i| i.pointer.primary_clicked())
        && let Some(p) = pointer
    {
        state.selected = pick_marker(&markers, p).or_else(|| raycast(world, state, view.ray(p)).and_then(|hit| hit.owner));
        if let Some(Selection::GltfNode(node)) = state.selected {
            state.ensure_node_edit(node);
        }
    }

    draw_add_menu(ctx, state, world, &view, pointer, pointer_free && drag.is_none() && hovered.is_none());
}

/// Right-click (without dragging, which orbits) in the viewport opens an Add menu that places
/// the new light or object where the click ray meets the scene.
fn draw_add_menu(ctx: &egui::Context, state: &mut EditorState, world: &World, view: &View, pointer: Option<Pos2>, pointer_free: bool) {
    let menu_id = Id::unique("viewport_add_menu");
    if pointer_free
        && !ctx.is_pointer_over_egui()
        && ctx.input(|i| i.pointer.secondary_clicked())
        && let Some(p) = pointer
    {
        let menu = AddMenu { pos: p, placement: placement(world, state, view, p) };
        ctx.data_mut(|d| d.insert_temp(menu_id, menu));
    }
    let Some(menu) = ctx.data(|d| d.get_temp::<AddMenu>(menu_id)) else { return };

    let mut picked = None;
    egui::Area::new(menu_id).order(Order::Foreground).fixed_pos(menu.pos).show(ctx, |ui| {
        egui::Frame::menu(ui.style()).show(ui, |ui| {
            ui.label(egui::RichText::new("Add").strong());
            ui.separator();
            picked = super::add_menu::items(ui, state);
        });
    });

    // Menus and their submenus live on foreground layers; a press anywhere else closes it.
    let pressed_elsewhere = ctx.input(|i| i.pointer.any_pressed())
        && pointer.is_some_and(|p| ctx.layer_id_at(p).is_none_or(|layer| layer.order != Order::Foreground));
    let chose = picked.is_some();
    if let Some(kind) = picked {
        state.add(kind, menu.placement);
    }
    if chose || pressed_elsewhere || ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
        ctx.data_mut(|d| d.remove::<AddMenu>(menu_id));
    }
}

/// Adds `kind` where the ray through `at` (the viewport centre if `None`) meets the scene.
pub fn place(ctx: &egui::Context, state: &mut EditorState, world: &World, kind: AddKind, at: Option<Pos2>) {
    let Some(camera) = state.editor_camera_key.and_then(|key| world.get_entity::<Camera>(key)) else { return };
    let view = View::new(camera, ctx.content_rect());
    let placement = placement(world, state, &view, at.unwrap_or(view.screen.center()));
    state.add(kind, placement);
}

/// Where a new object goes for a click at `pointer`: the nearest surface hit, else the ground
/// plane, else a few meters along the ray.
fn placement(world: &World, state: &EditorState, view: &View, pointer: Pos2) -> Placement {
    let (origin, dir) = view.ray(pointer);
    if let Some(hit) = raycast(world, state, (origin, dir)) {
        return Placement { point: hit.point, normal: hit.normal, view_dir: dir };
    }
    let t = if dir.y < -1e-4 { -origin.y / dir.y } else { FALLBACK_DISTANCE };
    Placement { point: origin + dir * t.min(1000.0), normal: Vec3::unit_y(), view_dir: dir }
}

fn selected_target(state: &EditorState) -> Option<Target> {
    match state.selected {
        Some(Selection::GltfNode(idx)) => Some(Target::Node(idx)),
        Some(Selection::Light(idx)) if idx < state.lights.len() => Some(Target::Light(idx)),
        Some(Selection::Object(id)) if state.objects.iter().any(|o| o.id == id) => Some(Target::Object(id)),
        _ => None,
    }
}

fn object(state: &EditorState, id: u64) -> Option<&crate::state::ObjectData> {
    state.objects.iter().find(|o| o.id == id).map(|o| &o.data)
}

/// Axes the gizmo offers in `mode`, or `None` when the target can't be edited that way.
fn editable_axes(state: &EditorState, target: Target, mode: GizmoMode) -> Option<[bool; 3]> {
    let idx = match target {
        Target::Node(_) => return Some(ALL_AXES),
        Target::Object(id) => {
            return match (&object(state, id)?.kind, mode) {
                (ObjectKind::PlayerStart, GizmoMode::Scale) => None,
                _ => Some(ALL_AXES),
            };
        }
        Target::Light(idx) => idx,
    };
    match (&state.lights.get(idx)?.data, mode) {
        (LightData::Point { .. }, GizmoMode::Translate)
        | (LightData::Spot { .. }, GizmoMode::Translate | GizmoMode::Rotate)
        | (LightData::Sky { .. }, GizmoMode::Rotate)
        | (LightData::Area { .. }, GizmoMode::Translate | GizmoMode::Rotate) => Some(ALL_AXES),
        (LightData::Area { .. }, GizmoMode::Scale) => Some([true, true, false]),
        _ => None,
    }
}

fn read_pose(state: &EditorState, target: Target) -> Option<Pose> {
    match target {
        Target::Node(idx) => state.node_edits.get(&idx).map(|edit| Pose {
            translation: edit.translation,
            rotation: edit.rotation,
            scale: edit.scale,
        }),
        Target::Light(idx) => Some(light_pose(&state.lights.get(idx)?.data)),
        Target::Object(id) => object(state, id).map(|o| Pose { translation: o.translation, rotation: o.rotation, scale: o.scale }),
    }
}

fn write_pose(state: &mut EditorState, target: Target, pose: Pose) {
    match target {
        Target::Node(idx) => {
            if let Some(edit) = state.node_edits.get_mut(&idx) {
                edit.translation = pose.translation;
                edit.rotation = pose.rotation;
                edit.scale = pose.scale;
            }
        }
        Target::Light(idx) => {
            if let Some(light) = state.lights.get_mut(idx) {
                apply_light_pose(&mut light.data, pose);
            }
        }
        Target::Object(id) => {
            if let Some(object) = state.objects.iter_mut().find(|o| o.id == id) {
                object.data.translation = pose.translation;
                object.data.rotation = pose.rotation;
                object.data.scale = pose.scale;
            }
        }
    }
}

fn target_world(state: &EditorState, target: Target) -> Option<Mat4> {
    match target {
        Target::Node(idx) => Some(state.node_world(idx)),
        Target::Light(_) | Target::Object(_) => {
            let pose = read_pose(state, target)?;
            Some(math::trs(pose.translation, pose.rotation, pose.scale))
        }
    }
}

fn target_parent_world(state: &EditorState, target: Target) -> Mat4 {
    match target {
        Target::Node(idx) => state.parent_world(idx),
        Target::Light(_) | Target::Object(_) => Mat4::identity(),
    }
}

/// Lights aim down their local -Z. An area light's rectangle spans local X (`right`) and Y
/// (`up`), with the half-extents as scale. Sky lights have no position, so they sit at the
/// scene origin.
fn light_pose(data: &LightData) -> Pose {
    let identity = [0.0, 0.0, 0.0, 1.0];
    match *data {
        LightData::Point { position, .. } => Pose { translation: position, rotation: identity, scale: [1.0; 3] },
        LightData::Spot { position, direction, .. } => {
            Pose { translation: position, rotation: math::quat_look(direction.into()), scale: [1.0; 3] }
        }
        LightData::Sky { direction, .. } => {
            Pose { translation: [0.0; 3], rotation: math::quat_look(direction.into()), scale: [1.0; 3] }
        }
        LightData::Area { position, right, up, .. } => {
            let (right, up) = (Vec3::from(right), Vec3::from(up));
            let normal = right.cross(up);
            let basis = Mat4::new(
                Vec4::new(right.x, right.y, right.z, 0.0),
                Vec4::new(up.x, up.y, up.z, 0.0),
                Vec4::new(normal.x, normal.y, normal.z, 0.0),
                Vec4::new(0.0, 0.0, 0.0, 1.0),
            );
            Pose {
                translation: position,
                rotation: math::quat_from_matrix(&basis),
                scale: [right.mag(), up.mag(), 1.0],
            }
        }
    }
}

fn apply_light_pose(data: &mut LightData, pose: Pose) {
    let forward = math::rotate(pose.rotation, -Vec3::unit_z()).into();
    match data {
        LightData::Point { position, .. } => *position = pose.translation,
        LightData::Spot { position, direction, .. } => {
            *position = pose.translation;
            *direction = forward;
        }
        LightData::Sky { direction, .. } => *direction = forward,
        LightData::Area { position, right, up, .. } => {
            *position = pose.translation;
            *right = (math::rotate(pose.rotation, Vec3::unit_x()) * pose.scale[0]).into();
            *up = (math::rotate(pose.rotation, Vec3::unit_y()) * pose.scale[1]).into();
        }
    }
}

/// Draws each player start (a post to eye height with an arrow along its heading) and returns
/// their screen positions for picking.
fn paint_player_starts(painter: &egui::Painter, view: &View, state: &EditorState) -> Vec<(Selection, Pos2)> {
    let mut markers = Vec::new();
    let mut marks = Marks::default();
    for object in &state.objects {
        if !matches!(object.data.kind, ObjectKind::PlayerStart) {
            continue;
        }
        let selected = state.selected == Some(Selection::Object(object.id));
        if !state.show_gizmos && !selected {
            continue;
        }
        let outline = if selected { ACTIVE_COLOR } else { Color32::from_gray(230) };
        let feet = Vec3::from(object.data.translation);
        let eye = feet + Vec3::unit_y() * PLAYER_EYE_HEIGHT;
        let heading = math::rotate(object.data.rotation, -Vec3::unit_z());
        let ahead = eye + Vec3::new(heading.x, 0.0, heading.z).normalized() * view.world_size(eye, SPOT_LINE_PX);
        let (Some(feet_px), Some(eye_px), Some(ahead_px)) = (view.project(feet), view.project(eye), view.project(ahead)) else {
            continue;
        };
        marks.0.push(Mark::Line(vec![feet_px, eye_px], outline));
        marks.0.push(Mark::Line(vec![eye_px, ahead_px], outline));
        marks.0.push(Mark::Disc(eye_px, MARKER_RADIUS_PX, PLAYER_START_COLOR, outline));
        markers.push((Selection::Object(object.id), eye_px));
    }
    marks.paint(painter);
    markers
}

/// Draws a marker for every positioned light and returns their screen positions for picking.
fn paint_light_markers(painter: &egui::Painter, view: &View, state: &EditorState) -> Vec<(Selection, Pos2)> {
    let selected = match state.selected {
        Some(Selection::Light(idx)) => Some(idx),
        _ => None,
    };
    let mut markers = Vec::new();
    let mut marks = Marks::default();
    for (idx, light) in state.lights.iter().enumerate() {
        if !state.show_gizmos && selected != Some(idx) {
            continue;
        }
        let stroke_color = if selected == Some(idx) { ACTIVE_COLOR } else { Color32::from_gray(230) };
        let (position, color) = match light.data {
            LightData::Sky { direction, color, .. } => {
                if let Some(sun) = sun_marks(&mut marks, view, direction.into(), color32(color), stroke_color) {
                    markers.push((Selection::Light(idx), sun));
                }
                continue;
            }
            LightData::Point { position, color, .. }
            | LightData::Spot { position, color, .. }
            | LightData::Area { position, color, .. } => (Vec3::from(position), color),
        };
        let Some(screen) = view.project(position) else { continue };
        let fill = color32(color);

        match light.data {
            LightData::Spot { direction, .. } => {
                let tip = position + Vec3::from(direction).normalized() * view.world_size(position, SPOT_LINE_PX);
                if let Some(tip) = view.project(tip) {
                    marks.0.push(Mark::Line(vec![screen, tip], stroke_color));
                }
            }
            LightData::Area { right, up, .. } => {
                let (right, up) = (Vec3::from(right), Vec3::from(up));
                let corners: Option<Vec<Pos2>> = [right + up, right - up, -right - up, -right + up]
                    .into_iter()
                    .map(|offset| view.project(position + offset))
                    .collect();
                if let Some(corners) = corners {
                    marks.0.push(Mark::Loop(corners, stroke_color));
                }
            }
            _ => {}
        }
        marks.0.push(Mark::Disc(screen, MARKER_RADIUS_PX, fill, stroke_color));
        markers.push((Selection::Light(idx), screen));
    }
    marks.paint(painter);
    markers
}

enum Mark {
    Line(Vec<Pos2>, Color32),
    Loop(Vec<Pos2>, Color32),
    Disc(Pos2, f32, Color32, Color32),
    Triangle([Pos2; 3], Color32),
}

/// Marker shapes, painted as all halos first and then all colors, so one marker's halo
/// never covers another's lines.
#[derive(Default)]
struct Marks(Vec<Mark>);

impl Marks {
    fn paint(self, painter: &egui::Painter) {
        let halo = Stroke::new(MARK_STROKE_PX + HALO_PX * 2.0, HALO);
        for mark in &self.0 {
            painter.add(match mark {
                Mark::Line(points, _) => Shape::line(points.clone(), halo),
                Mark::Loop(points, _) => Shape::closed_line(points.clone(), halo),
                Mark::Disc(center, radius, _, _) => Shape::circle_filled(*center, radius + MARK_STROKE_PX / 2.0 + HALO_PX, HALO),
                Mark::Triangle(points, _) => Shape::convex_polygon(points.to_vec(), HALO, Stroke::new(HALO_PX * 2.0, HALO)),
            });
        }
        for mark in self.0 {
            painter.add(match mark {
                Mark::Line(points, color) => Shape::line(points, Stroke::new(MARK_STROKE_PX, color)),
                Mark::Loop(points, color) => Shape::closed_line(points, Stroke::new(MARK_STROKE_PX, color)),
                Mark::Disc(center, radius, fill, outline) => {
                    egui::epaint::CircleShape { center, radius, fill, stroke: Stroke::new(MARK_STROKE_PX, outline) }.into()
                }
                Mark::Triangle(points, color) => Shape::convex_polygon(points.to_vec(), color, Stroke::NONE),
            });
        }
    }
}

fn color32(color: [f32; 3]) -> Color32 {
    let [r, g, b] = color.map(|c| (c.clamp(0.0, 1.0) * 255.0) as u8);
    Color32::from_rgb(r, g, b)
}

/// Sun on the side the sky light comes from, with an arrow along the light into the scene
/// origin (where the sky gizmo sits). Greyed out below the horizon, where `lighting.wgsl`
/// ignores the light. Returns the sun's screen position for picking.
fn sun_marks(marks: &mut Marks, view: &View, direction: Vec3, fill: Color32, outline: Color32) -> Option<Pos2> {
    let direction = direction.normalized();
    let sun = view.project(-direction * view.world_size(Vec3::zero(), SUN_DISTANCE_PX))?;
    let origin = view.project(Vec3::zero())?;
    let (fill, outline) = if direction.y > 0.0 { (SUN_BELOW_HORIZON, SUN_BELOW_HORIZON) } else { (fill, outline) };

    let to_origin = origin - sun;
    if to_origin.length() > SUN_RADIUS_PX * 3.0 {
        let dir = to_origin.normalized();
        let tip = origin - dir * 4.0;
        let (back, perp) = (tip - dir * 10.0, dir.rot90() * 5.0);
        marks.0.push(Mark::Line(vec![sun + dir * (SUN_RADIUS_PX + 8.0), back], outline));
        marks.0.push(Mark::Triangle([tip, back + perp, back - perp], outline));
    }

    for i in 0..8 {
        let ray = egui::Vec2::angled(i as f32 * std::f32::consts::TAU / 8.0);
        marks.0.push(Mark::Line(vec![sun + ray * (SUN_RADIUS_PX + 3.0), sun + ray * (SUN_RADIUS_PX + 7.0)], outline));
    }
    marks.0.push(Mark::Disc(sun, SUN_RADIUS_PX, fill, outline));
    Some(sun)
}

fn pick_marker(markers: &[(Selection, Pos2)], pointer: Pos2) -> Option<Selection> {
    markers
        .iter()
        .map(|(target, pos)| (target, pos.distance(pointer)))
        .filter(|&(_, dist)| dist <= MARKER_RADIUS_PX + PICK_PX * 0.5)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(target, _)| target.clone())
}

fn draw_mode_toolbar(ctx: &egui::Context, state: &mut EditorState, target: Option<Target>, viewport: Rect) {
    let supported: [bool; 3] =
        MODES.map(|mode| target.is_none_or(|t| editable_axes(state, t, mode).is_some()));

    if !ctx.egui_wants_keyboard_input() {
        ctx.input(|i| {
            for (key, (mode, ok)) in [egui::Key::Num1, egui::Key::Num2, egui::Key::Num3]
                .into_iter()
                .zip(MODES.into_iter().zip(supported))
            {
                if ok && i.key_pressed(key) {
                    state.gizmo_mode = mode;
                }
            }
        });
    }

    egui::Area::new(Id::unique("gizmo_toolbar"))
        .order(Order::Middle)
        .fixed_pos(viewport.min + egui::vec2(8.0, 8.0))
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(Color32::from_black_alpha(160))
                .corner_radius(4.0)
                .inner_margin(egui::Margin::symmetric(6, 4))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        for ((mode, label), ok) in MODES.into_iter().zip(["Move (1)", "Rotate (2)", "Scale (3)"]).zip(supported) {
                            ui.add_enabled_ui(ok, |ui| ui.selectable_value(&mut state.gizmo_mode, mode, label));
                        }
                    });
                });
        });
}

/// World axes for translate/rotate, the target's own (normalized) axes for scale.
/// Axes the target can't be edited along are `None`.
fn gizmo_axes(mode: GizmoMode, world_matrix: &Mat4, mask: [bool; 3]) -> [Option<Vec3>; 3] {
    let world = [Vec3::unit_x(), Vec3::unit_y(), Vec3::unit_z()];
    let axes = match mode {
        GizmoMode::Translate | GizmoMode::Rotate => world,
        GizmoMode::Scale => {
            let cols = world_matrix.truncate().cols;
            std::array::from_fn(|i| {
                let len = cols[i].mag();
                if len > 1e-6 { cols[i] / len } else { world[i] }
            })
        }
    };
    std::array::from_fn(|i| mask[i].then_some(axes[i]))
}

fn ring_points(view: &View, center: Vec3, axis: Vec3, radius: f32) -> Option<Vec<Pos2>> {
    let (u, w) = plane_basis(axis);
    (0..=RING_SEGMENTS)
        .map(|i| {
            let t = i as f32 / RING_SEGMENTS as f32 * std::f32::consts::TAU;
            view.project(center + (u * t.cos() + w * t.sin()) * radius)
        })
        .collect()
}

fn hit_test(view: &View, mode: GizmoMode, center: Vec3, axes: [Option<Vec3>; 3], len: f32, pointer: Pos2) -> Option<Handle> {
    let origin = view.project(center)?;

    if mode == GizmoMode::Scale && origin.distance(pointer) < PICK_PX * 1.5 {
        return Some(Handle::Uniform);
    }

    let mut best: Option<(f32, Handle)> = None;
    for (i, axis) in axes.iter().enumerate() {
        let Some(axis) = *axis else { continue };
        let dist = match mode {
            GizmoMode::Translate | GizmoMode::Scale => {
                let Some(tip) = view.project(center + axis * len) else { continue };
                if origin.distance(tip) < MIN_AXIS_PX {
                    continue;
                }
                segment_distance(pointer, origin, tip)
            }
            GizmoMode::Rotate => {
                let Some(points) = ring_points(view, center, axis, len) else { continue };
                points
                    .windows(2)
                    .map(|s| segment_distance(pointer, s[0], s[1]))
                    .fold(f32::MAX, f32::min)
            }
        };
        if dist < PICK_PX && best.is_none_or(|(d, _)| dist < d) {
            best = Some((dist, Handle::Axis(i)));
        }
    }
    best.map(|(_, handle)| handle)
}

fn paint(
    painter: &egui::Painter,
    view: &View,
    mode: GizmoMode,
    center: Vec3,
    axes: [Option<Vec3>; 3],
    len: f32,
    active: Option<Handle>,
) {
    let Some(origin) = view.project(center) else { return };
    let color_for = |handle: Handle, base: Color32| if active == Some(handle) { ACTIVE_COLOR } else { base };

    for (i, axis) in axes.iter().enumerate() {
        let Some(axis) = *axis else { continue };
        let color = color_for(Handle::Axis(i), AXIS_COLORS[i]);
        let stroke = Stroke::new(3.0, color);
        match mode {
            GizmoMode::Translate | GizmoMode::Scale => {
                let Some(tip) = view.project(center + axis * len) else { continue };
                if origin.distance(tip) < MIN_AXIS_PX {
                    continue;
                }
                painter.line_segment([origin, tip], stroke);
                let dir = (tip - origin).normalized();
                if mode == GizmoMode::Translate {
                    let perp = dir.rot90() * 6.0;
                    painter.add(Shape::convex_polygon(
                        vec![tip + dir * 14.0, tip + perp, tip - perp],
                        color,
                        Stroke::NONE,
                    ));
                } else {
                    painter.rect_filled(Rect::from_center_size(tip, egui::vec2(10.0, 10.0)), 1.0, color);
                }
            }
            GizmoMode::Rotate => {
                if let Some(points) = ring_points(view, center, axis, len) {
                    painter.add(Shape::line(points, stroke));
                }
            }
        }
    }

    if mode == GizmoMode::Scale {
        let color = color_for(Handle::Uniform, Color32::from_gray(230));
        painter.rect_filled(Rect::from_center_size(origin, egui::vec2(12.0, 12.0)), 1.0, color);
    } else {
        painter.circle_filled(origin, 3.5, Color32::from_gray(230));
    }
}

fn begin_drag(
    state: &EditorState,
    view: &View,
    target: Target,
    handle: Handle,
    axes: [Option<Vec3>; 3],
    len: f32,
    start_pointer: Pos2,
) -> Option<Drag> {
    let start_pose = read_pose(state, target)?;
    let parent_world = target_parent_world(state, target);
    let start_center = target_world(state, target)?.extract_translation();
    let axis = match handle {
        Handle::Axis(i) => axes[i]?,
        Handle::Uniform => Vec3::zero(),
    };
    let ray = view.ray(start_pointer);

    let start_param = match (state.gizmo_mode, handle) {
        (GizmoMode::Rotate, _) => ring_angle(start_center, axis, ray)?,
        (_, Handle::Axis(_)) => axis_param(start_center, axis, ray)?,
        (_, Handle::Uniform) => 0.0,
    };
    if state.gizmo_mode == GizmoMode::Scale && handle != Handle::Uniform && start_param.abs() < len * 0.05 {
        return None;
    }

    Some(Drag {
        target,
        mode: state.gizmo_mode,
        handle,
        axis,
        start_pose,
        start_center,
        parent_world,
        start_pointer,
        start_param,
        last_angle: start_param,
        accumulated_angle: 0.0,
    })
}

fn update_drag(state: &mut EditorState, view: &View, drag: &mut Drag, pointer: Pos2) {
    let ray = view.ray(pointer);
    let mut pose = drag.start_pose;

    match (drag.mode, drag.handle) {
        (GizmoMode::Translate, Handle::Axis(_)) => {
            let Some(t) = axis_param(drag.start_center, drag.axis, ray) else { return };
            let world_pos = drag.start_center + drag.axis * (t - drag.start_param);
            pose.translation = drag.parent_world.inversed().transform_point3(world_pos).into();
        }
        (GizmoMode::Rotate, Handle::Axis(_)) => {
            let Some(angle) = ring_angle(drag.start_center, drag.axis, ray) else { return };
            let mut delta = angle - drag.last_angle;
            if delta > std::f32::consts::PI {
                delta -= std::f32::consts::TAU;
            } else if delta < -std::f32::consts::PI {
                delta += std::f32::consts::TAU;
            }
            drag.accumulated_angle += delta;
            drag.last_angle = angle;

            // World-space rotation about the axis, expressed in the parent's frame.
            let parent_rotation = math::quat_from_matrix(&drag.parent_world);
            let world_delta = math::quat_from_axis_angle(drag.axis, drag.accumulated_angle);
            let local_delta = math::quat_mul(
                math::quat_conjugate(parent_rotation),
                math::quat_mul(world_delta, parent_rotation),
            );
            pose.rotation = math::quat_normalize(math::quat_mul(local_delta, drag.start_pose.rotation));
        }
        (GizmoMode::Scale, Handle::Axis(i)) => {
            let Some(t) = axis_param(drag.start_center, drag.axis, ray) else { return };
            let factor = t / drag.start_param;
            pose.scale[i] = (drag.start_pose.scale[i] * factor).max(MIN_SCALE);
        }
        (GizmoMode::Scale, Handle::Uniform) => {
            let factor = ((pointer.x - drag.start_pointer.x) * UNIFORM_SCALE_PER_PX).exp();
            pose.scale = drag.start_pose.scale.map(|s| (s * factor).max(MIN_SCALE));
        }
        _ => {}
    }

    write_pose(state, drag.target, pose);
}

/// Parameter `t` of the point on the line `origin + axis * t` closest to the ray.
fn axis_param(origin: Vec3, axis: Vec3, (ray_origin, ray_dir): (Vec3, Vec3)) -> Option<f32> {
    let w0 = origin - ray_origin;
    let a = axis.dot(axis);
    let b = axis.dot(ray_dir);
    let c = ray_dir.dot(ray_dir);
    let d = axis.dot(w0);
    let e = ray_dir.dot(w0);
    let denom = a * c - b * b;
    if denom.abs() < 1e-6 {
        return None;
    }
    Some((b * e - c * d) / denom)
}

/// Two unit vectors spanning the plane perpendicular to `axis`, with `u × w = axis`.
fn plane_basis(axis: Vec3) -> (Vec3, Vec3) {
    let helper = if axis.x.abs() < 0.9 { Vec3::unit_x() } else { Vec3::unit_y() };
    let u = axis.cross(helper).normalized();
    (u, axis.cross(u))
}

/// Angle (right-handed about `axis`) where the ray crosses the plane through
/// `center` perpendicular to `axis`.
fn ring_angle(center: Vec3, axis: Vec3, (ray_origin, ray_dir): (Vec3, Vec3)) -> Option<f32> {
    let denom = ray_dir.dot(axis);
    if denom.abs() < 1e-4 {
        return None;
    }
    let s = (center - ray_origin).dot(axis) / denom;
    if s < 0.0 {
        return None;
    }
    let offset = ray_origin + ray_dir * s - center;
    let (u, w) = plane_basis(axis);
    Some(offset.dot(w).atan2(offset.dot(u)))
}

fn segment_distance(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_sq().max(1e-6)).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

/// The nearest triangle a pick ray hits, among meshes of visible scene nodes and objects.
struct Hit {
    t: f32,
    point: Vec3,
    /// Facing back along the ray.
    normal: Vec3,
    owner: Option<Selection>,
}

/// Nearest triangle along `ray` (world space; `dir` normalized). Each mesh is tested in its own
/// space after a bounds check, so only meshes the ray reaches cost a triangle walk.
fn raycast(world: &World, state: &EditorState, (origin, dir): (Vec3, Vec3)) -> Option<Hit> {
    let mut best: Option<Hit> = None;
    for handle in world.get_entity_handles::<Mesh>() {
        let owner = match state.entity_nodes.get(&handle.id) {
            Some(&node) if state.is_node_deleted(node) => continue,
            Some(&node) => Selection::GltfNode(node),
            None => match state.entity_objects.get(&handle.id) {
                Some(&id) => Selection::Object(id),
                None => continue,
            },
        };
        let mesh = handle.entity;
        if mesh.vertices.is_empty() || mesh.transform.determinant().abs() < 1e-12 {
            continue;
        }
        let to_local = mesh.transform.inversed();
        // Same ray parameter in both spaces: t along `dir` in world is t along `local_dir` here.
        let (local_origin, local_dir) = (to_local.transform_point3(origin), to_local.transform_vec3(dir));

        let (mut lo, mut hi) = (Vec3::broadcast(f32::MAX), Vec3::broadcast(f32::MIN));
        for v in &mesh.vertices {
            let p = Vec3::from(v.position);
            lo = lo.min_by_component(p);
            hi = hi.max_by_component(p);
        }
        match ray_aabb((local_origin, local_dir), lo, hi) {
            Some(t) if best.as_ref().is_none_or(|b| t < b.t) => {}
            _ => continue,
        }

        for tri in mesh.indices.as_chunks::<3>().0 {
            let [a, b, c] = tri.map(|i| Vec3::from(mesh.vertices[i as usize].position));
            let Some(t) = ray_triangle((local_origin, local_dir), a, b, c) else { continue };
            if best.as_ref().is_some_and(|b| t >= b.t) {
                continue;
            }
            let normal = mesh.transform.transform_vec3(b - a).cross(mesh.transform.transform_vec3(c - a)).normalized();
            let normal = if normal.dot(dir) > 0.0 { -normal } else { normal };
            best = Some(Hit { t, point: origin + dir * t, normal, owner: Some(owner.clone()) });
        }
    }
    best
}

/// Möller–Trumbore, both faces. `t` in units of `dir`.
fn ray_triangle((origin, dir): (Vec3, Vec3), a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
    let (e1, e2) = (b - a, c - a);
    let p = dir.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    let s = origin - a;
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = dir.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    (t > 1e-5).then_some(t)
}

fn ray_aabb((origin, dir): (Vec3, Vec3), lo: Vec3, hi: Vec3) -> Option<f32> {
    let mut t_min = 0.0f32;
    let mut t_max = f32::MAX;
    for axis in 0..3 {
        let (o, d, l, h) = (origin[axis], dir[axis], lo[axis], hi[axis]);
        if d.abs() < 1e-8 {
            if o < l || o > h {
                return None;
            }
            continue;
        }
        let (t0, t1) = ((l - o) / d, (h - o) / d);
        t_min = t_min.max(t0.min(t1));
        t_max = t_max.min(t0.max(t1));
        if t_min > t_max {
            return None;
        }
    }
    Some(t_min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::LightDataExt;

    #[test]
    fn axis_param_finds_closest_point() {
        let ray = (Vec3::new(3.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let t = axis_param(Vec3::zero(), Vec3::unit_x(), ray).unwrap();
        assert!((t - 3.0).abs() < 1e-4);
    }

    #[test]
    fn axis_param_rejects_parallel_ray() {
        let ray = (Vec3::new(0.0, 1.0, 0.0), Vec3::unit_x());
        assert!(axis_param(Vec3::zero(), Vec3::unit_x(), ray).is_none());
    }

    #[test]
    fn ring_angle_is_right_handed() {
        let (u, w) = plane_basis(Vec3::unit_y());
        assert!((u.cross(w) - Vec3::unit_y()).mag() < 1e-5);

        let hit_at = |p: Vec3| ring_angle(Vec3::zero(), Vec3::unit_y(), (p + Vec3::unit_y() * 5.0, -Vec3::unit_y())).unwrap();
        let a0 = hit_at(u);
        let a1 = hit_at(w);
        assert!((a1 - a0 - std::f32::consts::FRAC_PI_2).abs() < 1e-4);
    }

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        (Vec3::from(a) - Vec3::from(b)).mag() < 1e-4
    }

    #[test]
    fn light_pose_round_trips() {
        let spot = LightData::Spot {
            position: [1.0, 2.0, 3.0],
            direction: [0.0, -0.6, 0.8],
            color: [1.0; 3],
            intensity: 1.0,
            radius: 1.0,
            source_radius: 0.0,
            inner_angle: 0.1,
            outer_angle: 0.2,
        };
        let area = LightData::Area {
            position: [0.0, 1.0, 0.0],
            right: [0.0, 0.0, 2.0],
            up: [0.0, 0.5, 0.0],
            color: [1.0; 3],
            intensity: 1.0,
        };
        for original in [spot, area] {
            let mut data = original.clone();
            apply_light_pose(&mut data, light_pose(&original));
            match (original, data) {
                (LightData::Spot { position: p0, direction: d0, .. }, LightData::Spot { position: p1, direction: d1, .. }) => {
                    assert!(close(p0, p1) && close(d0, d1));
                }
                (LightData::Area { right: r0, up: u0, .. }, LightData::Area { right: r1, up: u1, .. }) => {
                    assert!(close(r0, r1) && close(u0, u1));
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn rotating_a_spot_turns_its_direction() {
        let mut data = LightData::default_for(crate::state::LightKind::Spot);
        let mut pose = light_pose(&data);
        let quarter_turn = math::quat_from_axis_angle(Vec3::unit_x(), std::f32::consts::FRAC_PI_2);
        pose.rotation = math::quat_mul(quarter_turn, pose.rotation);
        apply_light_pose(&mut data, pose);
        let LightData::Spot { direction, .. } = data else { unreachable!() };
        assert!(close(direction, [0.0, 0.0, -1.0]));
    }

    #[test]
    fn ray_hits_box_in_front_only() {
        let lo = Vec3::broadcast(-1.0);
        let hi = Vec3::broadcast(1.0);
        let t = ray_aabb((Vec3::new(0.0, 0.0, 5.0), -Vec3::unit_z()), lo, hi).unwrap();
        assert!((t - 4.0).abs() < 1e-4);
        assert!(ray_aabb((Vec3::new(0.0, 0.0, 5.0), Vec3::unit_z()), lo, hi).is_none());
    }
}
