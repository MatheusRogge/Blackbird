//! Viewport transform gizmo for the selected glTF node, plus click-to-select.
//!
//! Translate and rotate act along world axes. Scale acts along the node's own
//! axes, and the white centre square scales uniformly. Edits are written to
//! `EditorState::node_edits`; the plugin pushes them into the mesh entities.

use egui::{Color32, Id, LayerId, Order, Pos2, Rect, Sense, Shape, Stroke};
use engine::rendering::camera::Camera;
use engine::rendering::mesh::Mesh;
use engine_core::world::World;
use ultraviolet::{Mat4, Vec3, Vec4};

use crate::math::{self, Quat};
use crate::state::{EditorState, GizmoMode, Selection};

const GIZMO_PX: f32 = 90.0;
const PICK_PX: f32 = 8.0;
const MIN_AXIS_PX: f32 = 10.0;
const RING_SEGMENTS: usize = 64;
const MIN_SCALE: f32 = 1e-3;
const UNIFORM_SCALE_PER_PX: f32 = 0.01;

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

/// State of an in-progress drag, kept in egui temp memory between frames.
#[derive(Clone)]
struct Drag {
    node: usize,
    mode: GizmoMode,
    handle: Handle,
    axis: Vec3,
    start_translation: [f32; 3],
    start_rotation: Quat,
    start_scale: [f32; 3],
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
        let view_proj = camera.view_proj_matrix();
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

    draw_mode_toolbar(ctx, state, viewport);

    let drag_id = Id::unique("gizmo_drag");
    let pressed_id = Id::unique("gizmo_pressed_handle");
    let mut drag: Option<Drag> = ctx.data(|d| d.get_temp(drag_id));

    let selected_node = match state.selected {
        Some(Selection::GltfNode(idx)) => Some(idx),
        _ => None,
    };
    if drag.as_ref().is_some_and(|d| Some(d.node) != selected_node || d.mode != state.gizmo_mode) {
        drag = None;
    }

    let pointer = ctx.input(|i| i.pointer.hover_pos());
    let gizmo_layer = LayerId::new(Order::Middle, Id::unique("gizmo_input"));
    let pointer_free = pointer.is_some_and(|p| {
        viewport.contains(p) && ctx.layer_id_at(p).is_none_or(|layer| layer == gizmo_layer)
    });

    let mut hovered = None;
    if let Some(node) = selected_node {
        let world_matrix = state.node_world(node);
        let center = world_matrix.extract_translation();
        let len = view.world_size(center, GIZMO_PX);
        let axes = gizmo_axes(state.gizmo_mode, &world_matrix);

        if drag.is_none() && pointer_free {
            hovered = pointer.and_then(|p| hit_test(&view, state.gizmo_mode, center, axes, len, p));
        }

        let active = drag.as_ref().map(|d| d.handle).or(hovered);
        let painter = ctx
            .layer_painter(LayerId::new(Order::Background, Id::unique("gizmo_paint")))
            .with_clip_rect(viewport);
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
                state.ensure_node_edit(node);
                drag = begin_drag(state, &view, node, handle, axes, len, start_pointer);
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
        let picked = pick(world, state, view.ray(p));
        state.selected = picked.map(Selection::GltfNode);
        if let Some(node) = picked {
            state.ensure_node_edit(node);
        }
    }
}

fn draw_mode_toolbar(ctx: &egui::Context, state: &mut EditorState, viewport: Rect) {
    if !ctx.egui_wants_keyboard_input() {
        ctx.input(|i| {
            if i.key_pressed(egui::Key::Num1) {
                state.gizmo_mode = GizmoMode::Translate;
            } else if i.key_pressed(egui::Key::Num2) {
                state.gizmo_mode = GizmoMode::Rotate;
            } else if i.key_pressed(egui::Key::Num3) {
                state.gizmo_mode = GizmoMode::Scale;
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
                        ui.selectable_value(&mut state.gizmo_mode, GizmoMode::Translate, "Move (1)");
                        ui.selectable_value(&mut state.gizmo_mode, GizmoMode::Rotate, "Rotate (2)");
                        ui.selectable_value(&mut state.gizmo_mode, GizmoMode::Scale, "Scale (3)");
                    });
                });
        });
}

/// World axes for translate/rotate, the node's own (normalized) axes for scale.
fn gizmo_axes(mode: GizmoMode, world_matrix: &Mat4) -> [Vec3; 3] {
    match mode {
        GizmoMode::Translate | GizmoMode::Rotate => [Vec3::unit_x(), Vec3::unit_y(), Vec3::unit_z()],
        GizmoMode::Scale => {
            let cols = world_matrix.truncate().cols;
            let fallback = [Vec3::unit_x(), Vec3::unit_y(), Vec3::unit_z()];
            std::array::from_fn(|i| {
                let len = cols[i].mag();
                if len > 1e-6 { cols[i] / len } else { fallback[i] }
            })
        }
    }
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

fn hit_test(view: &View, mode: GizmoMode, center: Vec3, axes: [Vec3; 3], len: f32, pointer: Pos2) -> Option<Handle> {
    let origin = view.project(center)?;

    if mode == GizmoMode::Scale && origin.distance(pointer) < PICK_PX * 1.5 {
        return Some(Handle::Uniform);
    }

    let mut best: Option<(f32, Handle)> = None;
    for (i, &axis) in axes.iter().enumerate() {
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
    axes: [Vec3; 3],
    len: f32,
    active: Option<Handle>,
) {
    let Some(origin) = view.project(center) else { return };
    let color_for = |handle: Handle, base: Color32| if active == Some(handle) { ACTIVE_COLOR } else { base };

    for (i, &axis) in axes.iter().enumerate() {
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
    node: usize,
    handle: Handle,
    axes: [Vec3; 3],
    len: f32,
    start_pointer: Pos2,
) -> Option<Drag> {
    let edit = state.node_edits.get(&node)?;
    let parent_world = state.parent_world(node);
    let start_center = state.node_world(node).extract_translation();
    let axis = match handle {
        Handle::Axis(i) => axes[i],
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
        node,
        mode: state.gizmo_mode,
        handle,
        axis,
        start_translation: edit.translation,
        start_rotation: edit.rotation,
        start_scale: edit.scale,
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
    let Some(edit) = state.node_edits.get_mut(&drag.node) else { return };

    match (drag.mode, drag.handle) {
        (GizmoMode::Translate, Handle::Axis(_)) => {
            let Some(t) = axis_param(drag.start_center, drag.axis, ray) else { return };
            let world_pos = drag.start_center + drag.axis * (t - drag.start_param);
            edit.translation = drag.parent_world.inversed().transform_point3(world_pos).into();
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
            edit.rotation = math::quat_normalize(math::quat_mul(local_delta, drag.start_rotation));
        }
        (GizmoMode::Scale, Handle::Axis(i)) => {
            let Some(t) = axis_param(drag.start_center, drag.axis, ray) else { return };
            let factor = t / drag.start_param;
            edit.scale[i] = (drag.start_scale[i] * factor).max(MIN_SCALE);
        }
        (GizmoMode::Scale, Handle::Uniform) => {
            let factor = ((pointer.x - drag.start_pointer.x) * UNIFORM_SCALE_PER_PX).exp();
            edit.scale = drag.start_scale.map(|s| (s * factor).max(MIN_SCALE));
        }
        _ => {}
    }

    if drag.mode != GizmoMode::Translate {
        edit.translation = drag.start_translation;
    }
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

/// Nearest glTF node whose mesh bounds the ray hits.
fn pick(world: &World, state: &EditorState, ray: (Vec3, Vec3)) -> Option<usize> {
    let mut best: Option<(f32, usize)> = None;
    for handle in world.get_entity_handles::<Mesh>() {
        let Some(&node) = state.entity_nodes.get(&handle.id) else { continue };
        let mesh = handle.entity;
        if mesh.vertices.is_empty() {
            continue;
        }

        let (mut lo, mut hi) = (Vec3::broadcast(f32::MAX), Vec3::broadcast(f32::MIN));
        for v in &mesh.vertices {
            let p = Vec3::from(v.position);
            lo = lo.min_by_component(p);
            hi = hi.max_by_component(p);
        }

        let (mut wlo, mut whi) = (Vec3::broadcast(f32::MAX), Vec3::broadcast(f32::MIN));
        for corner in 0..8 {
            let local = Vec3::new(
                if corner & 1 == 0 { lo.x } else { hi.x },
                if corner & 2 == 0 { lo.y } else { hi.y },
                if corner & 4 == 0 { lo.z } else { hi.z },
            );
            let p = mesh.transform.transform_point3(local);
            wlo = wlo.min_by_component(p);
            whi = whi.max_by_component(p);
        }

        if let Some(t) = ray_aabb(ray, wlo, whi)
            && best.is_none_or(|(bt, _)| t < bt)
        {
            best = Some((t, node));
        }
    }
    best.map(|(_, node)| node)
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

    #[test]
    fn ray_hits_box_in_front_only() {
        let lo = Vec3::broadcast(-1.0);
        let hi = Vec3::broadcast(1.0);
        let t = ray_aabb((Vec3::new(0.0, 0.0, 5.0), -Vec3::unit_z()), lo, hi).unwrap();
        assert!((t - 4.0).abs() < 1e-4);
        assert!(ray_aabb((Vec3::new(0.0, 0.0, 5.0), Vec3::unit_z()), lo, hi).is_none());
    }
}
