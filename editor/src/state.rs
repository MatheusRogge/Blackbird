use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use engine_core::entity::EntityKey;
use gltf_api::Document;
use ultraviolet::{Mat4, Vec3};

pub use engine::gltf::camera::PhysicalCamera;
pub use engine::gltf::lights::LightData;
pub use engine::gltf::objects::{ObjectKind, SceneObject as ObjectData, Shape, ShapeMaterial};
pub use project::CameraSettings;

use crate::asset_browser::AssetBrowserState;
use crate::history::History;
use engine::windowed::CloseGuard;
use crate::math;
use build_tool::BuildKind;
use project::Project;
use crate::recent::RecentProjects;

pub type SharedState = Arc<Mutex<EditorState>>;

#[derive(Clone, Debug)]
pub struct SceneNode {
    pub index: usize,
    pub name: String,
    pub kind: NodeKind,
    pub children: Vec<SceneNode>,
}

#[derive(Clone, Debug)]
pub enum NodeKind {
    Mesh(String),
    Camera,
    /// Carries a `KHR_lights_punctual` light, which is edited as a `SceneLight`.
    Light,
    Empty,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NodeEdit {
    /// Empty means unnamed.
    pub name: String,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub base_color: Option<[f32; 4]>,
    pub metallic_factor: Option<f32>,
    pub roughness_factor: Option<f32>,
    pub emissive_factor: Option<[f32; 3]>,
    pub camera: Option<CameraEdit>,
}

/// A perspective camera's glTF settings plus its Blackbird `extras`.
#[derive(Clone, Debug, PartialEq)]
pub struct CameraEdit {
    pub fov_y_deg: f32,
    pub near: f32,
    /// `None` is an infinite far plane.
    pub far: Option<f32>,
    /// Width / height. `None` follows the viewport.
    pub aspect: Option<f32>,
    /// `None` uses the project exposure.
    pub physical: Option<PhysicalCamera>,
}

impl Default for NodeEdit {
    fn default() -> Self {
        Self {
            name: String::new(),
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
            base_color: None,
            metallic_factor: None,
            roughness_factor: None,
            emissive_factor: None,
            camera: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GizmoMode {
    #[default]
    Translate,
    Rotate,
    Scale,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Selection {
    GltfNode(usize),
    Light(usize),
    /// `EditorObject::id`.
    Object(u64),
}

#[derive(Clone, Debug, PartialEq)]
pub enum LightKind {
    Sky,
    Point,
    Spot,
    Area,
}

/// Editor-only presentation of a light.
pub trait LightDataExt {
    fn icon(&self) -> &'static str;
    fn label(&self) -> &'static str;
    fn default_for(kind: LightKind) -> Self;
}

impl LightDataExt for LightData {
    fn icon(&self) -> &'static str {
        match self {
            LightData::Sky { .. }   => "☀",
            LightData::Point { .. } => "💡",
            LightData::Spot { .. }  => "◎",
            LightData::Area { .. }  => "▪",
        }
    }

    fn label(&self) -> &'static str {
        match self {
            LightData::Sky { .. }   => "☀ Sky Light",
            LightData::Point { .. } => "💡 Point Light",
            LightData::Spot { .. }  => "◎ Spot Light",
            LightData::Area { .. }  => "▪ Area Light",
        }
    }

    fn default_for(kind: LightKind) -> Self {
        match kind {
            LightKind::Sky => LightData::Sky {
                color: [0.8, 0.9, 1.0],
                intensity: 5.0,
                direction: [-0.3, -1.0, 0.2],
            },
            LightKind::Point => LightData::Point {
                position: [0.0, 2.0, 0.0],
                color: [1.0, 1.0, 1.0],
                intensity: 80.0,
                radius: 10.0,
                source_radius: 0.0,
            },
            LightKind::Spot => LightData::Spot {
                position: [0.0, 3.0, 0.0],
                direction: [0.0, -1.0, 0.0],
                color: [1.0, 1.0, 1.0],
                intensity: 80.0,
                radius: 10.0,
                source_radius: 0.0,
                inner_angle: 15.0_f32.to_radians(),
                outer_angle: 30.0_f32.to_radians(),
            },
            LightKind::Area => LightData::Area {
                position: [0.0, 3.0, 0.0],
                right: [1.0, 0.0, 0.0],
                up: [0.0, 0.0, -1.0],
                color: [1.0, 1.0, 1.0],
                intensity: 15.0,
            },
        }
    }
}

/// The plugin keeps one entity per light id, spawning and despawning to match `lights`.
#[derive(Clone, Debug, PartialEq)]
pub struct SceneLight {
    pub id: u64,
    pub data: LightData,
    /// glTF node carrying the light's `KHR_lights_punctual` entry, if it was loaded from one.
    pub node: Option<usize>,
}

/// A shape, placed model or player start (`engine::gltf::objects`). The plugin keeps its
/// entities in sync. A model's `uri` is an absolute path while in the editor; save writes it
/// relative to the scene file.
#[derive(Clone, Debug, PartialEq)]
pub struct EditorObject {
    pub id: u64,
    pub data: ObjectData,
}

impl EditorObject {
    pub fn icon(&self) -> &'static str {
        match self.data.kind {
            ObjectKind::Shape { .. } => "⬛",
            ObjectKind::Model { .. } => "📦",
            ObjectKind::PlayerStart => "🚶",
        }
    }
}

/// A question the editor is waiting on the user for.
#[derive(Clone, Debug, PartialEq)]
pub enum Prompt {
    /// The window's close button was pressed with unsaved changes.
    Exit,
    /// The scene file changed on disk while there are unsaved changes.
    Reload,
    /// The opened scene has a recovery file newer than itself.
    Recover(PathBuf),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PromptAnswer {
    /// Exit: save, then exit. Reload: reload. Recover: restore.
    Yes,
    /// Exit: exit without saving. Reload: keep the changes. Recover: discard the recovery file.
    No,
    /// Exit only: stay.
    Cancel,
}

/// Where auto-save keeps an unsaved scene's recovery copy: a hidden file beside it, so the
/// scene's relative URIs still resolve from it.
pub fn recovery_path(scene: &Path) -> PathBuf {
    let name = scene.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    scene.with_file_name(format!(".{name}.autosave.gltf"))
}

pub fn is_recovery_path(path: &Path) -> bool {
    path.file_name().map(|n| n.to_string_lossy()).is_some_and(|n| n.starts_with('.') && n.ends_with(".autosave.gltf"))
}

/// What the Add menu creates.
#[derive(Clone, Debug, PartialEq)]
pub enum AddKind {
    Light(LightKind),
    Shape(Shape),
    /// Absolute path of a `.gltf` in the project.
    Model(PathBuf),
    PlayerStart,
}

/// Where the Add menu's pick ray met the scene: the point, the surface normal there, and the
/// view direction (for facing new objects away from the camera).
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    pub point: Vec3,
    pub normal: Vec3,
    pub view_dir: Vec3,
}

/// Everything undo / redo restores. `node_edits` only holds edits that differ from the loaded
/// node, so selecting a node (which seeds an edit) isn't a step.
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    node_edits: Vec<(usize, NodeEdit)>,
    deleted_nodes: BTreeSet<usize>,
    lights: Vec<SceneLight>,
    objects: Vec<EditorObject>,
    main_camera: Option<usize>,
}

#[derive(Default)]
pub struct BuildState {
    pub requested: Option<BuildKind>,
    pub running: Option<BuildKind>,
    pub log: Vec<String>,
    pub show_log: bool,
    /// Executable produced by the last successful build.
    pub last_output: Option<PathBuf>,
}

pub struct EditorState {
    pub recent_projects: RecentProjects,

    pub project: Option<Project>,
    pub project_dir: Option<PathBuf>,
    /// Scenes in the project's scenes dir, refreshed by the plugin when they can change.
    pub project_scenes: Vec<String>,

    pub file_path: Option<PathBuf>,
    pub document: Option<Document>,
    pub scene_nodes: Vec<SceneNode>,
    /// Child node index → parent node index.
    pub node_parents: HashMap<usize, usize>,
    /// Mesh entity → glTF node it was loaded from. Filled by the plugin as the
    /// scene streams in.
    pub entity_nodes: HashMap<EntityKey, usize>,

    pub selected: Option<Selection>,
    pub node_edits: HashMap<usize, NodeEdit>,
    /// Each edited node as loaded, to tell real edits from seeded ones.
    node_baselines: HashMap<usize, NodeEdit>,
    /// Deleted glTF nodes (their subtrees go with them): collapsed in the viewport, detached from
    /// the scene on save.
    pub deleted_nodes: HashSet<usize>,
    pub gizmo_mode: GizmoMode,
    /// Light markers for unselected lights. The selection's gizmo is always drawn.
    pub show_gizmos: bool,

    pub editor_camera_key: Option<EntityKey>,
    pub camera_settings: CameraSettings,

    pub lights: Vec<SceneLight>,
    pub objects: Vec<EditorObject>,
    /// Mesh entity → object it belongs to. Filled by the plugin, used for picking.
    pub entity_objects: HashMap<EntityKey, u64>,
    next_id: u64,
    pub history: History<Snapshot>,
    /// The scene as last saved (or loaded). `None` with a document open means it was restored
    /// from a recovery file, so it differs from the file on disk.
    saved: Option<Snapshot>,
    /// Vetoes closing the window while there are unsaved changes.
    pub close_guard: CloseGuard,
    pub prompt: Option<Prompt>,
    /// The user's answer to `prompt`, for the plugin to act on.
    pub prompt_answer: Option<PromptAnswer>,
    pub exit_after_save: bool,

    pub camera_preview_node: Option<usize>,
    /// Node whose camera the player renders from, saved in the scene's `extras`.
    pub main_camera: Option<usize>,

    pub new_project_requested: bool,
    pub open_project_requested: bool,
    pub open_recent_path: Option<PathBuf>,
    pub open_requested: bool,
    pub export_requested: bool,
    pub save_requested: bool,
    pub import_requested: bool,
    pub new_project_path: Option<(PathBuf, String)>,
    pub open_project_path: Option<PathBuf>,
    pub load_path: Option<PathBuf>,
    pub export_path: Option<PathBuf>,
    pub import_path: Option<PathBuf>,

    pub build: BuildState,
    pub assets: AssetBrowserState,

    pub status: String,
}

impl EditorState {
    pub fn new() -> SharedState {
        Arc::new(Mutex::new(Self {
            recent_projects: RecentProjects::load(),
            project: None,
            project_dir: None,
            project_scenes: Vec::new(),
            file_path: None,
            document: None,
            scene_nodes: Vec::new(),
            node_parents: HashMap::new(),
            entity_nodes: HashMap::new(),
            selected: None,
            node_edits: HashMap::new(),
            node_baselines: HashMap::new(),
            deleted_nodes: HashSet::new(),
            gizmo_mode: GizmoMode::default(),
            show_gizmos: true,
            camera_preview_node: None,
            main_camera: None,
            editor_camera_key: None,
            camera_settings: CameraSettings::default(),
            lights: Vec::new(),
            objects: Vec::new(),
            entity_objects: HashMap::new(),
            next_id: 1,
            history: History::default(),
            saved: None,
            close_guard: CloseGuard::default(),
            prompt: None,
            prompt_answer: None,
            exit_after_save: false,
            new_project_requested: false,
            open_project_requested: false,
            open_recent_path: None,
            open_requested: false,
            export_requested: false,
            save_requested: false,
            import_requested: false,
            new_project_path: None,
            open_project_path: None,
            load_path: None,
            export_path: None,
            import_path: None,
            build: BuildState::default(),
            assets: AssetBrowserState::default(),
            status: "No project loaded.".to_string(),
        }))
    }

    pub fn ensure_node_edit(&mut self, node_idx: usize) {
        if self.node_edits.contains_key(&node_idx) {
            return;
        }
        let mut edit = NodeEdit::default();
        if let Some(doc) = &self.document
            && let Some(node) = doc.nodes().nth(node_idx)
        {
            edit.name = node.name().unwrap_or_default().to_string();
            let (translation, rotation, scale) = node.transform().decomposed();
            edit.translation = translation;
            edit.rotation = rotation;
            edit.scale = scale;
            if let Some(mesh) = node.mesh()
                && let Some(prim) = mesh.primitives().next()
            {
                let mat = prim.material();
                let pbr = mat.pbr_metallic_roughness();
                edit.base_color = Some(pbr.base_color_factor());
                edit.metallic_factor = Some(pbr.metallic_factor());
                edit.roughness_factor = Some(pbr.roughness_factor());
                edit.emissive_factor = Some(mat.emissive_factor());
            }
            if let Some(cam) = node.camera()
                && let gltf_api::camera::Projection::Perspective(p) = cam.projection()
            {
                edit.camera = Some(CameraEdit {
                    fov_y_deg: p.yfov().to_degrees(),
                    near: p.znear(),
                    far: p.zfar(),
                    aspect: p.aspect_ratio(),
                    physical: PhysicalCamera::read(&cam),
                });
            }
        }
        self.node_baselines.insert(node_idx, edit.clone());
        self.node_edits.insert(node_idx, edit);
    }

    /// Forgets node edits and baselines, for a new scene.
    pub fn clear_node_edits(&mut self) {
        self.node_edits.clear();
        self.node_baselines.clear();
    }

    pub fn new_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    /// Whether the node or one of its ancestors was deleted.
    pub fn is_node_deleted(&self, node_idx: usize) -> bool {
        let mut current = Some(node_idx);
        while let Some(node) = current {
            if self.deleted_nodes.contains(&node) {
                return true;
            }
            current = self.node_parents.get(&node).copied();
        }
        false
    }

    pub fn snapshot(&self) -> Snapshot {
        let mut node_edits: Vec<_> = self
            .node_edits
            .iter()
            .filter(|(idx, edit)| self.node_baselines.get(idx) != Some(edit))
            .map(|(&idx, edit)| (idx, edit.clone()))
            .collect();
        node_edits.sort_by_key(|(idx, _)| *idx);
        Snapshot {
            node_edits,
            deleted_nodes: self.deleted_nodes.iter().copied().collect(),
            lights: self.lights.clone(),
            objects: self.objects.clone(),
            main_camera: self.main_camera,
        }
    }

    pub fn restore(&mut self, snapshot: Snapshot) {
        for (idx, edit) in &mut self.node_edits {
            if let Some(baseline) = self.node_baselines.get(idx) {
                *edit = baseline.clone();
            }
        }
        for (idx, edit) in snapshot.node_edits {
            self.node_edits.insert(idx, edit);
        }
        self.deleted_nodes = snapshot.deleted_nodes.into_iter().collect();
        self.lights = snapshot.lights;
        self.objects = snapshot.objects;
        self.main_camera = snapshot.main_camera;

        let names: HashMap<usize, String> = self.node_edits.iter().map(|(&idx, e)| (idx, e.name.clone())).collect();
        rename_nodes(&mut self.scene_nodes, &names);
        let valid = match &self.selected {
            Some(Selection::GltfNode(idx)) => !self.is_node_deleted(*idx),
            Some(Selection::Light(idx)) => *idx < self.lights.len(),
            Some(Selection::Object(id)) => self.objects.iter().any(|o| o.id == *id),
            None => true,
        };
        if !valid {
            self.selected = None;
        }
        if self.camera_preview_node.is_some_and(|idx| self.is_node_deleted(idx)) {
            self.camera_preview_node = None;
        }
    }

    /// Records the current scene as what's on disk.
    pub fn mark_saved(&mut self) {
        self.saved = Some(self.snapshot());
    }

    /// Marks the scene as differing from its file (it was restored from a recovery file).
    pub fn mark_unsaved(&mut self) {
        self.saved = None;
    }

    /// Whether the open scene has changes its file doesn't.
    pub fn is_dirty(&self) -> bool {
        self.document.is_some() && self.saved.as_ref() != Some(&self.snapshot())
    }

    pub fn undo(&mut self) {
        let now = self.snapshot();
        match self.history.undo(now) {
            Some(previous) => self.restore(previous),
            None => self.status = "Nothing to undo.".to_string(),
        }
    }

    pub fn redo(&mut self) {
        let now = self.snapshot();
        match self.history.redo(now) {
            Some(next) => self.restore(next),
            None => self.status = "Nothing to redo.".to_string(),
        }
    }

    /// Removes the selection from the scene. Lights on a deleted node go with it, and a light
    /// that was its node's only content takes the node along.
    pub fn delete_selected(&mut self) {
        match self.selected.take() {
            Some(Selection::GltfNode(idx)) => {
                self.deleted_nodes.insert(idx);
                let doomed: Vec<bool> =
                    self.lights.iter().map(|l| l.node.is_some_and(|n| self.is_node_deleted(n))).collect();
                let mut doomed = doomed.into_iter();
                self.lights.retain(|_| !doomed.next().unwrap_or(false));
                if self.main_camera.is_some_and(|n| self.is_node_deleted(n)) {
                    self.main_camera = None;
                }
                if self.camera_preview_node.is_some_and(|n| self.is_node_deleted(n)) {
                    self.camera_preview_node = None;
                }
            }
            Some(Selection::Light(idx)) if idx < self.lights.len() => {
                let light = self.lights.remove(idx);
                if let Some(node) = light.node
                    && self.node_kind(node).is_some_and(|k| matches!(k, NodeKind::Light))
                {
                    self.deleted_nodes.insert(node);
                }
            }
            Some(Selection::Object(id)) => self.objects.retain(|o| o.id != id),
            _ => {}
        }
    }

    fn node_kind(&self, idx: usize) -> Option<&NodeKind> {
        fn find(nodes: &[SceneNode], idx: usize) -> Option<&NodeKind> {
            nodes.iter().find_map(|n| if n.index == idx { Some(&n.kind) } else { find(&n.children, idx) })
        }
        find(&self.scene_nodes, idx)
    }

    /// Adds a light or object at `placement` and selects it.
    pub fn add(&mut self, kind: AddKind, placement: Placement) {
        let Placement { point, normal, view_dir } = placement;
        // Facing away from the camera, upright.
        let heading = {
            let flat = Vec3::new(view_dir.x, 0.0, view_dir.z);
            if flat.mag_sq() > 1e-6 { math::quat_look(flat.normalized()) } else { [0.0, 0.0, 0.0, 1.0] }
        };
        let object = |name: &str, translation: Vec3, rotation, kind| ObjectData {
            name: name.to_string(),
            translation: translation.into(),
            rotation,
            scale: [1.0; 3],
            kind,
        };
        let data = match kind {
            AddKind::Light(kind) => {
                let mut data = LightData::default_for(kind);
                let position: [f32; 3] = (point + normal * LIGHT_OFFSET).into();
                match &mut data {
                    LightData::Sky { .. } => {}
                    LightData::Point { position: p, .. } => *p = position,
                    LightData::Spot { position: p, direction, .. } => {
                        *p = position;
                        *direction = (-normal).into();
                    }
                    LightData::Area { position: p, right, up, .. } => {
                        *p = position;
                        let r = math::rotate(math::quat_look(-normal), Vec3::unit_x());
                        let u = math::rotate(math::quat_look(-normal), Vec3::unit_y());
                        *right = (r * 0.5).into();
                        *up = (u * 0.5).into();
                    }
                }
                let id = self.new_id();
                self.lights.push(SceneLight { id, data, node: None });
                self.selected = Some(Selection::Light(self.lights.len() - 1));
                return;
            }
            AddKind::Shape(shape) => object(
                shape.label(),
                point + normal * shape.half_height(),
                math::quat_between(Vec3::unit_y(), normal),
                ObjectKind::Shape { shape, material: ShapeMaterial::default() },
            ),
            AddKind::Model(path) => {
                let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                object(&name, point, heading, ObjectKind::Model { uri: path.to_string_lossy().into_owned() })
            }
            AddKind::PlayerStart => object("Player Start", point, heading, ObjectKind::PlayerStart),
        };
        let id = self.new_id();
        self.objects.push(EditorObject { id, data });
        self.selected = Some(Selection::Object(id));
    }

    /// The open scene as a project-relative path, if it lives inside the project.
    pub fn scene_in_project(&self) -> Option<String> {
        Project::relative_path(self.project_dir.as_ref()?, self.file_path.as_ref()?)
    }

    /// Node transform relative to its parent, with any edit applied.
    pub fn node_local(&self, node_idx: usize) -> Mat4 {
        if let Some(edit) = self.node_edits.get(&node_idx) {
            return math::trs(edit.translation, edit.rotation, edit.scale);
        }
        self.document
            .as_ref()
            .and_then(|doc| doc.nodes().nth(node_idx))
            .map(|node| node.transform().matrix().into())
            .unwrap_or_else(Mat4::identity)
    }

    pub fn node_world(&self, node_idx: usize) -> Mat4 {
        self.parent_world(node_idx) * self.node_local(node_idx)
    }

    /// World transform of the node's parent (identity for a root node).
    pub fn parent_world(&self, node_idx: usize) -> Mat4 {
        let mut world = Mat4::identity();
        let mut current = self.node_parents.get(&node_idx).copied();
        while let Some(parent) = current {
            world = self.node_local(parent) * world;
            current = self.node_parents.get(&parent).copied();
        }
        world
    }
}

/// How far new lights sit off the surface they were placed on.
const LIGHT_OFFSET: f32 = 0.5;

fn rename_nodes(nodes: &mut [SceneNode], names: &HashMap<usize, String>) {
    for node in nodes {
        if let Some(name) = names.get(&node.index) {
            node.name = if name.is_empty() { "(unnamed)".to_string() } else { name.clone() };
        }
        rename_nodes(&mut node.children, names);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_with_scene() -> SharedState {
        let json = r#"{ "asset": { "version": "2.0" }, "scenes": [{ "nodes": [0] }], "nodes": [{ "name": "A" }] }"#;
        let shared = EditorState::new();
        {
            let mut state = shared.lock().unwrap();
            state.document = Some(gltf_api::Gltf::from_slice(json.as_bytes()).unwrap().document);
            state.mark_saved();
            let now = state.snapshot();
            state.history.commit(now);
        }
        shared
    }

    #[test]
    fn dirty_follows_changes_and_undo() {
        let shared = state_with_scene();
        let mut state = shared.lock().unwrap();
        assert!(!state.is_dirty());

        // Selecting seeds an edit equal to the loaded node: still clean.
        state.ensure_node_edit(0);
        assert!(!state.is_dirty());

        state.node_edits.get_mut(&0).unwrap().translation = [1.0, 0.0, 0.0];
        assert!(state.is_dirty());

        state.undo();
        assert!(!state.is_dirty(), "undoing back to the saved state is clean");
        state.redo();
        assert!(state.is_dirty());
        state.mark_saved();
        assert!(!state.is_dirty());

        state.mark_unsaved();
        assert!(state.is_dirty(), "a restored recovery copy differs from the file");
    }

    #[test]
    fn recovery_files_are_hidden_siblings() {
        let path = recovery_path(Path::new("/project/scenes/level.gltf"));
        assert_eq!(path, Path::new("/project/scenes/.level.autosave.gltf"));
        assert!(is_recovery_path(&path));
        assert!(!is_recovery_path(Path::new("/project/scenes/level.gltf")));
    }
}
