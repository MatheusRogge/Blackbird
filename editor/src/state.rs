use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use engine_core::entity::EntityKey;
use gltf_api::Document;
use ultraviolet::Mat4;

pub use engine::gltf::lights::LightData;
pub use project::CameraSettings;

use crate::asset_browser::AssetBrowserState;
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

#[derive(Clone, Debug)]
pub struct NodeEdit {
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub base_color: Option<[f32; 4]>,
    pub metallic_factor: Option<f32>,
    pub roughness_factor: Option<f32>,
    pub emissive_factor: Option<[f32; 3]>,
    pub fov_y_deg: Option<f32>,
    pub near_plane: Option<f32>,
    pub far_plane: Option<f32>,
}

impl Default for NodeEdit {
    fn default() -> Self {
        Self {
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
            base_color: None,
            metallic_factor: None,
            roughness_factor: None,
            emissive_factor: None,
            fov_y_deg: None,
            near_plane: None,
            far_plane: None,
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
            },
            LightKind::Spot => LightData::Spot {
                position: [0.0, 3.0, 0.0],
                direction: [0.0, -1.0, 0.0],
                color: [1.0, 1.0, 1.0],
                intensity: 80.0,
                radius: 10.0,
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

pub struct SceneLight {
    pub key: EntityKey,
    pub data: LightData,
    /// glTF node carrying the light's `KHR_lights_punctual` entry, if it was loaded from one.
    pub node: Option<usize>,
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
    pub gizmo_mode: GizmoMode,
    /// Light markers for unselected lights. The selection's gizmo is always drawn.
    pub show_gizmos: bool,

    pub editor_camera_key: Option<EntityKey>,
    pub camera_settings: CameraSettings,

    pub lights: Vec<SceneLight>,
    pub add_light_requested: Option<LightKind>,

    pub camera_preview_node: Option<usize>,

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
            gizmo_mode: GizmoMode::default(),
            show_gizmos: true,
            camera_preview_node: None,
            editor_camera_key: None,
            camera_settings: CameraSettings::default(),
            lights: Vec::new(),
            add_light_requested: None,
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
                edit.fov_y_deg = Some(p.yfov().to_degrees());
                edit.near_plane = Some(p.znear());
                edit.far_plane = p.zfar();
            }
        }
        self.node_edits.insert(node_idx, edit);
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
