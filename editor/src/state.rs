use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use engine_core::entity::EntityKey;
use gltf_api::Document;

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

#[derive(Clone, Debug, PartialEq)]
pub enum Selection {
    GltfNode(usize),
    SkyLight,
}

pub struct EditorState {
    pub file_path: Option<PathBuf>,
    pub document: Option<Document>,
    pub scene_nodes: Vec<SceneNode>,

    pub selected: Option<Selection>,
    pub node_edits: HashMap<usize, NodeEdit>,

    pub editor_camera_key: Option<EntityKey>,
    pub sky_light_key: Option<EntityKey>,

    pub sky_light_color: [f32; 3],
    pub sky_light_intensity: f32,
    pub sky_light_direction: [f32; 3],

    pub open_requested: bool,
    pub export_requested: bool,
    pub load_path: Option<PathBuf>,
    pub export_path: Option<PathBuf>,

    pub status: String,
}

impl EditorState {
    pub fn new() -> SharedState {
        Arc::new(Mutex::new(Self {
            file_path: None,
            document: None,
            scene_nodes: Vec::new(),
            selected: None,
            node_edits: HashMap::new(),
            editor_camera_key: None,
            sky_light_key: None,
            sky_light_color: [0.8, 0.9, 1.0],
            sky_light_intensity: 1.5,
            sky_light_direction: [-0.3, -1.0, 0.2],
            open_requested: false,
            export_requested: false,
            load_path: None,
            export_path: None,
            status: "No file loaded.".to_string(),
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
            match node.transform() {
                gltf_api::scene::Transform::Decomposed { translation, rotation, scale } => {
                    edit.translation = translation;
                    edit.rotation = rotation;
                    edit.scale = scale;
                }
                gltf_api::scene::Transform::Matrix { .. } => {}
            }
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
}
