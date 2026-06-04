use std::path::PathBuf;
use std::sync::Arc;

use engine::Engine;
use engine::EntityKey;
use engine::assets::AssetLoadHandle;
use engine::gltf::{GLTFAsset, GltfScene};
use engine::plugin::{EnginePluginError, Plugin};
use engine::rendering::camera::Camera;
use engine::player_controller::PlayerController;
use engine::rendering::camera_controller::{CameraController, CameraMode};
use engine::rendering::light::SkyLight;

use crate::state::{NodeKind, SceneNode, Selection, SharedState};

pub struct EditorPlugin {
    state: SharedState,
    camera_controller: Option<CameraController>,
    player_controller: PlayerController,
    scene: Option<GltfScene>,
    pending_gltf: Option<AssetLoadHandle<GLTFAsset>>,
}

impl EditorPlugin {
    pub fn new(state: SharedState) -> Self {
        Self {
            state,
            camera_controller: None,
            player_controller: PlayerController::new(20.0),
            scene: None,
            pending_gltf: None,
        }
    }

    fn reset_world(&mut self, engine: &mut Engine) -> (EntityKey, EntityKey) {
        engine.world().clear();

        let (sky_color, sky_intensity, sky_dir) = {
            let s = self.state.lock().unwrap();
            (s.sky_light_color, s.sky_light_intensity, s.sky_light_direction)
        };

        let camera_key = engine.world().add_entity(Camera {
            up: (0.0, 1.0, 0.0).into(),
            eye: (0.0, 5.0, 15.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            aspect: 16.0 / 9.0,
            fovy: 45.0_f32.to_radians(),
            near: 0.01,
            far: 10000.0,
        });
        self.camera_controller = Some(CameraController::new(camera_key, CameraMode::ThirdPerson));
        self.player_controller.attach(camera_key);

        let sky_key = engine.world().add_entity(SkyLight {
            direction: sky_dir.into(),
            color: sky_color.into(),
            intensity: sky_intensity,
        });

        (camera_key, sky_key)
    }
}

impl Plugin for EditorPlugin {
    fn setup(&mut self, engine: &mut Engine) -> Result<(), EnginePluginError> {
        let (camera_key, sky_key) = self.reset_world(engine);
        let mut s = self.state.lock().unwrap();
        s.editor_camera_key = Some(camera_key);
        s.sky_light_key = Some(sky_key);
        Ok(())
    }

    fn tick(&mut self, engine: &mut Engine, delta: f32) {
        self.handle_signals(engine);
        self.poll_load(engine);
        self.apply_edits(engine);

        if let Some(scene) = &mut self.scene {
            scene.load_batch(&mut engine.world(), 10);
        }

        self.player_controller.tick::<Camera>(engine, delta);

        let ui_wants_pointer = engine.ui_wants_pointer();
        if let Some(ctrl) = &mut self.camera_controller {
            let (input, mut world) = engine.input_and_world();
            ctrl.tick(input, &mut world, delta, ui_wants_pointer);
        }
    }
}

impl EditorPlugin {
    fn handle_signals(&mut self, engine: &mut Engine) {
        let needs_open_dialog = {
            let s = self.state.lock().unwrap();
            s.open_requested && s.load_path.is_none()
        };
        if needs_open_dialog {
            let picked = rfd::FileDialog::new()
                .add_filter("GLTF / GLB", &["gltf", "glb"])
                .pick_file();
            let mut s = self.state.lock().unwrap();
            match picked {
                Some(path) => s.load_path = Some(path),
                None => s.open_requested = false,
            }
        }

        let needs_export_dialog = {
            let s = self.state.lock().unwrap();
            s.export_requested && s.export_path.is_none() && s.document.is_some()
        };
        if needs_export_dialog {
            let default_dir = self.state.lock().unwrap()
                .file_path.clone()
                .and_then(|p| p.parent().map(|d| d.to_path_buf()));
            let mut dialog = rfd::FileDialog::new()
                .add_filter("GLTF", &["gltf"])
                .set_file_name("export.gltf");
            if let Some(dir) = default_dir {
                dialog = dialog.set_directory(dir);
            }
            let picked = dialog.save_file();
            let mut s = self.state.lock().unwrap();
            match picked {
                Some(path) => s.export_path = Some(path),
                None => s.export_requested = false,
            }
        }

        let load_path = self.state.lock().unwrap().load_path.take();
        if let Some(path) = load_path {
            self.start_load(engine, path);
        }

        let export_path = self.state.lock().unwrap().export_path.take();
        if let Some(path) = export_path {
            let result = {
                let s = self.state.lock().unwrap();
                crate::export::write_gltf(&s, &path)
            };
            let mut s = self.state.lock().unwrap();
            s.export_requested = false;
            s.status = match result {
                Ok(()) => format!("Exported to {}", path.display()),
                Err(e) => format!("Export failed: {e}"),
            };
        }
    }

    fn start_load(&mut self, engine: &mut Engine, path: PathBuf) {
        let (camera_key, sky_key) = self.reset_world(engine);
        self.scene = None;
        let handle = engine.assets().load_async::<GLTFAsset>(path.to_str().unwrap_or(""));
        self.pending_gltf = Some(handle);

        let mut s = self.state.lock().unwrap();
        s.editor_camera_key = Some(camera_key);
        s.sky_light_key = Some(sky_key);
        s.file_path = Some(path);
        s.document = None;
        s.scene_nodes.clear();
        s.node_edits.clear();
        s.selected = None;
        s.open_requested = false;
        s.status = "Loading…".to_string();
    }

    fn poll_load(&mut self, _engine: &mut Engine) {
        let result = match &self.pending_gltf {
            Some(h) => h.try_get(),
            None => return,
        };
        let Some(result) = result else { return };
        self.pending_gltf = None;

        match result {
            Ok(asset) => {
                let doc = asset.document.clone();
                let nodes = build_scene_nodes(&doc);
                self.scene = Some(GltfScene::new(Arc::clone(&asset)));
                let mut s = self.state.lock().unwrap();
                let path_display = s.file_path.as_ref()
                    .map(|p| p.file_name().unwrap_or_default().to_string_lossy().into_owned())
                    .unwrap_or_default();
                s.document = Some(doc);
                s.scene_nodes = nodes;
                s.status = format!("Loaded: {path_display}");
            }
            Err(e) => {
                let mut s = self.state.lock().unwrap();
                s.status = format!("Load error: {e}");
            }
        }
    }

    fn apply_edits(&self, engine: &mut Engine) {
        let s = self.state.lock().unwrap();

        if let Some(key) = s.sky_light_key {
            let color = s.sky_light_color;
            let intensity = s.sky_light_intensity;
            let direction = s.sky_light_direction;
            drop(s);
            let mut world = engine.world();
            if let Some(light) = world.get_entity_mut::<SkyLight>(key) {
                light.color = color.into();
                light.intensity = intensity;
                light.direction = direction.into();
            }
            return;
        }

        if let (Some(cam_key), Some(Selection::GltfNode(idx))) =
            (s.editor_camera_key, s.selected.clone())
        {
            let edit = s.node_edits.get(&idx).cloned();
            drop(s);
            if let Some(edit) = edit {
                let mut world = engine.world();
                if let Some(cam) = world.get_entity_mut::<Camera>(cam_key) {
                    if let Some(fov_deg) = edit.fov_y_deg {
                        cam.fovy = fov_deg.to_radians();
                    }
                    if let Some(near) = edit.near_plane { cam.near = near; }
                    if let Some(far) = edit.far_plane { cam.far = far; }
                }
            }
        } else {
            drop(s);
        }
    }
}

fn build_scene_nodes(doc: &gltf_api::Document) -> Vec<SceneNode> {
    let root_nodes: Vec<_> = doc.default_scene()
        .map(|s| s.nodes().collect::<Vec<_>>())
        .unwrap_or_else(|| doc.nodes().collect::<Vec<_>>());
    root_nodes.iter().map(node_snapshot).collect()
}

fn node_snapshot(node: &gltf_api::Node) -> SceneNode {
    SceneNode {
        index: node.index(),
        name: node.name().unwrap_or("(unnamed)").to_string(),
        kind: if let Some(mesh) = node.mesh() {
            NodeKind::Mesh(mesh.name().unwrap_or("").to_string())
        } else if node.camera().is_some() {
            NodeKind::Camera
        } else {
            NodeKind::Empty
        },
        children: node.children().map(|c| node_snapshot(&c)).collect(),
    }
}
