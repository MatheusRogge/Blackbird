use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Instant;

use engine::Engine;
use engine::EntityKey;
use engine::assets::AssetLoadHandle;
use engine::gltf::{GLTFAsset, GltfScene};
use engine::plugin::{EnginePluginError, Plugin};
use engine::rendering::camera::Camera;
use engine::player_controller::PlayerController;
use engine::rendering::camera_controller::{CameraController, CameraMode};
use engine::rendering::light::{AreaLight, PointLight, SkyLight, SpotLight};

use crate::state::{LightData, NodeKind, SceneLight, SceneNode, Selection, SharedState};

pub struct EditorPlugin {
    state: SharedState,
    camera_controller: Option<CameraController>,
    player_controller: PlayerController,
    scene: Option<GltfScene>,
    pending_gltf: Option<AssetLoadHandle<GLTFAsset>>,
    watcher: Option<notify::RecommendedWatcher>,
    watch_rx: Option<mpsc::Receiver<notify::Result<notify::Event>>>,
    last_reload: Option<Instant>,
}

impl EditorPlugin {
    pub fn new(state: SharedState) -> Self {
        Self {
            state,
            camera_controller: None,
            player_controller: PlayerController::new(20.0),
            scene: None,
            pending_gltf: None,
            watcher: None,
            watch_rx: None,
            last_reload: None,
        }
    }

    fn reset_world(&mut self, engine: &mut Engine) -> EntityKey {
        engine.world().clear();

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

        camera_key
    }
}

impl Plugin for EditorPlugin {
    fn setup(&mut self, engine: &mut Engine) -> Result<(), EnginePluginError> {
        let camera_key = self.reset_world(engine);
        let mut s = self.state.lock().unwrap();
        s.editor_camera_key = Some(camera_key);
        Ok(())
    }

    fn tick(&mut self, engine: &mut Engine, delta: f32) {
        self.handle_signals(engine);
        self.poll_load(engine);
        self.check_file_changes(engine);
        self.handle_light_requests(engine);
        self.apply_edits(engine);

        if let Some(scene) = &mut self.scene {
            scene.load_batch(&mut engine.world(), 10);
        }

        let in_camera_preview = self.state.lock().unwrap().camera_preview_node.is_some();

        if !in_camera_preview {
            self.player_controller.tick::<Camera>(engine, delta);

            let ui_wants_pointer = engine.ui_wants_pointer();
            if let Some(ctrl) = &mut self.camera_controller {
                let (input, mut world) = engine.input_and_world();
                ctrl.tick(input, &mut world, delta, ui_wants_pointer);
            }
        }
    }
}

impl EditorPlugin {
    fn handle_signals(&mut self, engine: &mut Engine) {
        self.handle_new_project();
        self.handle_open_project();
        self.handle_open_recent(engine);
        self.handle_open_scene(engine);
        self.handle_export_scene();
    }

    fn handle_new_project(&mut self) {
        let needs_dialog = {
            let s = self.state.lock().unwrap();
            s.new_project_requested && s.new_project_path.is_none()
        };
        if needs_dialog {
            let picked = rfd::FileDialog::new().pick_folder();
            let mut s = self.state.lock().unwrap();
            match picked {
                Some(dir) => {
                    let name = dir
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "untitled".to_string());
                    s.new_project_path = Some((dir, name));
                }
                None => s.new_project_requested = false,
            }
        }

        let pending = self.state.lock().unwrap().new_project_path.take();
        if let Some((dir, name)) = pending {
            let result = crate::project::Project::create(&dir, &name);
            let mut s = self.state.lock().unwrap();
            s.new_project_requested = false;
            match result {
                Ok(project) => {
                    s.status = format!("Created project \"{}\"", project.name);
                    Self::record_recent(&mut s, &project.name, &dir);
                    s.project = Some(project);
                    s.project_dir = Some(dir);
                }
                Err(e) => s.status = format!("New project failed: {e}"),
            }
        }
    }

    fn handle_open_project(&mut self) {
        let needs_dialog = {
            let s = self.state.lock().unwrap();
            s.open_project_requested && s.open_project_path.is_none()
        };
        if needs_dialog {
            let picked = rfd::FileDialog::new().pick_folder();
            let mut s = self.state.lock().unwrap();
            match picked {
                Some(dir) => s.open_project_path = Some(dir),
                None => s.open_project_requested = false,
            }
        }

        let pending = self.state.lock().unwrap().open_project_path.take();
        if let Some(dir) = pending {
            self.load_project(dir);
        }
    }

    fn handle_open_recent(&mut self, _engine: &mut Engine) {
        let path = self.state.lock().unwrap().open_recent_path.take();
        if let Some(dir) = path {
            self.load_project(dir);
        }
    }

    fn load_project(&mut self, dir: PathBuf) {
        let result = crate::project::Project::load(&dir);
        let mut s = self.state.lock().unwrap();
        s.open_project_requested = false;
        match result {
            Ok(project) => {
                let default_scene = project.default_scene_path(&dir);
                s.status = format!("Opened project \"{}\"", project.name);
                Self::record_recent(&mut s, &project.name, &dir);
                s.project = Some(project);
                s.project_dir = Some(dir);
                if let Some(scene) = default_scene {
                    s.load_path = Some(scene);
                }
            }
            Err(e) => s.status = format!("Open project failed: {e}"),
        }
    }

    fn record_recent(s: &mut crate::state::EditorState, name: &str, dir: &Path) {
        s.recent_projects.push(name, dir);
        if let Err(e) = s.recent_projects.save() {
            log::warn!("could not save recent projects: {e}");
        }
    }

    fn handle_open_scene(&mut self, engine: &mut Engine) {
        let needs_dialog = {
            let s = self.state.lock().unwrap();
            s.open_requested && s.load_path.is_none()
        };
        if needs_dialog {
            let start_dir = self.state.lock().unwrap()
                .project_dir.as_ref()
                .map(|d| crate::project::Project::scenes_dir(d));
            let mut dialog = rfd::FileDialog::new().add_filter("GLTF / GLB", &["gltf", "glb"]);
            if let Some(dir) = start_dir {
                dialog = dialog.set_directory(dir);
            }
            let picked = dialog.pick_file();
            let mut s = self.state.lock().unwrap();
            match picked {
                Some(path) => s.load_path = Some(path),
                None => s.open_requested = false,
            }
        }

        let load_path = self.state.lock().unwrap().load_path.take();
        if let Some(path) = load_path {
            self.start_load(engine, path);
        }
    }

    fn handle_export_scene(&mut self) {
        let needs_dialog = {
            let s = self.state.lock().unwrap();
            s.export_requested && s.export_path.is_none() && s.document.is_some()
        };
        if needs_dialog {
            let default_dir = {
                let s = self.state.lock().unwrap();
                s.project_dir.as_ref()
                    .map(|d| crate::project::Project::scenes_dir(d))
                    .or_else(|| s.file_path.as_ref()
                        .and_then(|p| p.parent().map(|d| d.to_path_buf())))
            };
            let mut dialog = rfd::FileDialog::new()
                .add_filter("GLTF", &["gltf"])
                .set_file_name("scene.gltf");
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
        self.watcher = None;
        self.watch_rx = None;
        let camera_key = self.reset_world(engine);
        self.scene = None;
        let handle = engine.assets().load_async::<GLTFAsset>(path.to_str().unwrap_or(""));
        self.pending_gltf = Some(handle);

        let mut s = self.state.lock().unwrap();
        s.editor_camera_key = Some(camera_key);
        s.lights.clear();
        s.file_path = Some(path);
        s.document = None;
        s.scene_nodes.clear();
        s.node_edits.clear();
        s.selected = None;
        s.camera_preview_node = None;
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
                let scene_path = {
                    let mut s = self.state.lock().unwrap();
                    let path_display = s.file_path.as_ref()
                        .map(|p| p.file_name().unwrap_or_default().to_string_lossy().into_owned())
                        .unwrap_or_default();
                    s.document = Some(doc.clone());
                    s.scene_nodes = nodes;
                    s.status = format!("Loaded: {path_display}");
                    s.file_path.clone()
                };
                if let Some(path) = scene_path {
                    self.setup_watcher(&path, &doc);
                }
            }
            Err(e) => {
                let mut s = self.state.lock().unwrap();
                s.status = format!("Load error: {e}");
            }
        }
    }

    fn handle_light_requests(&mut self, engine: &mut Engine) {
        let request = self.state.lock().unwrap().add_light_requested.take();
        let Some(kind) = request else { return };

        let data = LightData::default_for(kind);
        let key = match &data {
            LightData::Sky { color, intensity, direction } => engine.world().add_entity(SkyLight {
                direction: (*direction).into(),
                color: (*color).into(),
                intensity: *intensity,
            }),
            LightData::Point { position, color, intensity, radius } => engine.world().add_entity(PointLight {
                position: (*position).into(),
                color: (*color).into(),
                intensity: *intensity,
                radius: *radius,
            }),
            LightData::Spot { position, direction, color, intensity, radius, inner_angle, outer_angle } => {
                engine.world().add_entity(SpotLight {
                    position: (*position).into(),
                    direction: (*direction).into(),
                    color: (*color).into(),
                    intensity: *intensity,
                    radius: *radius,
                    inner_angle: *inner_angle,
                    outer_angle: *outer_angle,
                })
            }
            LightData::Area { position, right, up, color, intensity } => engine.world().add_entity(AreaLight {
                position: (*position).into(),
                right: (*right).into(),
                up: (*up).into(),
                color: (*color).into(),
                intensity: *intensity,
            }),
        };
        self.state.lock().unwrap().lights.push(SceneLight { key, data });
    }

    fn apply_edits(&self, engine: &mut Engine) {
        let s = self.state.lock().unwrap();

        // Update all lights every tick so inspector edits take effect immediately.
        let light_updates: Vec<_> = s.lights.iter()
            .map(|l| (l.key, l.data.clone()))
            .collect();
        drop(s);

        {
            let mut world = engine.world();
            for (key, data) in light_updates {
                match data {
                    LightData::Sky { color, intensity, direction } => {
                        if let Some(l) = world.get_entity_mut::<SkyLight>(key) {
                            l.color = color.into();
                            l.intensity = intensity;
                            l.direction = direction.into();
                        }
                    }
                    LightData::Point { position, color, intensity, radius } => {
                        if let Some(l) = world.get_entity_mut::<PointLight>(key) {
                            l.position = position.into();
                            l.color = color.into();
                            l.intensity = intensity;
                            l.radius = radius;
                        }
                    }
                    LightData::Spot { position, direction, color, intensity, radius, inner_angle, outer_angle } => {
                        if let Some(l) = world.get_entity_mut::<SpotLight>(key) {
                            l.position = position.into();
                            l.direction = direction.into();
                            l.color = color.into();
                            l.intensity = intensity;
                            l.radius = radius;
                            l.inner_angle = inner_angle;
                            l.outer_angle = outer_angle;
                        }
                    }
                    LightData::Area { position, right, up, color, intensity } => {
                        if let Some(l) = world.get_entity_mut::<AreaLight>(key) {
                            l.position = position.into();
                            l.right = right.into();
                            l.up = up.into();
                            l.color = color.into();
                            l.intensity = intensity;
                        }
                    }
                }
            }
        }

        let s = self.state.lock().unwrap();

        if let (Some(cam_key), Some(idx)) = (s.editor_camera_key, s.camera_preview_node) {
            if let Some(edit) = s.node_edits.get(&idx) {
                let [tx, ty, tz] = edit.translation;
                let [qx, qy, qz, qw] = edit.rotation;
                let fov = edit.fov_y_deg;
                let near = edit.near_plane;
                let far = edit.far_plane;
                drop(s);

                let eye = ultraviolet::Vec3::new(tx, ty, tz);
                let forward = rotate_by_quat(ultraviolet::Vec3::new(0.0, 0.0, -1.0), qx, qy, qz, qw);
                let up = rotate_by_quat(ultraviolet::Vec3::new(0.0, 1.0, 0.0), qx, qy, qz, qw);

                let mut world = engine.world();
                if let Some(cam) = world.get_entity_mut::<Camera>(cam_key) {
                    cam.eye = eye;
                    cam.target = eye + forward;
                    cam.up = up;
                    if let Some(fov_deg) = fov { cam.fovy = fov_deg.to_radians(); }
                    if let Some(n) = near { cam.near = n; }
                    if let Some(f) = far { cam.far = f; }
                }
            } else {
                drop(s);
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

    fn setup_watcher(&mut self, scene_path: &Path, doc: &gltf_api::Document) {
        use notify::Watcher;

        let (tx, rx) = mpsc::channel();
        let mut watcher = match notify::RecommendedWatcher::new(
            move |res| { let _ = tx.send(res); },
            notify::Config::default(),
        ) {
            Ok(w) => w,
            Err(e) => { log::warn!("file watcher init failed: {e}"); return; }
        };

        let watch = |w: &mut notify::RecommendedWatcher, path: &Path| {
            if let Err(e) = w.watch(path, notify::RecursiveMode::NonRecursive) {
                log::warn!("could not watch {}: {e}", path.display());
            }
        };

        watch(&mut watcher, scene_path);

        let base = scene_path.parent().unwrap_or(Path::new("."));
        for buffer in doc.buffers() {
            if let gltf_api::buffer::Source::Uri(uri) = buffer.source() {
                if !uri.starts_with("data:") {
                    watch(&mut watcher, &base.join(uri));
                }
            }
        }
        for image in doc.images() {
            if let gltf_api::image::Source::Uri { uri, .. } = image.source() {
                if !uri.starts_with("data:") {
                    watch(&mut watcher, &base.join(uri));
                }
            }
        }

        self.watcher = Some(watcher);
        self.watch_rx = Some(rx);
    }

    fn check_file_changes(&mut self, engine: &mut Engine) {
        use notify::EventKind;

        let changed = match &self.watch_rx {
            None => return,
            Some(rx) => rx.try_iter().any(|res| {
                matches!(
                    res,
                    Ok(notify::Event { kind: EventKind::Modify(_) | EventKind::Create(_), .. })
                )
            }),
        };

        if !changed {
            return;
        }

        let now = Instant::now();
        let debounce_elapsed = self.last_reload
            .map(|t| now.duration_since(t).as_millis() >= 300)
            .unwrap_or(true);

        if !debounce_elapsed {
            return;
        }

        let path = match self.state.lock().unwrap().file_path.clone() {
            Some(p) => p,
            None => return,
        };

        self.last_reload = Some(now);
        self.start_load(engine, path);
    }
}

fn rotate_by_quat(v: ultraviolet::Vec3, qx: f32, qy: f32, qz: f32, qw: f32) -> ultraviolet::Vec3 {
    let q = ultraviolet::Vec3::new(qx, qy, qz);
    let t = q.cross(v) * 2.0;
    v + t * qw + q.cross(t)
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
