use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use engine::Engine;
use engine::EntityKey;
use engine::assets::AssetLoadHandle;
use engine::gltf::{GLTFAsset, GltfScene};
use engine::plugin::{EnginePluginError, Plugin};
use engine::rendering::camera::Camera;
use engine::rendering::light::{AreaLight, PointLight, SkyLight, SpotLight};
use engine::rendering::mesh::Mesh;

use crate::camera::EditorCameraController;
use crate::math;
use crate::project::Project;
use crate::state::{LightData, NodeKind, SceneLight, SceneNode, Selection, SharedState};

/// TRS of every edited node, used to detect when transforms must be reapplied.
type TransformSnapshot = Vec<(usize, [f32; 10])>;

/// Quiet period after the last project change before `project.json` is written.
const PROJECT_SAVE_DELAY: Duration = Duration::from_millis(500);

pub struct EditorPlugin {
    state: SharedState,
    camera_controller: Option<EditorCameraController>,
    scene: Option<GltfScene>,
    pending_gltf: Option<AssetLoadHandle<GLTFAsset>>,
    watcher: Option<notify::RecommendedWatcher>,
    watch_rx: Option<mpsc::Receiver<notify::Result<notify::Event>>>,
    last_reload: Option<Instant>,
    ignore_changes_until: Option<Instant>,
    applied_transforms: TransformSnapshot,
    applied_entities: usize,
    /// Project as last written to (or read from) `project.json`.
    saved_project: Option<Project>,
    /// Unsaved project contents and when they were last changed.
    pending_project: Option<(Project, Instant)>,
}

impl EditorPlugin {
    pub fn new(state: SharedState) -> Self {
        Self {
            state,
            camera_controller: None,
            scene: None,
            pending_gltf: None,
            watcher: None,
            watch_rx: None,
            last_reload: None,
            ignore_changes_until: None,
            applied_transforms: Vec::new(),
            applied_entities: 0,
            saved_project: None,
            pending_project: None,
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
            exposure: 1.0,
        });
        self.camera_controller = Some(EditorCameraController::new(camera_key));

        camera_key
    }
}

impl Plugin for EditorPlugin {
    fn setup(&mut self, engine: &mut Engine) -> Result<(), EnginePluginError> {
        let camera_key = self.reset_world(engine);
        let mut s = self.state.lock().expect("editor state mutex poisoned");
        s.editor_camera_key = Some(camera_key);
        Ok(())
    }

    fn tick(&mut self, engine: &mut Engine, delta: f32) {
        self.handle_signals(engine);
        self.poll_load(engine);
        self.check_file_changes(engine);
        self.handle_light_requests(engine);
        self.apply_edits(engine);
        self.sync_project();

        if let Some(scene) = &mut self.scene {
            scene.load_batch(&mut engine.world(), 10);
        }
        self.apply_node_transforms(engine);

        let (in_camera_preview, mut settings) = {
            let s = self.state.lock().expect("editor state mutex poisoned");
            (s.camera_preview_node.is_some(), s.camera_settings.clone())
        };
        if !in_camera_preview && let Some(ctrl) = &mut self.camera_controller {
            let ui_wants_pointer = engine.ui_wants_pointer();
            let ui_wants_keyboard = engine.ui_wants_keyboard();
            let speed = settings.move_speed;
            {
                let (input, mut world) = engine.input_and_world();
                ctrl.tick(input, &mut world, &mut settings, delta, ui_wants_pointer, ui_wants_keyboard);
            }
            if settings.move_speed != speed {
                self.state.lock().expect("editor state mutex poisoned").camera_settings.move_speed = settings.move_speed;
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
        self.handle_import_scene();
        self.handle_save_scene();
        self.handle_export_scene();
    }

    fn handle_new_project(&mut self) {
        let needs_dialog = {
            let s = self.state.lock().expect("editor state mutex poisoned");
            s.new_project_requested && s.new_project_path.is_none()
        };
        if needs_dialog {
            let picked = rfd::FileDialog::new().pick_folder();
            let mut s = self.state.lock().expect("editor state mutex poisoned");
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

        let pending = self.state.lock().expect("editor state mutex poisoned").new_project_path.take();
        if let Some((dir, name)) = pending {
            let result = Project::create(&dir, &name);
            let state = Arc::clone(&self.state);
            let mut s = state.lock().expect("editor state mutex poisoned");
            s.new_project_requested = false;
            match result {
                Ok(project) => {
                    s.status = format!("Created project \"{}\"", project.name);
                    Self::record_recent(&mut s, &project.name, &dir);
                    self.set_project(&mut s, project, dir);
                }
                Err(e) => s.status = format!("New project failed: {e}"),
            }
        }
    }

    fn handle_open_project(&mut self) {
        let needs_dialog = {
            let s = self.state.lock().expect("editor state mutex poisoned");
            s.open_project_requested && s.open_project_path.is_none()
        };
        if needs_dialog {
            let picked = rfd::FileDialog::new().pick_folder();
            let mut s = self.state.lock().expect("editor state mutex poisoned");
            match picked {
                Some(dir) => s.open_project_path = Some(dir),
                None => s.open_project_requested = false,
            }
        }

        let pending = self.state.lock().expect("editor state mutex poisoned").open_project_path.take();
        if let Some(dir) = pending {
            self.load_project(dir);
        }
    }

    fn handle_open_recent(&mut self, _engine: &mut Engine) {
        let path = self.state.lock().expect("editor state mutex poisoned").open_recent_path.take();
        if let Some(dir) = path {
            self.load_project(dir);
        }
    }

    fn load_project(&mut self, dir: PathBuf) {
        let result = Project::load(&dir);
        let state = Arc::clone(&self.state);
        let mut s = state.lock().expect("editor state mutex poisoned");
        s.open_project_requested = false;
        match result {
            Ok(project) => {
                let default_scene = project.default_scene_path(&dir);
                s.status = format!("Opened project \"{}\"", project.name);
                Self::record_recent(&mut s, &project.name, &dir);
                self.set_project(&mut s, project, dir);
                if let Some(scene) = default_scene {
                    s.load_path = Some(scene);
                }
            }
            Err(e) => s.status = format!("Open project failed: {e}"),
        }
    }

    fn set_project(&mut self, s: &mut crate::state::EditorState, project: Project, dir: PathBuf) {
        s.camera_settings = project.camera.clone();
        s.project_scenes = Project::list_scenes(&dir);
        self.saved_project = Some(project.clone());
        self.pending_project = None;
        s.project = Some(project);
        s.project_dir = Some(dir);
    }

    /// Writes `project.json` once project settings have stopped changing for a moment.
    fn sync_project(&mut self) {
        let mut s = self.state.lock().expect("editor state mutex poisoned");
        let camera = s.camera_settings.clone();
        let Some(project) = s.project.as_mut() else { return };
        project.camera = camera;
        if self.saved_project.as_ref() == Some(project) {
            self.pending_project = None;
            return;
        }

        let now = Instant::now();
        match &self.pending_project {
            Some((pending, changed_at)) if pending == project => {
                if now.duration_since(*changed_at) < PROJECT_SAVE_DELAY {
                    return;
                }
            }
            _ => {
                self.pending_project = Some((project.clone(), now));
                return;
            }
        }

        let project = project.clone();
        let Some(dir) = s.project_dir.clone() else { return };
        self.pending_project = None;
        // On failure the snapshot still advances, so the write is retried on the next change rather than every tick.
        if let Err(e) = project.save(&dir) {
            s.status = format!("Saving project settings failed: {e:#}");
        }
        self.saved_project = Some(project);
    }

    fn refresh_project_scenes(s: &mut crate::state::EditorState) {
        if let Some(dir) = &s.project_dir {
            s.project_scenes = Project::list_scenes(dir);
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
            let s = self.state.lock().expect("editor state mutex poisoned");
            s.open_requested && s.load_path.is_none()
        };
        if needs_dialog {
            let start_dir = self.state.lock().expect("editor state mutex poisoned")
                .project_dir.as_ref()
                .map(|d| Project::scenes_dir(d));
            let mut dialog = rfd::FileDialog::new().add_filter("GLTF / GLB", &["gltf", "glb"]);
            if let Some(dir) = start_dir {
                dialog = dialog.set_directory(dir);
            }
            let picked = dialog.pick_file();
            let mut s = self.state.lock().expect("editor state mutex poisoned");
            match picked {
                Some(path) => s.load_path = Some(path),
                None => s.open_requested = false,
            }
        }

        let load_path = self.state.lock().expect("editor state mutex poisoned").load_path.take();
        if let Some(path) = load_path {
            self.start_load(engine, path);
        }
    }

    fn handle_import_scene(&mut self) {
        let needs_dialog = {
            let s = self.state.lock().expect("editor state mutex poisoned");
            s.import_requested && s.import_path.is_none() && s.project_dir.is_some()
        };
        if needs_dialog {
            let picked = rfd::FileDialog::new()
                .add_filter("GLTF / GLB", &["gltf", "glb"])
                .pick_file();
            let mut s = self.state.lock().expect("editor state mutex poisoned");
            match picked {
                Some(path) => s.import_path = Some(path),
                None => s.import_requested = false,
            }
        }

        let (source, project_dir) = {
            let mut s = self.state.lock().expect("editor state mutex poisoned");
            match (s.import_path.take(), s.project_dir.clone()) {
                (Some(source), Some(dir)) => (source, dir),
                _ => return,
            }
        };
        let result = crate::import::import_scene(&source, &project_dir);
        let mut s = self.state.lock().expect("editor state mutex poisoned");
        s.import_requested = false;
        match result {
            Ok(scene) => {
                s.status = format!("Imported {}", scene.display());
                s.load_path = Some(scene);
                Self::refresh_project_scenes(&mut s);
            }
            Err(e) => s.status = format!("Import failed: {e:#}"),
        }
    }

    fn handle_save_scene(&mut self) {
        let mut s = self.state.lock().expect("editor state mutex poisoned");
        if !std::mem::take(&mut s.save_requested) {
            return;
        }
        let Some(path) = s.file_path.clone() else { return };
        if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("glb")) {
            s.status = "Save failed: .glb scenes can't be saved in place, use Export Scene…".to_string();
            return;
        }

        let result = crate::export::write_gltf(&s, &path);
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        s.status = match result {
            Ok(()) => {
                // Our own write would otherwise trigger a hot reload that drops lights and selection.
                self.ignore_changes_until = Some(Instant::now() + Duration::from_millis(500));
                format!("Saved {name}")
            }
            Err(e) => format!("Save failed: {e:#}"),
        };
    }

    fn handle_export_scene(&mut self) {
        let needs_dialog = {
            let s = self.state.lock().expect("editor state mutex poisoned");
            s.export_requested && s.export_path.is_none() && s.document.is_some()
        };
        if needs_dialog {
            let default_dir = {
                let s = self.state.lock().expect("editor state mutex poisoned");
                s.project_dir.as_ref()
                    .map(|d| Project::scenes_dir(d))
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
            let mut s = self.state.lock().expect("editor state mutex poisoned");
            match picked {
                Some(path) => s.export_path = Some(path),
                None => s.export_requested = false,
            }
        }

        let export_path = self.state.lock().expect("editor state mutex poisoned").export_path.take();
        if let Some(path) = export_path {
            let result = {
                let s = self.state.lock().expect("editor state mutex poisoned");
                crate::export::write_gltf(&s, &path)
            };
            let mut s = self.state.lock().expect("editor state mutex poisoned");
            s.export_requested = false;
            Self::refresh_project_scenes(&mut s);
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
        self.applied_transforms.clear();
        self.applied_entities = 0;
        let handle = engine.assets().load_async::<GLTFAsset>(path.to_str().unwrap_or(""));
        self.pending_gltf = Some(handle);

        let mut s = self.state.lock().expect("editor state mutex poisoned");
        s.editor_camera_key = Some(camera_key);
        s.lights.clear();
        s.file_path = Some(path);
        s.document = None;
        s.scene_nodes.clear();
        s.node_parents.clear();
        s.entity_nodes.clear();
        s.node_edits.clear();
        s.selected = None;
        s.camera_preview_node = None;
        s.open_requested = false;
        s.status = "Loading…".to_string();
    }

    fn poll_load(&mut self, engine: &mut Engine) {
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
                let (lights, lights_error) = match crate::lights::read_lights(&doc) {
                    Ok(lights) => (lights, None),
                    Err(e) => (Vec::new(), Some(e)),
                };
                let lights: Vec<SceneLight> = lights
                    .into_iter()
                    .map(|(node, data)| SceneLight { key: spawn_light(engine, &data), data, node })
                    .collect();
                self.scene = Some(GltfScene::new(Arc::clone(&asset)));
                let scene_path = {
                    let mut s = self.state.lock().expect("editor state mutex poisoned");
                    let path_display = s.file_path.as_ref()
                        .map(|p| p.file_name().unwrap_or_default().to_string_lossy().into_owned())
                        .unwrap_or_default();
                    s.document = Some(doc.clone());
                    s.scene_nodes = nodes;
                    s.node_parents = doc
                        .nodes()
                        .flat_map(|parent| parent.children().map(move |child| (child.index(), parent.index())))
                        .collect();
                    s.lights = lights;
                    s.status = match lights_error {
                        Some(e) => format!("Loaded: {path_display} (lights skipped: {e:#})"),
                        None => format!("Loaded: {path_display}"),
                    };
                    s.file_path.clone()
                };
                if let Some(path) = scene_path {
                    self.setup_watcher(&path, &doc);
                }
            }
            Err(e) => {
                let mut s = self.state.lock().expect("editor state mutex poisoned");
                s.status = format!("Load error: {e}");
            }
        }
    }

    fn handle_light_requests(&mut self, engine: &mut Engine) {
        let request = self.state.lock().expect("editor state mutex poisoned").add_light_requested.take();
        let Some(kind) = request else { return };

        let data = LightData::default_for(kind);
        let key = spawn_light(engine, &data);
        self.state.lock().expect("editor state mutex poisoned").lights.push(SceneLight { key, data, node: None });
    }

    fn apply_edits(&self, engine: &mut Engine) {
        let s = self.state.lock().expect("editor state mutex poisoned");

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

        let s = self.state.lock().expect("editor state mutex poisoned");

        if let (Some(cam_key), Some(idx)) = (s.editor_camera_key, s.camera_preview_node) {
            if let Some(edit) = s.node_edits.get(&idx) {
                let [tx, ty, tz] = edit.translation;
                let [qx, qy, qz, qw] = edit.rotation;
                let fov = edit.fov_y_deg;
                let near = edit.near_plane;
                let far = edit.far_plane;
                drop(s);

                let eye = ultraviolet::Vec3::new(tx, ty, tz);
                let rotation = [qx, qy, qz, qw];
                let forward = math::rotate(rotation, ultraviolet::Vec3::new(0.0, 0.0, -1.0));
                let up = math::rotate(rotation, ultraviolet::Vec3::new(0.0, 1.0, 0.0));

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

    /// Pushes node world transforms into their mesh entities whenever a node
    /// edit changes or new meshes have streamed in.
    fn apply_node_transforms(&mut self, engine: &mut Engine) {
        let Some(scene) = &self.scene else { return };
        let loaded: usize = scene.node_entities().values().map(Vec::len).sum();

        let mut s = self.state.lock().expect("editor state mutex poisoned");
        let mut snapshot: TransformSnapshot = s
            .node_edits
            .iter()
            .map(|(&idx, e)| {
                let [tx, ty, tz] = e.translation;
                let [rx, ry, rz, rw] = e.rotation;
                let [sx, sy, sz] = e.scale;
                (idx, [tx, ty, tz, rx, ry, rz, rw, sx, sy, sz])
            })
            .collect();
        snapshot.sort_unstable_by_key(|(idx, _)| *idx);

        if snapshot == self.applied_transforms && loaded == self.applied_entities {
            return;
        }

        if loaded != self.applied_entities {
            s.entity_nodes = scene
                .node_entities()
                .iter()
                .flat_map(|(&node, keys)| keys.iter().map(move |&key| (key, node)))
                .collect();
        }

        let worlds: HashMap<usize, ultraviolet::Mat4> = scene
            .node_entities()
            .keys()
            .map(|&node| (node, s.node_world(node)))
            .collect();
        drop(s);

        let mut world = engine.world();
        for (node, keys) in scene.node_entities() {
            for &key in keys {
                if let Some(mesh) = world.get_entity_mut::<Mesh>(key) {
                    mesh.transform = worlds[node];
                }
            }
        }

        self.applied_transforms = snapshot;
        self.applied_entities = loaded;
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
            if let gltf_api::buffer::Source::Uri(uri) = buffer.source()
                && !uri.starts_with("data:")
            {
                watch(&mut watcher, &base.join(uri));
            }
        }
        for image in doc.images() {
            if let gltf_api::image::Source::Uri { uri, .. } = image.source()
                && !uri.starts_with("data:")
            {
                watch(&mut watcher, &base.join(uri));
            }
        }

        self.watcher = Some(watcher);
        self.watch_rx = Some(rx);
    }

    fn check_file_changes(&mut self, engine: &mut Engine) {
        use notify::EventKind;

        let changed = match &self.watch_rx {
            None => return,
            // Drain every event, so none of a save's events linger past the ignore window.
            Some(rx) => rx
                .try_iter()
                .filter(|res| {
                    matches!(
                        res,
                        Ok(notify::Event { kind: EventKind::Modify(_) | EventKind::Create(_), .. })
                    )
                })
                .count()
                > 0,
        };

        let now = Instant::now();
        if !changed || self.ignore_changes_until.is_some_and(|t| now < t) {
            return;
        }

        let debounce_elapsed = self.last_reload
            .map(|t| now.duration_since(t).as_millis() >= 300)
            .unwrap_or(true);

        if !debounce_elapsed {
            return;
        }

        let path = match self.state.lock().expect("editor state mutex poisoned").file_path.clone() {
            Some(p) => p,
            None => return,
        };

        self.last_reload = Some(now);
        self.start_load(engine, path);
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
        } else if node.light().is_some() {
            NodeKind::Light
        } else {
            NodeKind::Empty
        },
        children: node.children().map(|c| node_snapshot(&c)).collect(),
    }
}

fn spawn_light(engine: &mut Engine, data: &LightData) -> EntityKey {
    match data {
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
    }
}
