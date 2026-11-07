use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use engine::Engine;
use engine::EntityKey;
use engine::assets::AssetLoadHandle;
use engine::gltf::camera::INFINITE_FAR;
use engine::gltf::{GLTFAsset, GltfScene};
use engine::plugin::{EnginePluginError, Plugin};
use engine::rendering::camera::Camera;
use engine::rendering::mesh::Mesh;
use ultraviolet::{Mat4, Vec3};

use build_tool::{BuildKind, BuildOptions, BuildOutput};
use project::Project;

use crate::asset_browser::AssetWorker;
use crate::camera::EditorCameraController;
use crate::state::{
    EditorObject, EditorState, NodeKind, ObjectKind, Prompt, PromptAnswer, SceneLight, SceneNode, Shape, SharedState,
    Snapshot, is_recovery_path, recovery_path,
};

/// TRS of every edited node, used to detect when transforms must be reapplied.
type TransformSnapshot = Vec<(usize, [f32; 10])>;

/// Quiet period after the last project change before `project.json` is written.
const PROJECT_SAVE_DELAY: Duration = Duration::from_millis(500);
/// Lines of build output kept for the build log window.
const BUILD_LOG_LINES: usize = 4000;
/// Quiet period after the last scene file event before hot reloading.
const RELOAD_QUIET: Duration = Duration::from_millis(500);
/// Delay before retrying a failed hot reload, e.g. one that read a half-written file.
const RELOAD_RETRY_DELAY: Duration = Duration::from_secs(1);
const RELOAD_MAX_RETRIES: u32 = 3;
/// How long an unsaved change waits before auto-save writes the recovery copy.
const AUTOSAVE_DELAY: Duration = Duration::from_secs(60);
/// Lens of the editor's viewport camera, restored after Assume Camera View.
const EDITOR_FOV_Y_DEG: f32 = 45.0;
const EDITOR_NEAR: f32 = 0.01;
const EDITOR_FAR: f32 = 10000.0;

/// A model file placed by objects, loaded once and shared by all of them.
enum ModelAsset {
    Loading(AssetLoadHandle<GLTFAsset>),
    Ready(Arc<GLTFAsset>),
    Failed,
}

/// What an object's entities were built from; when it changes they are rebuilt.
#[derive(Clone, PartialEq)]
enum ObjectSource {
    Shape(Shape),
    Model(String),
    Nothing,
}

impl ObjectSource {
    fn of(kind: &ObjectKind) -> Self {
        match kind {
            ObjectKind::Shape { shape, .. } => Self::Shape(*shape),
            ObjectKind::Model { uri } => Self::Model(uri.clone()),
            ObjectKind::PlayerStart => Self::Nothing,
        }
    }
}

/// Entities spawned for one object.
struct ObjectEntities {
    source: ObjectSource,
    /// Each mesh entity with its transform relative to the object.
    meshes: Vec<(EntityKey, Mat4)>,
    /// Streams a model's meshes in.
    scene: Option<GltfScene>,
    /// The object as last written to its entities.
    applied: Option<EditorObject>,
}

impl ObjectEntities {
    fn new(source: ObjectSource) -> Self {
        Self { source, meshes: Vec::new(), scene: None, applied: None }
    }

    fn despawn(&self, world: &mut engine::world::World) {
        for &(key, _) in &self.meshes {
            world.remove_entity(key);
        }
    }
}

enum BuildEvent {
    Line(String),
    Finished(Result<BuildOutput, String>),
}

pub struct EditorPlugin {
    state: SharedState,
    camera_controller: Option<EditorCameraController>,
    scene: Option<GltfScene>,
    pending_gltf: Option<AssetLoadHandle<GLTFAsset>>,
    watcher: Option<notify::RecommendedWatcher>,
    watch_rx: Option<mpsc::Receiver<notify::Result<notify::Event>>>,
    /// Set when a load is a hot reload: the current scene stays until it succeeds.
    reloading: bool,
    /// Set when a load reads a recovery file in place of the scene: the result is unsaved.
    restoring: bool,
    /// The scene as last written to its recovery file.
    autosaved: Option<Snapshot>,
    /// Since when the scene has had changes the recovery file doesn't.
    unsaved_since: Option<Instant>,
    /// When to hot reload: pushed back by every file event, so a reload waits for the writer to finish.
    reload_due: Option<Instant>,
    reload_retries: u32,
    ignore_changes_until: Option<Instant>,
    applied_transforms: TransformSnapshot,
    applied_entities: usize,
    applied_deleted: Vec<usize>,
    /// Entity of each `SceneLight::id`.
    light_entities: HashMap<u64, EntityKey>,
    /// Entities of each `EditorObject::id`.
    object_entities: HashMap<u64, ObjectEntities>,
    /// Model files by absolute path.
    models: HashMap<String, ModelAsset>,
    /// Project as last written to (or read from) `project.json`.
    saved_project: Option<Project>,
    /// Unsaved project contents and when they were last changed.
    pending_project: Option<(Project, Instant)>,
    asset_worker: AssetWorker,
    build_events: Option<mpsc::Receiver<BuildEvent>>,
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
            reloading: false,
            restoring: false,
            autosaved: None,
            unsaved_since: None,
            reload_due: None,
            reload_retries: 0,
            ignore_changes_until: None,
            applied_transforms: Vec::new(),
            applied_entities: 0,
            applied_deleted: Vec::new(),
            light_entities: HashMap::new(),
            object_entities: HashMap::new(),
            models: HashMap::new(),
            saved_project: None,
            pending_project: None,
            asset_worker: AssetWorker::spawn(),
            build_events: None,
        }
    }

    fn reset_world(&mut self, engine: &mut Engine) -> EntityKey {
        engine.world().clear();

        let camera_key = engine.world().add_entity(Camera {
            up: (0.0, 1.0, 0.0).into(),
            eye: (0.0, 5.0, 15.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            aspect: None,
            fovy: EDITOR_FOV_Y_DEG.to_radians(),
            near: EDITOR_NEAR,
            far: EDITOR_FAR,
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
        self.handle_prompt_answer(engine);
        self.poll_load(engine);
        self.autosave();
        self.check_file_changes(engine);
        self.sync_lights(engine);
        self.sync_objects(engine);
        self.apply_edits(engine);
        self.sync_project();
        self.handle_build();
        self.sync_assets();

        if let Some(scene) = &mut self.scene
            && !scene.is_complete()
        {
            let mut world = engine.world();
            scene.load_batch(&mut world, 10);
            // The renderer draws from the first camera, which after `World::clear` can be a scene
            // camera rather than the editor's. Camera nodes are edited through `node_edits` instead.
            for &key in scene.node_cameras().values() {
                world.remove_entity(key);
            }
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

    fn set_project(&mut self, s: &mut EditorState, project: Project, dir: PathBuf) {
        s.camera_settings = project.camera.clone();
        s.project_scenes = Project::list_scenes(&dir);
        s.assets.folder.clear();
        s.assets.selected = None;
        self.asset_worker.rescan(&mut s.assets, &dir);
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
        // The default scene and build settings decide what a build ships.
        self.asset_worker.rescan(&mut s.assets, &dir);
        self.saved_project = Some(project);
    }

    /// Re-reads the project's files after something on disk changed.
    fn refresh_project_files(&self, s: &mut EditorState) {
        if let Some(dir) = s.project_dir.clone() {
            s.project_scenes = Project::list_scenes(&dir);
            self.asset_worker.rescan(&mut s.assets, &dir);
        }
    }

    fn record_recent(s: &mut EditorState, name: &str, dir: &Path) {
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
                self.refresh_project_files(&mut s);
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
        let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        s.status = match result {
            Ok(()) => {
                // Our own write would otherwise trigger a hot reload that drops lights and selection.
                self.ignore_changes_until = Some(Instant::now() + Duration::from_millis(500));
                self.refresh_project_files(&mut s);
                s.mark_saved();
                remove_recovery(&path);
                self.autosaved = None;
                self.unsaved_since = None;
                if s.exit_after_save {
                    s.close_guard.exit();
                }
                format!("Saved {name}")
            }
            Err(e) => {
                s.exit_after_save = false;
                format!("Save failed: {e:#}")
            }
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
            self.refresh_project_files(&mut s);
            s.status = match result {
                Ok(()) => format!("Exported to {}", path.display()),
                Err(e) => format!("Export failed: {e}"),
            };
        }
    }

    fn start_load(&mut self, engine: &mut Engine, path: PathBuf) {
        self.clear_scene(engine, path.clone());
        self.reload_due = None;
        self.reloading = false;
        self.restoring = false;
        self.pending_gltf = Some(engine.assets().load_async::<GLTFAsset>(path.to_str().unwrap_or("")));
    }

    /// Opens `scene` with the contents of its recovery file: saving writes `scene`.
    fn start_restore(&mut self, engine: &mut Engine, scene: PathBuf, recovery: &Path) {
        self.clear_scene(engine, scene);
        self.reload_due = None;
        self.reloading = false;
        self.restoring = true;
        self.pending_gltf = Some(engine.assets().load_async::<GLTFAsset>(recovery.to_str().unwrap_or("")));
    }

    /// Loads the open scene again in the background, keeping the current one until it succeeds.
    fn start_reload(&mut self, engine: &mut Engine) {
        let Some(path) = self.state.lock().expect("editor state mutex poisoned").file_path.clone() else { return };
        self.reloading = true;
        self.pending_gltf = Some(engine.assets().load_async::<GLTFAsset>(path.to_str().unwrap_or("")));
    }

    fn clear_scene(&mut self, engine: &mut Engine, path: PathBuf) {
        self.watcher = None;
        self.watch_rx = None;
        let camera_key = self.reset_world(engine);
        self.scene = None;
        self.applied_transforms.clear();
        self.applied_entities = 0;
        self.applied_deleted.clear();
        self.light_entities.clear();
        self.object_entities.clear();

        let mut s = self.state.lock().expect("editor state mutex poisoned");
        s.editor_camera_key = Some(camera_key);
        s.lights.clear();
        s.objects.clear();
        s.entity_objects.clear();
        s.deleted_nodes.clear();
        s.history.reset();
        s.file_path = Some(path);
        s.document = None;
        s.scene_nodes.clear();
        s.node_parents.clear();
        s.entity_nodes.clear();
        s.clear_node_edits();
        s.selected = None;
        s.camera_preview_node = None;
        s.main_camera = None;
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
        let reloading = std::mem::take(&mut self.reloading);
        let restoring = std::mem::take(&mut self.restoring);

        match result {
            Ok(asset) => {
                if reloading {
                    let path = self.state.lock().expect("editor state mutex poisoned").file_path.clone();
                    let view = self.editor_camera_pose(engine);
                    self.clear_scene(engine, path.unwrap_or_default());
                    self.restore_editor_camera_pose(engine, view);
                }
                self.reload_retries = 0;
                let doc = asset.document.clone();
                let nodes = build_scene_nodes(&doc);
                let (lights, lights_error) = match engine::gltf::lights::read_lights(&doc) {
                    Ok(lights) => (lights, None),
                    Err(e) => (Vec::new(), Some(e)),
                };
                let (objects, objects_error) = match engine::gltf::objects::read_objects(&doc) {
                    Ok(objects) => (objects, None),
                    Err(e) => (Vec::new(), Some(e)),
                };
                self.scene = Some(GltfScene::new(Arc::clone(&asset)));
                let scene_path = {
                    let mut s = self.state.lock().expect("editor state mutex poisoned");
                    let path_display = s.file_path.as_ref()
                        .map(|p| p.file_name().unwrap_or_default().to_string_lossy().into_owned())
                        .unwrap_or_default();
                    s.document = Some(doc.clone());
                    s.main_camera = doc
                        .default_scene()
                        .or_else(|| doc.scenes().next())
                        .and_then(|scene| engine::gltf::camera::main_camera(&scene));
                    s.scene_nodes = nodes;
                    s.node_parents = doc
                        .nodes()
                        .flat_map(|parent| parent.children().map(move |child| (child.index(), parent.index())))
                        .collect();
                    s.lights = lights
                        .into_iter()
                        .map(|(node, data)| SceneLight { id: s.new_id(), data, node })
                        .collect();
                    let scene_file = s.file_path.clone().unwrap_or_default();
                    s.objects = objects
                        .into_iter()
                        .map(|mut data| {
                            if let ObjectKind::Model { uri } = &mut data.kind {
                                *uri = model_path(&scene_file, uri);
                            }
                            EditorObject { id: s.new_id(), data }
                        })
                        .collect();
                    s.history.reset();
                    self.autosaved = None;
                    self.unsaved_since = None;
                    s.status = match (lights_error, objects_error) {
                        (Some(e), _) => format!("Loaded: {path_display} (lights skipped: {e:#})"),
                        (None, Some(e)) => format!("Loaded: {path_display} (objects skipped: {e:#})"),
                        (None, None) => format!("Loaded: {path_display}"),
                    };
                    if restoring {
                        s.mark_unsaved();
                        s.status = format!("Restored unsaved changes to {path_display} from its auto-save");
                    } else {
                        s.mark_saved();
                        let recovery = recovery_path(&scene_file);
                        let modified = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
                        if let (Some(saved), Some(autosaved)) = (modified(&scene_file), modified(&recovery))
                            && autosaved > saved
                        {
                            s.prompt = Some(Prompt::Recover(recovery));
                        }
                    }
                    s.file_path.clone()
                };
                if let Some(path) = scene_path {
                    self.setup_watcher(&path, Some(&doc));
                }
            }
            Err(e) if reloading => {
                let retry = self.reload_retries < RELOAD_MAX_RETRIES;
                if retry {
                    self.reload_retries += 1;
                    self.reload_due = Some(Instant::now() + RELOAD_RETRY_DELAY);
                }
                let mut s = self.state.lock().expect("editor state mutex poisoned");
                s.status = format!(
                    "Reload failed, keeping the previous version{}: {e}",
                    if retry { " (retrying)" } else { "" },
                );
            }
            Err(e) => {
                let path = {
                    let mut s = self.state.lock().expect("editor state mutex poisoned");
                    s.status = format!("Load error: {e}");
                    s.file_path.clone()
                };
                // Without a document only the scene file can be watched; fixing it reloads.
                if let Some(path) = path {
                    self.setup_watcher(&path, None);
                }
            }
        }
    }

    /// Spawns, updates and despawns light entities to match `lights`, every tick, so inspector
    /// edits, additions, deletions and undo all show at once.
    fn sync_lights(&mut self, engine: &mut Engine) {
        let lights = self.state.lock().expect("editor state mutex poisoned").lights.clone();
        let live: HashSet<u64> = lights.iter().map(|l| l.id).collect();
        let mut world = engine.world();
        self.light_entities.retain(|id, key| {
            let keep = live.contains(id);
            if !keep {
                world.remove_entity(*key);
            }
            keep
        });
        for light in &lights {
            let key = *self.light_entities.entry(light.id).or_insert_with(|| light.data.spawn(&mut world));
            light.data.apply(&mut world, key);
        }
    }

    /// Keeps each object's entities in line with `objects`: spawns shapes, streams placed models
    /// in, writes transforms and shape materials when they change, and despawns what was removed.
    fn sync_objects(&mut self, engine: &mut Engine) {
        let objects = self.state.lock().expect("editor state mutex poisoned").objects.clone();

        for object in &objects {
            if let ObjectKind::Model { uri } = &object.data.kind
                && !self.models.contains_key(uri)
            {
                let handle = engine.assets().load_async::<GLTFAsset>(uri);
                self.models.insert(uri.clone(), ModelAsset::Loading(handle));
            }
        }
        let mut failed = Vec::new();
        for (uri, model) in &mut self.models {
            if let ModelAsset::Loading(handle) = model
                && let Some(result) = handle.try_get()
            {
                *model = match result {
                    Ok(asset) => ModelAsset::Ready(asset),
                    Err(e) => {
                        failed.push(format!("{uri}: {e}"));
                        ModelAsset::Failed
                    }
                };
            }
        }
        if let Some(error) = failed.first() {
            self.state.lock().expect("editor state mutex poisoned").status = format!("Model failed to load: {error}");
        }

        let mut world = engine.world();
        let live: HashSet<u64> = objects.iter().map(|o| o.id).collect();
        let before = self.object_entities.values().map(|e| e.meshes.len()).sum::<usize>();
        let mut changed = false;
        self.object_entities.retain(|id, entities| {
            let keep = live.contains(id);
            if !keep {
                entities.despawn(&mut world);
                changed = true;
            }
            keep
        });

        for object in &objects {
            let source = ObjectSource::of(&object.data.kind);
            let entities = self.object_entities.entry(object.id).or_insert_with(|| ObjectEntities::new(source.clone()));
            if entities.source != source {
                entities.despawn(&mut world);
                *entities = ObjectEntities::new(source);
                changed = true;
            }

            let spawned = entities.meshes.len();
            match &object.data.kind {
                ObjectKind::Shape { shape, material } if entities.meshes.is_empty() => {
                    let key = world.add_entity(shape.mesh(material));
                    entities.meshes.push((key, Mat4::identity()));
                }
                ObjectKind::Model { uri } => {
                    if entities.scene.is_none()
                        && let Some(ModelAsset::Ready(asset)) = self.models.get(uri)
                    {
                        entities.scene = Some(GltfScene::new(Arc::clone(asset)));
                    }
                    if let Some(scene) = &mut entities.scene
                        && !scene.is_complete()
                    {
                        scene.load_batch(&mut world, 10);
                        // Placed models never bring their own cameras along.
                        for &key in scene.node_cameras().values() {
                            world.remove_entity(key);
                        }
                        let known: HashSet<EntityKey> = entities.meshes.iter().map(|(k, _)| *k).collect();
                        for &key in scene.node_entities().values().flatten() {
                            if !known.contains(&key)
                                && let Some(mesh) = world.get_entity::<Mesh>(key)
                            {
                                entities.meshes.push((key, mesh.transform));
                            }
                        }
                    }
                }
                _ => {}
            }
            if entities.meshes.len() != spawned {
                changed = true;
                entities.applied = None;
            }

            if entities.applied.as_ref() == Some(object) {
                continue;
            }
            let matrix = object.data.matrix();
            let material = match &object.data.kind {
                ObjectKind::Shape { material, .. } => Some(material.factors()),
                _ => None,
            };
            for &(key, base) in &entities.meshes {
                if let Some(mesh) = world.get_entity_mut::<Mesh>(key) {
                    mesh.transform = matrix * base;
                    if let Some(material) = material {
                        mesh.material = material;
                    }
                }
            }
            entities.applied = Some(object.clone());
        }
        drop(world);

        let after = self.object_entities.values().map(|e| e.meshes.len()).sum::<usize>();
        if changed || before != after {
            let map = self
                .object_entities
                .iter()
                .flat_map(|(&id, e)| e.meshes.iter().map(move |&(key, _)| (key, id)))
                .collect();
            self.state.lock().expect("editor state mutex poisoned").entity_objects = map;
        }
    }

    fn apply_edits(&self, engine: &mut Engine) {
        let s = self.state.lock().expect("editor state mutex poisoned");
        let Some(cam_key) = s.editor_camera_key else { return };
        let preview = s.camera_preview_node.and_then(|idx| {
            let camera = s.node_edits.get(&idx)?.camera.clone()?;
            Some((s.node_world(idx), camera))
        });
        let project_exposure = s.camera_settings.exposure();
        drop(s);

        let mut world = engine.world();
        let Some(cam) = world.get_entity_mut::<Camera>(cam_key) else { return };
        match preview {
            Some((node_world, edit)) => {
                cam.eye = node_world.extract_translation();
                cam.target = cam.eye + node_world.transform_vec3(-Vec3::unit_z()).normalized();
                cam.up = node_world.transform_vec3(Vec3::unit_y()).normalized();
                cam.fovy = edit.fov_y_deg.to_radians();
                cam.near = edit.near;
                cam.far = edit.far.unwrap_or(INFINITE_FAR);
                cam.aspect = edit.aspect;
                cam.exposure = edit.physical.map_or(project_exposure, |p| p.exposure());
            }
            None => {
                cam.fovy = EDITOR_FOV_Y_DEG.to_radians();
                cam.near = EDITOR_NEAR;
                cam.far = EDITOR_FAR;
                cam.aspect = None;
            }
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

        let mut deleted: Vec<usize> = s.deleted_nodes.iter().copied().collect();
        deleted.sort_unstable();
        if snapshot == self.applied_transforms && loaded == self.applied_entities && deleted == self.applied_deleted {
            return;
        }

        if loaded != self.applied_entities {
            s.entity_nodes = scene
                .node_entities()
                .iter()
                .flat_map(|(&node, keys)| keys.iter().map(move |&key| (key, node)))
                .collect();
        }

        // Deleted nodes collapse to a point until saved (meshes have no visibility flag), which
        // keeps undo a matter of restoring their transform.
        let worlds: HashMap<usize, ultraviolet::Mat4> = scene
            .node_entities()
            .keys()
            .map(|&node| {
                let world = s.node_world(node);
                let world = if s.is_node_deleted(node) { world * Mat4::from_scale(0.0) } else { world };
                (node, world)
            })
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
        self.applied_deleted = deleted;
    }

    fn sync_assets(&mut self) {
        let mut s = self.state.lock().expect("editor state mutex poisoned");
        if std::mem::take(&mut s.assets.refresh_requested) {
            self.refresh_project_files(&mut s);
        }
        let dir = s.project_dir.clone();
        self.asset_worker.sync(&mut s.assets, dir.as_deref());
    }

    /// Runs builds on a background thread, streaming their output into the build log.
    fn handle_build(&mut self) {
        let state = Arc::clone(&self.state);
        let mut s = state.lock().expect("editor state mutex poisoned");

        if let Some(rx) = &self.build_events {
            let mut finished = None;
            for event in rx.try_iter() {
                match event {
                    BuildEvent::Line(line) => s.build.log.push(line),
                    BuildEvent::Finished(result) => finished = Some(result),
                }
            }
            let overflow = s.build.log.len().saturating_sub(BUILD_LOG_LINES);
            s.build.log.drain(..overflow);

            if let Some(result) = finished {
                let kind = s.build.running.take().unwrap_or(BuildKind::Development);
                self.build_events = None;
                match result {
                    Ok(output) => {
                        s.status = format!(
                            "{kind} build finished: {} ({} assets, {})",
                            output.executable.display(),
                            output.assets,
                            build_tool::format_bytes(output.stored_bytes)
                        );
                        let line = s.status.clone();
                        s.build.log.push(line);
                        s.build.last_output = Some(output.executable);
                    }
                    Err(e) => {
                        s.status = format!("{kind} build failed: {e}");
                        let line = s.status.clone();
                        s.build.log.push(line);
                        s.build.show_log = true;
                    }
                }
                self.refresh_project_files(&mut s);
            }
        }

        let Some(kind) = s.build.requested.take() else { return };
        if s.build.running.is_some() {
            return;
        }
        let Some(dir) = s.project_dir.clone() else { return };
        if let Some((project, _)) = self.pending_project.take() {
            if let Err(e) = project.save(&dir) {
                s.status = format!("Saving project settings failed: {e}");
                return;
            }
            self.saved_project = Some(project);
        }

        let (tx, rx) = mpsc::channel();
        let options = BuildOptions::new(dir, kind);
        let spawned = std::thread::Builder::new().name("build".into()).spawn(move || {
            let lines = tx.clone();
            let result = build_tool::build(&options, &mut |line| {
                let _ = lines.send(BuildEvent::Line(line));
            });
            let _ = tx.send(BuildEvent::Finished(result.map_err(|e| e.to_string())));
        });
        if let Err(e) = spawned {
            s.status = format!("Starting build failed: {e}");
            return;
        }

        self.build_events = Some(rx);
        s.build.running = Some(kind);
        s.build.log.clear();
        s.build.last_output = None;
        s.build.show_log = true;
        s.status = format!("Building {kind}… (uses the saved scene files)");
    }

    /// Watches the scene file and, given its document, every external buffer and image.
    fn setup_watcher(&mut self, scene_path: &Path, doc: Option<&gltf_api::Document>) {
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
        if let Some(doc) = doc {
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
                .filter(|res| match res {
                    // Auto-save's own writes aren't changes to the scene.
                    Ok(event @ notify::Event { kind: EventKind::Modify(_) | EventKind::Create(_), .. }) => {
                        !event.paths.iter().all(|p| is_recovery_path(p))
                    }
                    _ => false,
                })
                .count()
                > 0,
        };

        let now = Instant::now();
        if changed && self.ignore_changes_until.is_none_or(|t| now >= t) {
            self.reload_due = Some(now + RELOAD_QUIET);
            self.reload_retries = 0;
        }

        if self.pending_gltf.is_none() && self.reload_due.is_some_and(|t| now >= t) {
            self.reload_due = None;
            // Reloading would drop the unsaved changes, so ask first.
            let mut s = self.state.lock().expect("editor state mutex poisoned");
            if s.is_dirty() {
                s.prompt.get_or_insert(Prompt::Reload);
            } else {
                drop(s);
                self.start_reload(engine);
            }
        }
    }

    /// Acts on the answer to `EditorState::prompt`.
    fn handle_prompt_answer(&mut self, engine: &mut Engine) {
        let (prompt, answer, file) = {
            let mut s = self.state.lock().expect("editor state mutex poisoned");
            let Some(answer) = s.prompt_answer.take() else { return };
            (s.prompt.take(), answer, s.file_path.clone())
        };
        match (prompt, answer) {
            (Some(Prompt::Exit), PromptAnswer::Yes) => {
                let mut s = self.state.lock().expect("editor state mutex poisoned");
                s.save_requested = true;
                s.exit_after_save = true;
            }
            (Some(Prompt::Exit), PromptAnswer::No) => {
                if let Some(file) = &file {
                    remove_recovery(file);
                }
                self.state.lock().expect("editor state mutex poisoned").close_guard.exit();
            }
            (Some(Prompt::Reload), PromptAnswer::Yes) => self.start_reload(engine),
            (Some(Prompt::Recover(recovery)), PromptAnswer::Yes) => {
                if let Some(file) = file {
                    self.start_restore(engine, file, &recovery);
                }
            }
            (Some(Prompt::Recover(recovery)), PromptAnswer::No) => {
                let _ = std::fs::remove_file(recovery);
            }
            _ => {}
        }
    }

    /// Writes the recovery copy once the scene has had changes it doesn't for `AUTOSAVE_DELAY`.
    /// Nothing is written while the scene matches its file or the last recovery copy.
    fn autosave(&mut self) {
        let mut s = self.state.lock().expect("editor state mutex poisoned");
        let Some(path) = s.file_path.clone() else { return };
        let snapshot = s.snapshot();
        if !s.is_dirty() || self.autosaved.as_ref() == Some(&snapshot) {
            self.unsaved_since = None;
            return;
        }
        let since = *self.unsaved_since.get_or_insert_with(Instant::now);
        if since.elapsed() < AUTOSAVE_DELAY {
            return;
        }
        let recovery = recovery_path(&path);
        match crate::export::write_gltf(&s, &recovery) {
            Ok(()) => {
                s.status = format!("Auto-saved unsaved changes to {}", recovery.file_name().unwrap_or_default().to_string_lossy());
                self.autosaved = Some(snapshot);
            }
            Err(e) => s.status = format!("Auto-save failed: {e:#}"),
        }
        self.unsaved_since = None;
    }

    fn editor_camera_pose(&self, engine: &mut Engine) -> Option<(Vec3, Vec3)> {
        let key = self.state.lock().expect("editor state mutex poisoned").editor_camera_key?;
        engine.world().get_entity::<Camera>(key).map(|cam| (cam.eye, cam.target))
    }

    fn restore_editor_camera_pose(&self, engine: &mut Engine, pose: Option<(Vec3, Vec3)>) {
        let Some((eye, target)) = pose else { return };
        let Some(key) = self.state.lock().expect("editor state mutex poisoned").editor_camera_key else { return };
        if let Some(cam) = engine.world().get_entity_mut::<Camera>(key) {
            cam.eye = eye;
            cam.target = target;
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
        } else if node.light().is_some() {
            NodeKind::Light
        } else {
            NodeKind::Empty
        },
        children: node.children().map(|c| node_snapshot(&c)).collect(),
    }
}

/// A model URI from a scene file as the absolute path the editor loads it from.
fn model_path(scene_file: &Path, uri: &str) -> String {
    let uri = engine::gltf::external_uri(uri).map(|u| u.into_owned()).unwrap_or_else(|| uri.to_string());
    engine::assets::path::resolve_relative(scene_file, &uri).to_string_lossy().into_owned()
}

/// Deletes the scene's recovery copy, if any: it holds nothing the scene file doesn't.
fn remove_recovery(scene: &Path) {
    let _ = std::fs::remove_file(recovery_path(scene));
}
