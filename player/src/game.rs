use std::path::Path;
use std::sync::Arc;

use engine::assets::AssetLoadHandle;
use engine::gltf::camera::PhysicalCamera;
use engine::gltf::objects::{self, ObjectKind, PLAYER_EYE_HEIGHT, SceneObject};
use engine::gltf::{GLTFAsset, GltfScene, lights};
use engine::player_controller::PlayerController;
use engine::plugin::{EnginePluginError, Plugin};
use engine::rendering::camera::Camera;
use engine::rendering::camera_controller::{CameraController, CameraMode};
use engine::{Engine, EntityKey};
use ultraviolet::{Mat4, Vec3};
#[cfg(feature = "dev")]
use engine::{InputEvent, Key};
use project::Project;

#[cfg(feature = "dev")]
use crate::dev_tools::{FreeCamera, SharedDevState};

/// Vertical field of view of the player-start camera.
const PLAYER_FOV_Y_DEG: f32 = 60.0;

/// A model placed by one of the scene's objects: loading, then streaming in under `root`.
struct PlacedModel {
    path: String,
    root: Mat4,
    pending: Option<AssetLoadHandle<GLTFAsset>>,
    scene: Option<GltfScene>,
}

/// First person at the scene's player start: WASD walks at the start's eye height (no collision
/// or gravity yet), LMB drag looks around.
struct Walker {
    camera: EntityKey,
    look: CameraController,
    movement: PlayerController,
    eye_height: f32,
}

impl Walker {
    fn new(camera: EntityKey, move_speed: f32, eye_height: f32) -> Self {
        let mut movement = PlayerController::new(move_speed);
        movement.attach(camera);
        Self { camera, look: CameraController::new(camera, CameraMode::FirstPerson), movement, eye_height }
    }

    fn tick(&mut self, engine: &mut Engine, delta: f32) {
        self.movement.tick::<Camera>(engine, delta);
        let ui_wants_pointer = engine.ui_wants_pointer();
        let (input, mut world) = engine.input_and_world();
        self.look.tick(input, &mut world, delta, ui_wants_pointer);
        if let Some(camera) = world.get_entity_mut::<Camera>(self.camera) {
            let climb = self.eye_height - camera.eye.y;
            camera.eye.y += climb;
            camera.target.y += climb;
        }
    }
}

/// Plays the project's default scene: streams it in, spawns its lights and objects, and makes
/// sure a camera exists (a walker at the player start, if the scene has one).
pub struct Game {
    project: Project,
    scene_path: String,
    pending: Option<AssetLoadHandle<GLTFAsset>>,
    scene: Option<GltfScene>,
    models: Vec<PlacedModel>,
    player_start: Option<SceneObject>,
    walker: Option<Walker>,
    camera_ready: bool,
    #[cfg(feature = "dev")]
    dev: SharedDevState,
    #[cfg(feature = "dev")]
    free_camera: Option<FreeCamera>,
}

impl Game {
    pub fn new(project: Project, scene_path: String, #[cfg(feature = "dev")] dev: SharedDevState) -> Self {
        Self {
            project,
            scene_path,
            pending: None,
            scene: None,
            models: Vec::new(),
            player_start: None,
            walker: None,
            camera_ready: false,
            #[cfg(feature = "dev")]
            dev,
            #[cfg(feature = "dev")]
            free_camera: None,
        }
    }

    fn poll_scene(&mut self, engine: &mut Engine) {
        let Some(result) = self.pending.as_ref().and_then(AssetLoadHandle::try_get) else { return };
        self.pending = None;

        match result {
            Ok(asset) => {
                match lights::read_lights(&asset.document) {
                    Ok(scene_lights) => {
                        let mut world = engine.world();
                        for (_, light) in scene_lights {
                            light.spawn(&mut world);
                        }
                    }
                    Err(e) => log::error!("{}: invalid lights: {e}", self.scene_path),
                }
                match objects::read_objects(&asset.document) {
                    Ok(scene_objects) => self.spawn_objects(engine, scene_objects),
                    Err(e) => log::error!("{}: invalid objects: {e}", self.scene_path),
                }
                self.scene = Some(GltfScene::new(Arc::clone(&asset)));
                log::info!("Loaded {}", self.scene_path);
            }
            Err(e) => log::error!("Loading {} failed: {e}", self.scene_path),
        }
        self.set_status();
    }

    /// Shapes go straight into the world, models start loading (paths relative to the scene), and
    /// the first player start is kept for `ensure_camera`.
    fn spawn_objects(&mut self, engine: &mut Engine, scene_objects: Vec<SceneObject>) {
        for object in scene_objects {
            match &object.kind {
                ObjectKind::Shape { shape, material } => {
                    engine.world().add_entity(shape.mesh(material).with_transform(object.matrix()));
                }
                ObjectKind::Model { uri } => {
                    let uri = engine::gltf::external_uri(uri).map_or_else(|| uri.clone(), |u| u.into_owned());
                    let resolved = engine::assets::path::resolve_relative(Path::new(&self.scene_path), &uri);
                    let path = resolved.to_string_lossy().replace('\\', "/");
                    let pending = Some(engine.assets().load_async::<GLTFAsset>(&path));
                    self.models.push(PlacedModel { path, root: object.matrix(), pending, scene: None });
                }
                ObjectKind::PlayerStart => {
                    if self.player_start.is_none() {
                        self.player_start = Some(object.clone());
                    }
                }
            }
        }
    }

    /// Streams placed models in, a few nodes per tick like the scene. Their cameras are dropped.
    fn stream_models(&mut self, engine: &mut Engine) {
        for model in &mut self.models {
            if let Some(result) = model.pending.as_ref().and_then(AssetLoadHandle::try_get) {
                model.pending = None;
                match result {
                    Ok(asset) => model.scene = Some(GltfScene::new(asset).with_root(model.root)),
                    Err(e) => log::error!("Loading model {} failed: {e}", model.path),
                }
            }
            if let Some(scene) = &mut model.scene
                && !scene.is_complete()
            {
                let mut world = engine.world();
                scene.load_batch(&mut world, 10);
                for &key in scene.node_cameras().values() {
                    world.remove_entity(key);
                }
            }
        }
    }

    /// Replaces every camera with one at the player start's eye height, facing its heading, and
    /// hands it to a `Walker`.
    fn spawn_walker(&mut self, engine: &mut Engine, start: &SceneObject) {
        let mut world = engine.world();
        let cameras: Vec<EntityKey> = world.get_entity_handles::<Camera>().iter().map(|h| h.id).collect();
        for key in cameras {
            world.remove_entity(key);
        }
        let eye = Vec3::from(start.translation) + Vec3::unit_y() * PLAYER_EYE_HEIGHT;
        let forward = start.matrix().transform_vec3(-Vec3::unit_z());
        let heading = Vec3::new(forward.x, 0.0, forward.z);
        let heading = if heading.mag_sq() > 1e-6 { heading.normalized() } else { -Vec3::unit_z() };
        let key = world.add_entity(Camera {
            up: Vec3::unit_y(),
            eye,
            target: eye + heading,
            aspect: None,
            fovy: PLAYER_FOV_Y_DEG.to_radians(),
            near: 0.01,
            far: 10000.0,
            exposure: self.project.camera.exposure(),
        });
        drop(world);
        self.walker = Some(Walker::new(key, self.project.camera.move_speed, eye.y));
    }

    /// After the first batch the scene's own cameras exist. Keeps only the scene's main camera if
    /// it marks one (the renderer draws from the first camera), gives cameras without physical
    /// settings the project's exposure, and falls back to a default camera.
    fn ensure_camera(&mut self, engine: &mut Engine) {
        let exposure = self.project.camera.exposure();
        let mut world = engine.world();
        if let Some(scene) = &self.scene {
            let document = scene.document();
            if let Some(main) = scene.main_camera() {
                for &key in scene.node_cameras().values().filter(|&&key| key != main) {
                    world.remove_entity(key);
                }
            }
            for (&node, &key) in scene.node_cameras() {
                let physical = document.nodes().nth(node).and_then(|n| n.camera()).and_then(|c| PhysicalCamera::read(&c));
                if physical.is_none()
                    && let Some(camera) = world.get_entity_mut::<Camera>(key)
                {
                    camera.exposure = exposure;
                }
            }
        }
        if world.get_entities::<Camera>().is_empty() {
            log::info!("{} has no camera, using a default one", self.scene_path);
            world.add_entity(Camera {
                up: (0.0, 1.0, 0.0).into(),
                eye: (0.0, 2.0, 8.0).into(),
                target: (0.0, 0.0, 0.0).into(),
                aspect: None,
                fovy: 45.0_f32.to_radians(),
                near: 0.01,
                far: 10000.0,
                exposure,
            });
        }
        drop(world);
        if let Some(start) = self.player_start.clone() {
            self.spawn_walker(engine, &start);
        }
        self.camera_ready = true;
    }

    #[cfg(feature = "dev")]
    fn set_status(&self) {
        let status = match (&self.pending, &self.scene) {
            (Some(_), _) => format!("loading {}", self.scene_path),
            (None, Some(scene)) if scene.is_complete() => format!("{} loaded", self.scene_path),
            (None, Some(_)) => format!("streaming {}", self.scene_path),
            (None, None) => format!("{} failed to load", self.scene_path),
        };
        self.dev.lock().expect("dev state poisoned").scene_status = status;
    }

    #[cfg(not(feature = "dev"))]
    fn set_status(&self) {}

    /// Handles dev hotkeys, drives the free camera and leaves no input queued.
    #[cfg(feature = "dev")]
    fn handle_input(&mut self, engine: &mut Engine, delta: f32) {
        let events: Vec<InputEvent> = std::iter::from_fn(|| engine.input().pop()).collect();
        let keyboard_free = !engine.ui_wants_keyboard();
        let free_camera_wanted = {
            let mut dev = self.dev.lock().expect("dev state poisoned");
            for event in &events {
                match event {
                    InputEvent::KeyPressed { key: Key::F1, .. } if keyboard_free => dev.show_tools = !dev.show_tools,
                    InputEvent::KeyPressed { key: Key::F2, .. } if keyboard_free => dev.free_camera = !dev.free_camera,
                    _ => {}
                }
            }
            dev.free_camera
        };

        match (free_camera_wanted, self.free_camera.is_some()) {
            (true, false) => {
                let key = engine.world().get_entity_handles::<Camera>().first().map(|h| h.id);
                self.free_camera = key.map(|key| FreeCamera::new(key, self.project.camera.move_speed));
            }
            (false, true) => self.free_camera = None,
            _ => {}
        }

        if let Some(free_camera) = &mut self.free_camera {
            for event in events {
                engine.input().push(event);
            }
            free_camera.tick(engine, delta);
        } else if let Some(walker) = &mut self.walker {
            for event in events {
                engine.input().push(event);
            }
            walker.tick(engine, delta);
        }
        while engine.input().pop().is_some() {}
    }

    #[cfg(not(feature = "dev"))]
    fn handle_input(&mut self, engine: &mut Engine, delta: f32) {
        if let Some(walker) = &mut self.walker {
            walker.tick(engine, delta);
        }
        while engine.input().pop().is_some() {}
    }
}

impl Plugin for Game {
    fn setup(&mut self, engine: &mut Engine) -> Result<(), EnginePluginError> {
        self.pending = Some(engine.assets().load_async::<GLTFAsset>(&self.scene_path));
        self.set_status();
        Ok(())
    }

    fn tick(&mut self, engine: &mut Engine, delta: f32) {
        self.poll_scene(engine);
        self.stream_models(engine);

        if let Some(scene) = &mut self.scene
            && !scene.is_complete()
        {
            scene.load_batch(&mut engine.world(), 10);
            if !self.camera_ready {
                self.ensure_camera(engine);
            }
            self.set_status();
        }

        self.handle_input(engine, delta);
    }
}
