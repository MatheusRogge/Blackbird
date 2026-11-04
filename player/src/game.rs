use std::sync::Arc;

use engine::assets::AssetLoadHandle;
use engine::gltf::{GLTFAsset, GltfScene, lights};
use engine::plugin::{EnginePluginError, Plugin};
use engine::Engine;
use engine::rendering::camera::Camera;
#[cfg(feature = "dev")]
use engine::{InputEvent, Key};
use project::Project;

#[cfg(feature = "dev")]
use crate::dev_tools::{FreeCamera, SharedDevState};

/// Plays the project's default scene: streams it in, spawns its lights and makes sure a camera
/// with the project's exposure exists.
pub struct Game {
    project: Project,
    scene_path: String,
    pending: Option<AssetLoadHandle<GLTFAsset>>,
    scene: Option<GltfScene>,
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
                self.scene = Some(GltfScene::new(Arc::clone(&asset)));
                log::info!("Loaded {}", self.scene_path);
            }
            Err(e) => log::error!("Loading {} failed: {e}", self.scene_path),
        }
        self.set_status();
    }

    /// After the first batch the scene's own cameras exist; fall back to a default one.
    fn ensure_camera(&mut self, engine: &mut Engine) {
        let exposure = self.project.camera.exposure();
        let mut world = engine.world();
        if world.get_entities::<Camera>().is_empty() {
            log::info!("{} has no camera, using a default one", self.scene_path);
            world.add_entity(Camera {
                up: (0.0, 1.0, 0.0).into(),
                eye: (0.0, 2.0, 8.0).into(),
                target: (0.0, 0.0, 0.0).into(),
                aspect: 16.0 / 9.0,
                fovy: 45.0_f32.to_radians(),
                near: 0.01,
                far: 10000.0,
                exposure,
            });
        }
        for camera in world.get_entities_mut::<Camera>() {
            camera.exposure = exposure;
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
        }
        while engine.input().pop().is_some() {}
    }

    #[cfg(not(feature = "dev"))]
    fn handle_input(&mut self, engine: &mut Engine, _delta: f32) {
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
