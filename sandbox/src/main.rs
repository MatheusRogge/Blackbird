use engine::{
    Engine, EntityKey,
    assets::AssetLoadHandle,
    gltf::{GLTFAsset, GltfScene},
    player_controller::PlayerController,
    plugin::{EnginePluginError, Plugin},
    rendering::{
        camera::Camera,
        camera_controller::{CameraController, CameraMode},
        light::PointLight,
        pbr::RenderGraphPBRBuilder,
        shader::ShaderAsset,
    },
    windowed::WindowedApplication,
};
use log::LevelFilter;

struct MyGame {
    scene: Option<GltfScene>,
    camera_controller: Option<CameraController>,
    player_controller: PlayerController,
    pending_gltf: Option<AssetLoadHandle<GLTFAsset>>,
    tick: u64,
}

impl Default for MyGame {
    fn default() -> Self {
        Self {
            scene: None,
            camera_controller: None,
            player_controller: PlayerController::new(20.5),
            pending_gltf: None,
            tick: 0,
        }
    }
}

impl Plugin for MyGame {
    fn setup(&mut self, engine: &mut Engine) -> Result<(), EnginePluginError> {
        let gltf_path: &str = if cfg!(target_os = "windows") {
            // "F:\\Desktop\\projects\\glTF-Sample-Assets\\Models\\BoxTextured\\gltf\\BoxTextured.gltf"
            "F:\\Desktop\\projects\\glTF-Sample-Assets\\Models\\Sponza\\glTF\\Sponza.gltf"
        } else {
            "/mnt/f/Desktop/projects/glTF-Sample-Assets/Models/BoxTextured/glTF/BoxTextured.gltf"
        };

        self.pending_gltf = Some(engine.assets().load_async::<GLTFAsset>(gltf_path));

        let camera_key: EntityKey = engine.world().add_entity(Camera {
            up: (0.0, 1.0, 0.0).into(),
            eye: (0.0, 40.0, 150.0).into(),
            target: (0.0, 40.0, 0.0).into(),
            aspect: 800.0 / 600.0,
            fovy: 45.0_f32.to_radians(),
            near: 0.01,
            far: 10000.0,
        });

        self.player_controller.attach(camera_key);
        self.camera_controller = Some(CameraController::new(camera_key, CameraMode::FirstPerson));

        let mut world = engine.world();
        world.add_entity(PointLight {
            position: (0.0, 100.0, 0.0).into(),
            color: (1.0, 1.0, 1.0).into(),
            intensity: 1000.0,
            radius: 500.0,
        });
        world.add_entity(PointLight {
            position: (-150.0, 50.0, 0.0).into(),
            color: (1.0, 0.5, 0.2).into(),
            intensity: 600.0,
            radius: 300.0,
        });

        Ok(())
    }

    fn tick(&mut self, engine: &mut Engine, delta: f32) {
        self.tick += 1;

        if self.tick.is_multiple_of(120) {
            log::info!("perf\n{}", engine.frame_stats());
        }

        self.player_controller.tick::<Camera>(engine, delta);

        if let Some(ctrl) = &mut self.camera_controller {
            let (input, mut world) = engine.input_and_world();
            ctrl.tick(input, &mut world, delta);
        }

        if let Some(handle) = &self.pending_gltf
            && let Some(result) = handle.try_get()
        {
            match result {
                Ok(asset) => {
                    self.scene = Some(GltfScene::new(asset));
                    self.pending_gltf = None;
                }
                Err(e) => log::error!("GLTF load failed: {}", e),
            }
        }

        if let Some(scene) = &mut self.scene {
            let mut world = engine.world();
            scene.load_batch(&mut world, 10);
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::builder()
        .filter_level(LevelFilter::Info)
        .format_source_path(true)
        .format_level(true)
        .format_timestamp_millis()
        .parse_default_env()
        .init();

    let gbuffer_shader = ShaderAsset::from_raw(include_str!("../shaders/gbuffer.wgsl"));
    let cluster_shader = ShaderAsset::from_raw(include_str!("../shaders/cluster_assignment.wgsl"));
    let lighting_shader = ShaderAsset::from_raw(include_str!("../shaders/lighting.wgsl"));
    let present_shader = ShaderAsset::from_raw(include_str!("../shaders/present.wgsl"));

    let render_graph = RenderGraphPBRBuilder::new(
        gbuffer_shader,
        cluster_shader,
        lighting_shader,
        present_shader,
    );

    WindowedApplication::new(render_graph)
        .add_plugin(MyGame::default())
        .run()?;

    Ok(())
}
