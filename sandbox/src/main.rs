use application::{Application, ApplicationError};
use engine::{
    Engine,
    input::{InputEvent, KeyCodes},
    plugin::EnginePlugin,
    world::DefaultKey,
};
use log::LevelFilter;
use rendering::{
    camera::Camera,
    pipeline::{RenderingPipelineDescriptor, StageShaderDescriptor},
    shader::{ShaderAsset, ShaderAssetResolver},
};

use gltf::{GLTFAsset, GLTFAssetResolver, GLTFEnginePlugin};
use window::WindowedApplication;

struct MyApplication {
    default_camera_id: DefaultKey,
    camera_speed: f32,
}

impl Application for MyApplication {
    fn setup(engine: &mut Engine) -> Result<Self, ApplicationError> {
        let gltf_plugin = GLTFEnginePlugin.setup(engine)?;

        let gltf_path: &str = {
            if cfg!(target_os = "windows") {
                "F:\\Desktop\\projects\\glTF-Sample-Assets\\Models\\Box\\glTF-Binary\\Box.glb"
                // "F:\\Desktop\\projects\\glTF-Sample-Assets\\Models\\Duck\\glTF-Binary\\Duck.glb"
            } else {
                "/mnt/f/Desktop/projects/glTF-Sample-Assets/Models/Box/glTF-Binary/Box.glb"
            }
        };

        let scene_file = engine.asset_manager().load_asset::<GLTFAsset>(gltf_path)?;
        gltf_plugin.load_scene(&scene_file, engine.world()).unwrap();

        let default_camera_id = engine.world().add_entity(Camera {
            eye: (1.0, 0.0, 0.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: (0.0, 1.0, 0.0).into(),
            aspect: (4 / 3) as f32,
            field_of_view: 45.0,
            znear: 0.1,
            zfar: 100.0,
        });

        Ok(Self {
            default_camera_id,
            camera_speed: 0.2,
        })
    }

    // Run game logic
    async fn run(&mut self, engine: &mut Engine) -> Result<(), ApplicationError> {
        let input = engine.input().pop();

        let camera = engine
            .world()
            .get_entity_mut::<Camera>(self.default_camera_id)
            .unwrap();

        let forward = camera.target - camera.eye;
        let forward_norm = forward.normalized();

        if let Some(InputEvent::KeyPressed { key_code }) = input {
            match key_code {
                KeyCodes::W => {
                    camera.eye += forward_norm * self.camera_speed;
                }
                KeyCodes::S => {
                    camera.eye -= forward_norm * self.camera_speed;
                }
                _ => {}
            }
        }

        Ok(())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::builder()
        .filter_level(LevelFilter::Warn)
        .format_source_path(true)
        .format_level(true)
        .format_timestamp_millis()
        .init();

    let mut engine = Engine::default();

    let asset_manager = engine.asset_manager();
    asset_manager.add_resolver("glb", GLTFAssetResolver);
    asset_manager.add_resolver("wgsl", ShaderAssetResolver);

    let shader_path: &str = {
        if cfg!(target_os = "windows") {
            "F:\\Desktop\\projects\\blackbird\\triangle.wgsl"
        } else {
            "/home/matheus/workspace/blackbird/sandbox/shaders/triangle.wgsl"
        }
    };

    let example_shader = engine
        .asset_manager()
        .load_asset::<ShaderAsset>(shader_path)?;

    let mut application = WindowedApplication::<MyApplication>::create(
        engine,
        RenderingPipelineDescriptor {
            vertex: StageShaderDescriptor {
                entrypoint: "vs_main",
                asset: &example_shader,
            },
            fragment: StageShaderDescriptor {
                entrypoint: "fs_main",
                asset: &example_shader,
            },
        },
    )?;

    application.execute().await?;
    Ok(())
}
