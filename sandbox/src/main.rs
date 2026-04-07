use std::sync::Arc;

use application::{Application, ApplicationError};
use engine::{
    Engine,
    input::{InputEvent, KeyCodes},
    plugin::EnginePlugin,
};
use gltf::{GLTFAsset, GLTFEnginePlugin};
use log::LevelFilter;
use rendering::{
    camera::Camera,
    mesh::Vec3,
    pbr::RenderGraphPBRBuilder,
    shader::{ShaderAsset, ShaderAssetResolver},
};
use window::WindowedApplication;

struct MyApplication {
    camera_speed: f32,
}

impl Application for MyApplication {
    fn setup(engine: &mut Engine) -> Result<Self, ApplicationError> {
        let gltf_plugin = GLTFEnginePlugin.setup(engine)?;

        let gltf_path: &str = {
            if cfg!(target_os = "windows") {
                "F:\\Desktop\\projects\\glTF-Sample-Assets\\Models\\Box\\glTF-Binary\\Box.glb"
                // "F:\\Desktop\\projects\\glTF-Sample-Assets\\Models\\Sponza\\gltf\\Sponza.gltf"
            } else {
                "/mnt/f/Desktop/projects/glTF-Sample-Assets/Models/Box/glTF-Binary/Box.glb"
            }
        };

        let scene_file = engine.asset_manager().load_asset::<GLTFAsset>(gltf_path)?;
        gltf_plugin.load_scene(&scene_file, engine.world()).unwrap();

        // Add a default camera if the scene didn't include one
        if engine.world().get_entities::<Camera>().is_empty() {
            engine.world().add_entity(Camera {
                up: (0.0, 1.0, 0.0).into(),
                eye: (0.0, 40.0, 150.0).into(),
                target: (0.0, 40.0, 0.0).into(),
                // target: (-60.52, 651.50, -38.69).into(),
                aspect: 800.0 / 600.0,
                fovy: 45.0_f32.to_radians(),
                near: 0.1,
                far: 1000.0,
            });
        }

        Ok(Self { camera_speed: 1.5 })
    }

    async fn run(&mut self, engine: &mut Engine) -> Result<(), ApplicationError> {
        let input = engine.input().pop();

        let mut cameras = engine.world().get_entities_mut::<Camera>();
        let Some(camera) = cameras.first_mut() else {
            return Ok(());
        };

        let up = -camera.eye.normalized().dot(Vec3::unit_y());
        let left = camera.eye.normalized().dot(Vec3::unit_x());

        if let Some(InputEvent::KeyPressed { key_code }) = input {
            match key_code {
                KeyCodes::W => {
                    camera.eye.y += up * self.camera_speed;
                }
                KeyCodes::S => {
                    camera.eye.y -= up * self.camera_speed;
                }
                KeyCodes::A => {
                    camera.eye.x += left * self.camera_speed;
                }
                KeyCodes::D => {
                    camera.eye.x += left * self.camera_speed;
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
        .filter_level(LevelFilter::Info)
        .format_source_path(true)
        .format_level(true)
        .format_timestamp_millis()
        .parse_default_env()
        .init();

    let mut engine = Engine::default();

    let asset_manager = engine.asset_manager();
    asset_manager.add_resolver("wgsl", ShaderAssetResolver);

    let gbuffer_shader = Arc::new(ShaderAsset::from_raw(include_str!(
        "../shaders/gbuffer.wgsl"
    )));

    let present_shader = Arc::new(ShaderAsset::from_raw(include_str!(
        "../shaders/present.wgsl"
    )));

    let builder = RenderGraphPBRBuilder::new(gbuffer_shader, present_shader);

    let mut application = WindowedApplication::<MyApplication, _>::create(engine, builder).unwrap();
    application.run().await.unwrap();

    Ok(())
}
