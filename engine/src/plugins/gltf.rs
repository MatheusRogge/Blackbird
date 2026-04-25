use crate::Engine;
use crate::plugin::{EnginePluginError, Plugin};

#[derive(Default)]
pub struct GltfPlugin;

impl Plugin for GltfPlugin {
    fn setup(&mut self, engine: &mut Engine) -> Result<(), EnginePluginError> {
        engine.assets_mut().add_resolver("gltf", gltf::GLTFAssetResolver);
        engine.assets_mut().add_resolver("glb", gltf::GLTFAssetResolver);
        Ok(())
    }
}
