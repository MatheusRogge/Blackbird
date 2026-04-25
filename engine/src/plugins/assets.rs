use crate::Engine;
use crate::plugin::{EnginePluginError, Plugin};

#[derive(Default)]
pub struct AssetsPlugin;

impl Plugin for AssetsPlugin {
    fn setup(&mut self, engine: &mut Engine) -> Result<(), EnginePluginError> {
        engine.assets.init_executor(engine.executor());
        Ok(())
    }
}
