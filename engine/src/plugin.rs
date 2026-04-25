use thiserror::Error;

use crate::Engine;

#[derive(Error, Debug)]
pub struct EnginePluginError {
    pub message: String,
    #[source]
    pub source: Box<dyn std::error::Error + Send + Sync + 'static>,
}

impl std::fmt::Display for EnginePluginError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EnginePluginError: {}", self.message)
    }
}

pub trait Plugin: Send + 'static {
    fn setup(&mut self, engine: &mut Engine) -> Result<(), EnginePluginError>;
    fn tick(&mut self, _engine: &mut Engine, _delta: f32) {}
}

/// Blanket impl: closures work as setup-only plugins.
impl<F> Plugin for F
where
    F: FnMut(&mut Engine) -> Result<(), EnginePluginError> + Send + 'static,
{
    fn setup(&mut self, engine: &mut Engine) -> Result<(), EnginePluginError> {
        self(engine)
    }
}
