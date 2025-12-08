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
        f.write_fmt(format_args!("EnginePluginError: {}", self.message))
    }
}

pub trait EnginePlugin {
    fn setup(&self, engine: &mut Engine) -> Result<Self, EnginePluginError>
    where
        Self: Sized;
}
