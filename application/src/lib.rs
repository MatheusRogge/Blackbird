use thiserror::Error;

use engine::{Engine, asset::AssetError, plugin::EnginePluginError};

#[derive(Error, Debug)]
pub struct ApplicationError {
    pub message: String,
    #[source]
    pub source: Box<dyn std::error::Error + Send + Sync + 'static>,
}

impl From<EnginePluginError> for ApplicationError {
    fn from(value: EnginePluginError) -> Self {
        ApplicationError {
            message: format!("Plugin initialization error: {}", value),
            source: Box::new(value),
        }
    }
}

impl From<AssetError> for ApplicationError {
    fn from(value: AssetError) -> Self {
        ApplicationError {
            message: format!("Asset loading error: {}", value),
            source: Box::new(value),
        }
    }
}

impl std::fmt::Display for ApplicationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("ApplicationError: {}", self.message))
    }
}

pub trait Application {
    fn setup(engine: &mut Engine) -> Result<Self, ApplicationError>
    where
        Self: Sized;

    fn run(&mut self, engine: &mut Engine) -> impl Future<Output = Result<(), ApplicationError>>;
}
