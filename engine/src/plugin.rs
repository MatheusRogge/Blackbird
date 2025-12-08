use thiserror::Error;

use crate::Engine;

#[derive(Error, Debug)]
pub enum EnginePluginError {
    #[error("unknown data store error")]
    Unknown,
}

pub trait EnginePlugin {
    fn setup(&self, engine: &mut Engine) -> Result<Self, EnginePluginError>
    where
        Self: Sized;
}
