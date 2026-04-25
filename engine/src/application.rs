use std::fmt;
use std::sync::Arc;
use std::time::Instant;
use std::thread;

use threading::{DefaultExecutor, Executor};

use crate::Engine;
use crate::plugin::{EnginePluginError, Plugin};

#[derive(Debug)]
pub struct ApplicationError {
    pub message: String,
    pub source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
}

impl fmt::Display for ApplicationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ApplicationError: {}", self.message)
    }
}

impl std::error::Error for ApplicationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_ref().map(|e| e.as_ref() as &dyn std::error::Error)
    }
}

impl From<EnginePluginError> for ApplicationError {
    fn from(e: EnginePluginError) -> Self {
        Self {
            message: e.to_string(),
            source: Some(Box::new(e)),
        }
    }
}

pub struct HeadlessApplication {
    pub(crate) engine:  Engine,
    pub(crate) plugins: Vec<Box<dyn Plugin>>,
}

impl HeadlessApplication {
    pub fn new() -> Self {
        Self::with_executor(Arc::new(DefaultExecutor::new()))
    }

    pub fn with_executor(exec: Arc<dyn Executor>) -> Self {
        Self {
            engine: Engine::new(exec),
            plugins: Vec::new(),
        }
    }

    pub fn add_plugin(mut self, mut plugin: impl Plugin) -> Self {
        // Run setup immediately so ordering guarantees hold (earlier plugins init first).
        // Errors are treated as fatal during construction.
        plugin.setup(&mut self.engine).expect("plugin setup failed");
        self.plugins.push(Box::new(plugin));
        self
    }

    /// Run a fixed-timestep headless game loop at ~60 Hz until all plugins complete
    /// or the process is interrupted. Intended for game servers and testing.
    pub fn run(mut self) -> Result<(), ApplicationError> {
        let target_delta = 1.0_f32 / 60.0;

        loop {
            let frame_start = Instant::now();

            {
                let mut world = self.engine.world();
                world.tick_all(target_delta);
            }

            for plugin in &mut self.plugins {
                plugin.tick(&mut self.engine, target_delta);
            }

            let elapsed = frame_start.elapsed().as_secs_f32();
            if elapsed < target_delta {
                let sleep = std::time::Duration::from_secs_f32(target_delta - elapsed);
                thread::sleep(sleep);
            }
        }
    }
}

impl Default for HeadlessApplication {
    fn default() -> Self {
        Self::new()
    }
}
