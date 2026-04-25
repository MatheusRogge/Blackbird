use std::sync::{Arc, RwLock, RwLockWriteGuard};

use engine_core::{input::InputBuffer, world::World};
use threading::{DefaultExecutor, Executor};

pub mod player_controller;
pub mod plugin;
pub mod application;

#[cfg(feature = "window")]
pub mod windowed;

#[cfg(feature = "assets")]
mod plugins;

// Re-export core primitives so users only need to import `engine`.
pub use engine_core::{entity, input, world};
pub use engine_core::entity::{Controllable, Entity, EntityKey};
pub use engine_core::input::{InputEvent, KeyCodes, MouseButton};
pub use engine_core::world::EntityHandle;

// Re-export subsystem crates so sandbox only imports `engine`.
#[cfg(feature = "assets")]
pub use assets;

#[cfg(feature = "gltf")]
pub use gltf;

#[cfg(feature = "window")]
pub use rendering;

pub struct Engine {
    world:    Arc<RwLock<World>>,
    input:    InputBuffer,
    executor: Arc<dyn Executor>,

    #[cfg(feature = "assets")]
    pub(crate) assets: assets::AssetManager,
}

impl Engine {
    pub(crate) fn new(executor: Arc<dyn Executor>) -> Self {
        Self {
            world: Arc::new(RwLock::new(World::default())),
            input: InputBuffer::default(),
            executor,

            #[cfg(feature = "assets")]
            assets: assets::AssetManager::default(),
        }
    }

    pub fn world(&mut self) -> RwLockWriteGuard<'_, World> {
        self.world.write().expect("world RwLock poisoned")
    }

    pub fn world_arc(&self) -> Arc<RwLock<World>> {
        Arc::clone(&self.world)
    }

    pub fn input(&mut self) -> &mut InputBuffer {
        &mut self.input
    }

    /// Split borrow: returns `(&mut InputBuffer, RwLockWriteGuard<World>)` simultaneously.
    /// Useful when a subsystem needs both input and world without passing `&mut Engine`.
    pub fn input_and_world(&mut self) -> (&mut InputBuffer, RwLockWriteGuard<'_, World>) {
        let world = self.world.write().expect("world RwLock poisoned");
        (&mut self.input, world)
    }

    pub(crate) fn executor(&self) -> Arc<dyn Executor> {
        Arc::clone(&self.executor)
    }

    #[cfg(feature = "assets")]
    pub fn assets(&self) -> &assets::AssetManager {
        &self.assets
    }

    #[cfg(feature = "assets")]
    pub fn assets_mut(&mut self) -> &mut assets::AssetManager {
        &mut self.assets
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new(Arc::new(DefaultExecutor::new()))
    }
}
