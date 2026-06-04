use std::sync::{Arc, RwLock, RwLockWriteGuard};
use std::sync::atomic::{AtomicBool, Ordering};

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
pub use engine_core::input::{InputEvent, Key, Modifiers, MouseButton};
pub use engine_core::world::EntityHandle;

// Re-export subsystem crates so sandbox only imports `engine`.
#[cfg(feature = "assets")]
pub use assets;

#[cfg(feature = "gltf")]
pub use gltf;

#[cfg(feature = "window")]
pub use rendering;

#[cfg(feature = "ui")]
pub use ui;

pub struct Engine {
    world:    Arc<RwLock<World>>,
    input:    InputBuffer,
    executor: Arc<dyn Executor>,

    ui_wants_pointer:  Arc<AtomicBool>,
    ui_wants_keyboard: Arc<AtomicBool>,

    #[cfg(feature = "assets")]
    pub(crate) assets: assets::AssetManager,

    #[cfg(feature = "window")]
    pub(crate) frame_stats: Arc<RwLock<rendering::FrameStats>>,
}

impl Engine {
    pub(crate) fn new(executor: Arc<dyn Executor>) -> Self {
        Self {
            world: Arc::new(RwLock::new(World::default())),
            input: InputBuffer::default(),
            executor,
            ui_wants_pointer:  Arc::new(AtomicBool::new(false)),
            ui_wants_keyboard: Arc::new(AtomicBool::new(false)),

            #[cfg(feature = "assets")]
            assets: assets::AssetManager::default(),

            #[cfg(feature = "window")]
            frame_stats: Arc::new(RwLock::new(rendering::FrameStats::default())),
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
    pub fn ui_wants_pointer(&self) -> bool {
        self.ui_wants_pointer.load(Ordering::Relaxed)
    }

    pub fn ui_wants_keyboard(&self) -> bool {
        self.ui_wants_keyboard.load(Ordering::Relaxed)
    }

    pub(crate) fn ui_flag_arcs(&self) -> (Arc<AtomicBool>, Arc<AtomicBool>) {
        (Arc::clone(&self.ui_wants_pointer), Arc::clone(&self.ui_wants_keyboard))
    }

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

    #[cfg(feature = "window")]
    pub(crate) fn frame_stats_arc(&self) -> Arc<RwLock<rendering::FrameStats>> {
        Arc::clone(&self.frame_stats)
    }

    #[cfg(feature = "window")]
    pub fn frame_stats(&self) -> rendering::FrameStats {
        self.frame_stats.read().expect("frame_stats RwLock poisoned").clone()
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new(Arc::new(DefaultExecutor::new()))
    }
}
