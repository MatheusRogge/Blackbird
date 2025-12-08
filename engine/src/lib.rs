use crate::{asset::AssetManager, input::InputBuffer, world::World};

pub mod application;
pub mod asset;
pub mod entity;
pub mod input;
pub mod plugin;
pub mod world;

#[derive(Default)]
pub struct Engine {
    world: World,
    input: InputBuffer,
    asset_manager: AssetManager,
}

impl Engine {
    pub fn asset_manager(&mut self) -> &mut AssetManager {
        &mut self.asset_manager
    }

    pub fn world(&mut self) -> &mut World {
        &mut self.world
    }

    pub fn input(&mut self) -> &mut InputBuffer {
        &mut self.input
    }
}
