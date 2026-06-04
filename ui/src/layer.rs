use engine_core::world::World;

pub trait UiLayer: Send + 'static {
    fn ui(&mut self, ctx: &egui::Context, world: &World);
}
