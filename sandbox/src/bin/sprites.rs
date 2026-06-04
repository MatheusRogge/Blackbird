use std::sync::Arc;

use engine::{
    Engine,
    plugin::{EnginePluginError, Plugin},
    windowed::WindowedApplication,
};
use rendering::rendering_2d::{Camera2D, Sprite, SpriteRenderGraphBuilder, SpriteTexture};
use ultraviolet::Vec2;

fn solid_color_texture(r: u8, g: u8, b: u8) -> Arc<SpriteTexture> {
    let size = 64u32;
    let data = (0..size * size)
        .flat_map(|_| [r, g, b, 255u8])
        .collect();
    Arc::new(SpriteTexture { data, width: size, height: size })
}

fn checkerboard_texture(r: u8, g: u8, b: u8) -> Arc<SpriteTexture> {
    let size = 64u32;
    let mut data = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let on = ((x / 8) + (y / 8)) % 2 == 0;
            let (pr, pg, pb) = if on { (r, g, b) } else { (20, 20, 20) };
            data.extend_from_slice(&[pr, pg, pb, 255]);
        }
    }
    Arc::new(SpriteTexture { data, width: size, height: size })
}

struct SpritesDemo {
    time: f32,
}

impl Default for SpritesDemo {
    fn default() -> Self {
        Self { time: 0.0 }
    }
}

impl Plugin for SpritesDemo {
    fn setup(&mut self, engine: &mut Engine) -> Result<(), EnginePluginError> {
        let mut world = engine.world();

        world.add_entity(Camera2D::default());

        // Red — back left, rotates
        let mut red = Sprite::new(solid_color_texture(220, 60, 60));
        red.position = Vec2::new(-200.0, 0.0);
        red.scale = Vec2::new(120.0, 120.0);
        red.z_order = 0.0;
        world.add_entity(red);

        // Checkerboard white — center, overlaps both neighbors, mid depth
        let mut checker = Sprite::new(checkerboard_texture(200, 200, 200));
        checker.position = Vec2::new(0.0, 0.0);
        checker.scale = Vec2::new(160.0, 160.0);
        checker.z_order = 0.5;
        world.add_entity(checker);

        // Blue — back right
        let mut blue = Sprite::new(solid_color_texture(60, 120, 220));
        blue.position = Vec2::new(200.0, 0.0);
        blue.scale = Vec2::new(120.0, 120.0);
        blue.z_order = 0.0;
        world.add_entity(blue);

        // Green — front center, small, drawn on top of everything
        let mut green = Sprite::new(solid_color_texture(60, 200, 90));
        green.position = Vec2::new(0.0, 0.0);
        green.scale = Vec2::new(60.0, 60.0);
        green.z_order = 1.0;
        world.add_entity(green);

        Ok(())
    }

    fn tick(&mut self, engine: &mut Engine, delta: f32) {
        self.time += delta;

        let mut world = engine.world();
        let mut sprites = world.get_entities_mut::<Sprite>();

        for (i, sprite) in sprites.iter_mut().enumerate() {
            let dir = if i % 2 == 0 { 1.0 } else { -1.0 };
            sprite.rotation = self.time * 0.8 * dir;

            // Gently bob sprites up and down at different phases
            let phase = i as f32 * std::f32::consts::TAU / 4.0;
            sprite.position.y = (self.time * 1.5 + phase).sin() * 30.0;
        }
    }
}

fn main() -> anyhow::Result<()> {
    WindowedApplication::new(SpriteRenderGraphBuilder)
        .add_plugin(SpritesDemo::default())
        .run()?;

    Ok(())
}
