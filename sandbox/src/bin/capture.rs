//! Renders a glTF scene offscreen with the PBR pipeline and writes the final frame as a PNG.
//!
//! ```text
//! cargo run -p sandbox --bin capture -- <asset-root> <scene> <out.png> [options]
//!   --size WxH           default 1280x720
//!   --frames N           frames rendered before the capture, so GI can converge (default 400)
//!   --eye x,y,z          camera position (default: the scene's main camera)
//!   --target x,y,z       point the camera looks at
//!   --exposure E         linear exposure override
//!   --no-specular --no-reflections --no-ssr --no-indirect --no-direct --no-shadows
//!   --view lit|albedo|normal|material|emissive
//!   --reloads N          unload and reload the scene N times first, like an editor hot reload
//! ```

use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result, bail};
use engine::assets::{AssetManager, DirectorySource};
use engine::gltf::{GLTFAsset, GltfScene, lights};
use engine::rendering::camera::Camera;
use engine::rendering::debug_flags::{GBufferView, RenderDebugFlags};
use engine::rendering::pbr::RenderGraphPBRBuilder;
use engine::rendering::shader::ShaderAsset;
use engine::rendering::renderer::Renderer;
use engine::rendering::light::{AreaLight, PointLight, SkyLight, SpotLight};
use engine::rendering::mesh::Mesh;
use engine::world::World;
use engine::{Entity, EntityKey};
use ultraviolet::Vec3;

struct Options {
    root: PathBuf,
    scene: String,
    out: PathBuf,
    size: (u32, u32),
    frames: u32,
    eye: Option<Vec3>,
    target: Option<Vec3>,
    exposure: Option<f32>,
    reloads: u32,
    flags: RenderDebugFlags,
}

fn parse_vec3(s: &str) -> Result<Vec3> {
    let v: Vec<f32> = s.split(',').map(str::parse).collect::<Result<_, _>>().context("expected x,y,z")?;
    let [x, y, z] = v[..] else { bail!("expected x,y,z, got {s}") };
    Ok(Vec3::new(x, y, z))
}

fn parse_args() -> Result<Options> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: capture <asset-root> <scene> <out.png> [options]";
    let root = args.next().context(usage)?.into();
    let scene = args.next().context(usage)?;
    let out = args.next().context(usage)?.into();
    let mut options = Options {
        root,
        scene,
        out,
        size: (1280, 720),
        frames: 400,
        eye: None,
        target: None,
        exposure: None,
        reloads: 0,
        flags: RenderDebugFlags::default(),
    };

    while let Some(arg) = args.next() {
        let mut value = || args.next().with_context(|| format!("{arg} needs a value"));
        let flags = &mut options.flags;
        match arg.as_str() {
            "--size" => {
                let v = value()?;
                let (w, h) = v.split_once('x').context("--size WxH")?;
                options.size = (w.parse()?, h.parse()?);
            }
            "--frames" => options.frames = value()?.parse()?,
            "--eye" => options.eye = Some(parse_vec3(&value()?)?),
            "--target" => options.target = Some(parse_vec3(&value()?)?),
            "--exposure" => options.exposure = Some(value()?.parse()?),
            "--reloads" => options.reloads = value()?.parse()?,
            "--no-specular" => flags.specular = false,
            "--no-reflections" => flags.reflections = false,
            "--no-ssr" => flags.ssr = false,
            "--no-indirect" => flags.indirect = false,
            "--no-direct" => flags.direct = false,
            "--no-shadows" => flags.shadows = false,
            "--view" => {
                flags.view = match value()?.as_str() {
                    "lit" => GBufferView::Lit,
                    "albedo" => GBufferView::Albedo,
                    "normal" => GBufferView::Normal,
                    "material" => GBufferView::Material,
                    "emissive" => GBufferView::Emissive,
                    other => bail!("unknown view {other}"),
                }
            }
            other => bail!("unknown option {other}\n{usage}"),
        }
    }
    Ok(options)
}

/// Loads the scene like the player does: lights, every mesh, and only the main camera. Each call
/// reads the file again, so its textures are new allocations.
fn load_scene(options: &Options, world: &mut World) -> Result<()> {
    let mut assets = AssetManager::default();
    assets.set_source(Arc::new(DirectorySource::new(&options.root)));
    engine::register_asset_resolvers(&mut assets);
    let asset: Arc<GLTFAsset> = assets.load_asset(&options.scene)?;

    for (_, light) in lights::read_lights(&asset.document)? {
        light.spawn(world);
    }
    let mut scene = GltfScene::new(Arc::clone(&asset));
    while scene.load_batch(world, 1000) {}
    if let Some(main) = scene.main_camera() {
        for &key in scene.node_cameras().values().filter(|&&key| key != main) {
            world.remove_entity(key);
        }
    }

    if world.get_entities::<Camera>().is_empty() {
        world.add_entity(Camera {
            up: Vec3::unit_y(),
            eye: Vec3::new(0.0, 2.0, 8.0),
            target: Vec3::zero(),
            aspect: None,
            fovy: 45.0_f32.to_radians(),
            near: 0.01,
            far: 10_000.0,
            exposure: 1.0,
        });
    }
    let key = world.get_entity_handles::<Camera>()[0].id;
    let camera = world.get_entity_mut::<Camera>(key).context("camera vanished")?;
    let forward = camera.target - camera.eye;
    if let Some(eye) = options.eye {
        camera.eye = eye;
        camera.target = eye + forward;
    }
    if let Some(target) = options.target {
        camera.target = target;
    }
    if let Some(exposure) = options.exposure {
        camera.exposure = exposure;
    }
    log::info!("camera eye {:?} target {:?} exposure {}", camera.eye, camera.target, camera.exposure);
    Ok(())
}

/// Removes everything `load_scene` spawned, keeping the world (and its entity key versions).
fn clear_scene(world: &mut World) {
    fn remove<E: Entity>(world: &mut World) {
        let keys: Vec<EntityKey> = world.get_entity_handles::<E>().iter().map(|h| h.id).collect();
        for key in keys {
            world.remove_entity(key);
        }
    }
    remove::<Mesh>(world);
    remove::<Camera>(world);
    remove::<PointLight>(world);
    remove::<SpotLight>(world);
    remove::<AreaLight>(world);
    remove::<SkyLight>(world);
}

fn shader(name: &str) -> Result<ShaderAsset> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders").join(name);
    let source = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    Ok(ShaderAsset::from_raw(&source))
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let mut options = parse_args()?;
    let mut world = World::default();
    load_scene(&options, &mut world)?;

    let flags = Arc::new(RwLock::new(std::mem::take(&mut options.flags)));
    let builder = RenderGraphPBRBuilder::new(
        shader("gbuffer.wgsl")?,
        shader("cluster_assignment.wgsl")?,
        shader("lighting.wgsl")?,
        shader("present.wgsl")?,
        shader("shadow.wgsl")?,
        shader("probe_trace.wgsl")?,
        shader("probe_update.wgsl")?,
        shader("debug_overlay.wgsl")?,
        shader("ssr.wgsl")?,
        shader("composite.wgsl")?,
    )
    .with_debug_flags(flags);

    let (width, height) = options.size;
    let mut renderer = pollster::block_on(Renderer::new_headless(builder, width, height))?;
    for _ in 0..options.reloads {
        for _ in 0..3 {
            renderer.render(&world, None);
        }
        clear_scene(&mut world);
        load_scene(&options, &mut world)?;
    }
    for _ in 0..options.frames {
        renderer.render(&world, None);
    }

    let frame = renderer.read_frame()?;
    image::save_buffer(&options.out, &frame.rgba, frame.width, frame.height, image::ExtendedColorType::Rgba8)?;
    println!("wrote {}", options.out.display());
    Ok(())
}
