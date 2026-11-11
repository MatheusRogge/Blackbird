//! Renders a glTF scene offscreen with the PBR pipeline and writes the final frame as a PNG.
//!
//! ```text
//! cargo run -p sandbox --bin capture -- <asset-root> <scene> <out.png> [options]
//!   --size WxH           default 1280x720
//!   --frames N           frames rendered before the capture, so GI can converge (default 400)
//!   --stats              print per-pass CPU/GPU timings after every frame
//!   --eye x,y,z          camera position (default: the scene's main camera)
//!   --target x,y,z       point the camera looks at
//!   --exposure E         linear exposure override
//!   --no-specular --no-reflections --no-ssr --no-reflection-probe --no-emissive-lights --no-indirect --no-direct --no-shadows --no-particles
//!   --no-ao --no-contact-shadows --no-shadow-mask --no-traced-reflections
//!   --view lit|albedo|normal|material|emissive|ao|indirect|ssr|reflections|reflection-sources|shadows|shadow-detail
//!   --gi-intensity X     indirect diffuse multiplier (default 1)
//!   --gi-bounce X        share of the probes' irradiance reflected again by their hits, 0–1 (default 1)
//!   --gi-spacing X       DDGI probe spacing in metres (default 5)
//!   --point x,y,z,I[,r]  add a white point light of I candela (range 10), source radius r (default 0)
//!   --reloads N          unload and reload the scene N times first, like an editor hot reload
//!   --burst N            every particle emitter bursts N particles on the second frame
//!   --move dx,dy,dz      over --move-frames frames before the end (and before --hold), move the camera by this much per frame
//!   --turn D             ... and turn it left by D degrees per frame
//!   --move-frames N      default 0
//!   --no-animation       keep scene animations (e.g. flickering lamps) at their first frame
//!   --hold N             ... then keep it still for the last N frames (default 0)
//!   --sequence           also write each frame from the start of the move, as <out>_<frame>.png
//! ```

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result, bail};
use engine::assets::{AssetManager, DirectorySource};
use engine::gltf::objects::{self, ObjectKind, PLAYER_EYE_HEIGHT};
use engine::gltf::{GLTFAsset, GltfScene, lights};
use engine::rendering::camera::Camera;
use engine::rendering::debug_flags::{GBufferView, RenderDebugFlags};
use engine::rendering::pbr::RenderGraphPBRBuilder;
use engine::rendering::shader::ShaderAsset;
use engine::rendering::renderer::Renderer;
use engine::rendering::light::{AreaLight, PointLight, SkyLight, SpotLight};
use engine::rendering::mesh::Mesh;
use engine::rendering::particles::ParticleEmitter;
use engine::world::World;
use engine::{Entity, EntityKey};
use ultraviolet::Vec3;

/// Simulated time per rendered frame (particles), so a capture is the same on any machine.
const CAPTURE_FRAME_TIME: f32 = 1.0 / 60.0;

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
    burst: u32,
    points: Vec<(Vec3, f32, f32)>,
    stats: bool,
    flags: RenderDebugFlags,
    step: Vec3,
    turn: f32,
    move_frames: u32,
    hold: u32,
    sequence: bool,
    animate: bool,
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
        burst: 0,
        points: Vec::new(),
        stats: false,
        flags: RenderDebugFlags::default(),
        step: Vec3::zero(),
        turn: 0.0,
        move_frames: 0,
        hold: 0,
        sequence: false,
        animate: true,
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
            "--stats" => options.stats = true,
            "--eye" => options.eye = Some(parse_vec3(&value()?)?),
            "--target" => options.target = Some(parse_vec3(&value()?)?),
            "--exposure" => options.exposure = Some(value()?.parse()?),
            "--reloads" => options.reloads = value()?.parse()?,
            "--burst" => options.burst = value()?.parse()?,
            "--move" => options.step = parse_vec3(&value()?)?,
            "--turn" => options.turn = value()?.parse::<f32>()?.to_radians(),
            "--move-frames" => options.move_frames = value()?.parse()?,
            "--hold" => options.hold = value()?.parse()?,
            "--sequence" => options.sequence = true,
            "--no-animation" => options.animate = false,
            "--point" => {
                let v = value()?;
                let parts: Vec<&str> = v.split(',').collect();
                let [x, y, z, intensity, rest @ ..] = parts.as_slice() else {
                    bail!("--point x,y,z,intensity[,source_radius]");
                };
                let source_radius = rest.first().map_or(Ok(0.0), |r| r.parse())?;
                options.points.push((parse_vec3(&format!("{x},{y},{z}"))?, intensity.parse()?, source_radius));
            }
            "--no-specular" => flags.specular = false,
            "--no-reflections" => flags.reflections = false,
            "--no-ssr" => flags.ssr = false,
            "--no-traced-reflections" => flags.traced_reflections = false,
            "--no-reflection-probe" => flags.reflection_probe = false,
            "--no-emissive-lights" => flags.emissive_lights = false,
            "--no-indirect" => flags.indirect = false,
            "--no-direct" => flags.direct = false,
            "--no-shadows" => flags.shadows = false,
            "--no-particles" => flags.particles = false,
            "--no-ao" => flags.ambient_occlusion = false,
            "--no-contact-shadows" => flags.contact_shadows = false,
            "--no-shadow-mask" => flags.shadow_mask = false,
            "--gi-intensity" => flags.ddgi_indirect_intensity = value()?.parse()?,
            "--gi-bounce" => flags.ddgi_bounce = value()?.parse()?,
            "--gi-spacing" => flags.ddgi_probe_spacing = value()?.parse()?,
            "--view" => {
                flags.view = match value()?.as_str() {
                    "lit" => GBufferView::Lit,
                    "albedo" => GBufferView::Albedo,
                    "normal" => GBufferView::Normal,
                    "material" => GBufferView::Material,
                    "emissive" => GBufferView::Emissive,
                    "shadows" => GBufferView::LocalShadows,
                    "shadow-detail" => GBufferView::ShadowDetail,
                    "ao" => GBufferView::AmbientOcclusion,
                    "indirect" => GBufferView::IndirectDiffuse,
                    "ssr" => GBufferView::Ssr,
                    "reflections" => GBufferView::Reflections,
                    "reflection-sources" => GBufferView::ReflectionSources,
                    other => bail!("unknown view {other}"),
                }
            }
            other => bail!("unknown option {other}\n{usage}"),
        }
    }
    Ok(options)
}

/// Loads the scene like the player does: lights, every mesh, the editor's objects (shapes, placed
/// models), and only the main camera, or one at the player start. Each call reads the file again,
/// so its textures are new allocations.
/// Returns the loaded scenes, which `animate` the world each frame.
fn load_scene(options: &Options, world: &mut World) -> Result<Vec<GltfScene>> {
    let mut assets = AssetManager::default();
    assets.set_source(Arc::new(DirectorySource::new(&options.root)));
    engine::register_asset_resolvers(&mut assets);
    let asset: Arc<GLTFAsset> = assets.load_asset(&options.scene)?;

    let lights: Vec<_> =
        lights::read_lights(&asset.document)?.into_iter().map(|(node, light)| (node, light.spawn(world))).collect();
    for &(position, intensity, source_radius) in &options.points {
        world.add_entity(PointLight { position, color: Vec3::one(), intensity, radius: 10.0, source_radius });
    }
    let mut scene = GltfScene::new(Arc::clone(&asset));
    scene.bind_lights(lights);
    while scene.load_batch(world, 1000) {}
    if let Some(main) = scene.main_camera() {
        for &key in scene.node_cameras().values().filter(|&&key| key != main) {
            world.remove_entity(key);
        }
    }

    let mut player_start = None;
    let objects = objects::read_objects(&asset.document)?;
    let mut scenes = vec![scene];
    for object in objects {
        match &object.kind {
            ObjectKind::Shape { shape, material } => {
                world.add_entity(shape.mesh(material).with_transform(object.matrix()));
            }
            ObjectKind::Model { uri } => {
                let uri = engine::gltf::external_uri(uri).map_or_else(|| uri.clone(), |u| u.into_owned());
                let path = engine::assets::path::resolve_relative(Path::new(&options.scene), &uri);
                let model: Arc<GLTFAsset> = assets.load_asset(path.to_string_lossy().replace('\\', "/"))?;
                let mut placed = GltfScene::new(model).with_root(object.matrix());
                while placed.load_batch(world, 1000) {}
                for &key in placed.node_cameras().values() {
                    world.remove_entity(key);
                }
                scenes.push(placed);
            }
            ObjectKind::PlayerStart => player_start = player_start.or(Some(object)),
        }
    }
    if let Some(start) = player_start {
        let cameras: Vec<_> = world.get_entity_handles::<Camera>().iter().map(|h| h.id).collect();
        for key in cameras {
            world.remove_entity(key);
        }
        let eye = Vec3::from(start.translation) + Vec3::unit_y() * PLAYER_EYE_HEIGHT;
        let forward = start.matrix().transform_vec3(-Vec3::unit_z());
        world.add_entity(Camera {
            up: Vec3::unit_y(),
            eye,
            target: eye + Vec3::new(forward.x, 0.0, forward.z),
            aspect: None,
            fovy: 60.0_f32.to_radians(),
            near: 0.01,
            far: 10_000.0,
            exposure: 1.0,
        });
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
    Ok(scenes)
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
    remove::<ParticleEmitter>(world);
}

fn move_camera(world: &mut World, step: Vec3, turn: f32) {
    let Some(camera) = world.get_entities_mut::<Camera>().into_iter().next() else { return };
    let forward = camera.target - camera.eye;
    let (sin, cos) = turn.sin_cos();
    let forward = Vec3::new(forward.x * cos + forward.z * sin, forward.y, forward.z * cos - forward.x * sin);
    camera.eye += step;
    camera.target = camera.eye + forward;
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
    let mut scenes = load_scene(&options, &mut world)?;

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
        shader("ssr_temporal.wgsl")?,
        shader("particle_sim.wgsl")?,
        shader("ao.wgsl")?,
    )
    .with_debug_flags(flags);

    let (width, height) = options.size;
    let mut renderer = pollster::block_on(Renderer::new_headless(builder, width, height))?;
    for _ in 0..options.reloads {
        for _ in 0..3 {
            renderer.render(&world, None);
        }
        clear_scene(&mut world);
        scenes = load_scene(&options, &mut world)?;
    }
    for frame in 0..options.frames {
        world.tick_all(CAPTURE_FRAME_TIME);
        if options.animate {
            for scene in &mut scenes {
                scene.animate(&mut world, CAPTURE_FRAME_TIME);
            }
        }
        let recorded = frame + options.move_frames + options.hold >= options.frames;
        if recorded && frame + options.hold < options.frames {
            move_camera(&mut world, options.step, options.turn);
        }
        if frame == 1 && options.burst > 0 {
            for emitter in world.get_entities_mut::<ParticleEmitter>() {
                emitter.burst(options.burst);
            }
        }
        renderer.render(&world, None);
        if options.stats {
            eprintln!("{}", renderer.frame_stats());
        }
        if recorded && options.sequence {
            let stem = options.out.with_extension("");
            save(&renderer, &PathBuf::from(format!("{}_{frame:04}.png", stem.display())))?;
        }
    }
    save(&renderer, &options.out)
}

fn save(renderer: &Renderer, path: &Path) -> Result<()> {
    let frame = renderer.read_frame()?;
    image::save_buffer(path, &frame.rgba, frame.width, frame.height, image::ExtendedColorType::Rgba8)?;
    println!("wrote {}", path.display());
    Ok(())
}
