#![cfg_attr(all(windows, not(feature = "dev")), windows_subsystem = "windows")]

#[cfg(feature = "dev")]
mod dev_tools;
mod game;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use engine::assets::pack::{PackReader, PackSource};
use engine::assets::{AssetManifest, AssetSource, DirectorySource};
use engine::logging::{LevelFilter, LogConfig};
use engine::rendering::pbr::RenderGraphPBRBuilder;
use engine::rendering::shader::ShaderAsset;
use engine::windowed::WindowedApplication;
use project::Project;

mod generated {
    use engine::assets::StaticAsset;
    include!(concat!(env!("OUT_DIR"), "/assets.rs"));
}

/// Where this binary gets its assets, decided at compile time by `build.rs`.
pub enum Content {
    /// Production: `data.bbpak` next to the executable.
    Pack,
    /// Development: loose files in `data/` next to the executable.
    Loose,
    /// Built without a manifest (plain `cargo run -p player`): a project dir given on the command line.
    Unbound,
}

impl Content {
    const CURRENT: Content = match (generated::PACK_ID, generated::ASSETS.is_empty()) {
        (Some(_), _) => Content::Pack,
        (None, false) => Content::Loose,
        (None, true) => Content::Unbound,
    };

    pub fn label(&self) -> &'static str {
        match self {
            Content::Pack => "packed",
            Content::Loose => "loose files",
            Content::Unbound => "unbound project dir",
        }
    }
}

fn exe_dir() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("locating the executable")?;
    Ok(exe.parent().context("executable has no parent dir")?.to_path_buf())
}

fn asset_source(exe_dir: &Path) -> Result<(Arc<dyn AssetSource>, Option<AssetManifest>)> {
    let manifest = AssetManifest::from_static(generated::ASSETS);
    match Content::CURRENT {
        Content::Pack => {
            let path = exe_dir.join(project::PACK_FILE);
            let reader = PackReader::open(&path).with_context(|| format!("opening {}", path.display()))?;
            let expected = generated::PACK_ID.unwrap_or_default();
            if reader.content_id() != expected {
                bail!(
                    "{} doesn't belong to this build (expected content id {expected:016x}, found {:016x})",
                    path.display(),
                    reader.content_id()
                );
            }
            Ok((Arc::new(PackSource::new(reader)), Some(manifest)))
        }
        Content::Loose => Ok((Arc::new(DirectorySource::new(exe_dir.join(project::DATA_DIR))), Some(manifest))),
        Content::Unbound => {
            let dir = std::env::args_os()
                .nth(1)
                .context("this player wasn't built for a project; pass a project directory as the first argument")?;
            Ok((Arc::new(DirectorySource::new(dir)), None))
        }
    }
}

fn main() -> Result<()> {
    let exe_dir = exe_dir()?;

    #[cfg(feature = "dev")]
    let log_history = engine::logging::init(LogConfig {
        level: LevelFilter::Info,
        file: Some(exe_dir.join("logs").join("player.log")),
        history: 500,
    })?;
    #[cfg(not(feature = "dev"))]
    engine::logging::init(LogConfig { level: LevelFilter::Warn, ..LogConfig::default() })?;

    let (source, manifest) = asset_source(&exe_dir)?;
    log::info!("Assets: {} ({})", source.describe(), Content::CURRENT.label());
    let project_json = source.read(Path::new(project::MANIFEST_FILE)).context("reading project.json")?;
    let project = Project::parse(&project_json)?;
    let scene = project.default_scene.clone().context("the project has no default scene")?;

    let render_graph = RenderGraphPBRBuilder::new(
        ShaderAsset::from_raw(include_str!("../../sandbox/shaders/gbuffer.wgsl")),
        ShaderAsset::from_raw(include_str!("../../sandbox/shaders/cluster_assignment.wgsl")),
        ShaderAsset::from_raw(include_str!("../../sandbox/shaders/lighting.wgsl")),
        ShaderAsset::from_raw(include_str!("../../sandbox/shaders/present.wgsl")),
        ShaderAsset::from_raw(include_str!("../../sandbox/shaders/shadow.wgsl")),
        ShaderAsset::from_raw(include_str!("../../sandbox/shaders/probe_trace.wgsl")),
        ShaderAsset::from_raw(include_str!("../../sandbox/shaders/probe_update.wgsl")),
        ShaderAsset::from_raw(include_str!("../../sandbox/shaders/debug_overlay.wgsl")),
        ShaderAsset::from_raw(include_str!("../../sandbox/shaders/ssr.wgsl")),
        ShaderAsset::from_raw(include_str!("../../sandbox/shaders/composite.wgsl")),
    );

    #[cfg(feature = "dev")]
    let dev = dev_tools::DevState::shared(&project, source.describe(), manifest.as_ref().map(AssetManifest::len));

    let title = project.name.clone();
    let configure_assets = move |engine: &mut engine::Engine| {
        engine.assets_mut().set_source(Arc::clone(&source));
        if let Some(manifest) = manifest.clone() {
            engine.assets_mut().set_manifest(manifest);
        }
        Ok(())
    };

    let app = WindowedApplication::new(render_graph)
        .with_title(title)
        .add_plugin(configure_assets)
        .add_plugin(game::Game::new(
            project,
            scene,
            #[cfg(feature = "dev")]
            Arc::clone(&dev),
        ));

    #[cfg(feature = "dev")]
    let app = app.add_ui_layer(dev_tools::DevToolsLayer::new(dev, log_history));

    app.run()?;
    Ok(())
}
