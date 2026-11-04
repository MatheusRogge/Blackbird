//! Turns an editor project into a standalone game: walks the project's assets through the
//! engine's resolvers, stages them (loose files or a `.bbpak`) and compiles `player` against
//! the resulting manifest.

use std::{
    fmt, fs,
    io::{self, BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::Arc,
    thread,
};

use engine::assets::{
    AssetError, AssetManager, AssetManifest, DirectorySource, manifest::content_hash, pack::PackWriter,
};
use project::{DATA_DIR, PACK_FILE, Project, ProjectError};
use thiserror::Error;

/// Environment variables read by `player/build.rs`.
pub const MANIFEST_ENV: &str = "BLACKBIRD_ASSET_MANIFEST";
pub const PACK_ID_ENV: &str = "BLACKBIRD_PACK_ID";
const MARKER_FILE: &str = ".blackbird-build";
const BUILDS_DIR: &str = "builds";

#[derive(Error, Debug)]
pub enum BuildError {
    #[error(transparent)]
    Project(#[from] ProjectError),

    #[error("collecting assets: {0}")]
    Assets(#[from] AssetError),

    #[error("{action} {path}: {source}")]
    Io { action: &'static str, path: PathBuf, source: io::Error },

    #[error("{0} changed while building; run the build again")]
    AssetChanged(String),

    #[error("{0} is not empty and wasn't created by a build; pick another output directory")]
    UnmanagedOutput(PathBuf),

    #[error("running cargo: {0}")]
    Spawn(io::Error),

    #[error("cargo build failed ({0})")]
    Cargo(std::process::ExitStatus),

    #[error("cargo didn't report a player executable")]
    NoExecutable,
}

fn io_err(action: &'static str, path: &Path) -> impl FnOnce(io::Error) -> BuildError {
    let path = path.to_path_buf();
    move |source| BuildError::Io { action, path, source }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildKind {
    /// Loose assets next to the binary, dev tools and file logging, light optimisation.
    Development,
    /// Assets in a compressed `.bbpak`, no dev tools, fully optimised and stripped.
    Production,
}

impl BuildKind {
    pub fn profile(self) -> &'static str {
        match self {
            BuildKind::Development => "development",
            BuildKind::Production => "production",
        }
    }
}

impl fmt::Display for BuildKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.profile())
    }
}

#[derive(Debug, Clone)]
pub struct BuildOptions {
    pub project_dir: PathBuf,
    pub kind: BuildKind,
    /// Defaults to `<project>/builds/<kind>`.
    pub out_dir: Option<PathBuf>,
    /// Cargo target triple to cross-compile for.
    pub target: Option<String>,
    /// zstd level for the pack, 1–22.
    pub compression_level: i32,
}

impl BuildOptions {
    pub fn new(project_dir: impl Into<PathBuf>, kind: BuildKind) -> Self {
        Self { project_dir: project_dir.into(), kind, out_dir: None, target: None, compression_level: 19 }
    }

    pub fn resolved_out_dir(&self) -> PathBuf {
        self.out_dir.clone().unwrap_or_else(|| default_out_dir(&self.project_dir, self.kind))
    }
}

pub fn default_out_dir(project_dir: &Path, kind: BuildKind) -> PathBuf {
    project_dir.join(BUILDS_DIR).join(kind.profile())
}

/// Top-level directory holding build outputs; asset browsers should skip it.
pub fn builds_dir(project_dir: &Path) -> PathBuf {
    project_dir.join(BUILDS_DIR)
}

#[derive(Debug, Clone)]
pub struct BuildOutput {
    pub out_dir: PathBuf,
    pub executable: PathBuf,
    pub assets: usize,
    pub raw_bytes: u64,
    /// Bytes on disk for the assets: the pack size in production, `raw_bytes` otherwise.
    pub stored_bytes: u64,
}

/// The workspace this tool was compiled from; `player` is built from it.
pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("build_tool lives in the workspace").to_path_buf()
}

/// An `AssetManager` reading loose project files, with the engine's resolvers.
pub fn project_asset_manager(project_dir: &Path) -> AssetManager {
    let mut manager = AssetManager::default();
    manager.set_source(Arc::new(DirectorySource::new(project_dir)));
    engine::register_asset_resolvers(&mut manager);
    manager
}

/// Everything a build of the project would ship.
pub fn collect_manifest(project_dir: &Path) -> Result<(Project, AssetManifest), BuildError> {
    let project = Project::load(project_dir)?;
    let manifest = project_asset_manager(project_dir).collect(&project.build_roots()?)?;
    Ok((project, manifest))
}

pub fn build(options: &BuildOptions, log: &mut dyn FnMut(String)) -> Result<BuildOutput, BuildError> {
    let project_dir = &options.project_dir;
    let out_dir = options.resolved_out_dir();

    log(format!("Collecting assets of {}", project_dir.display()));
    let (project, manifest) = collect_manifest(project_dir)?;
    log(format!("{} assets, {} total", manifest.len(), format_bytes(manifest.total_size())));
    for entry in manifest.entries() {
        log(format!("  {} ({})", entry.path, format_bytes(entry.size)));
    }

    prepare_out_dir(&out_dir)?;
    let read_asset = |path: &str| -> Result<Vec<u8>, BuildError> {
        let file = project_dir.join(path);
        let bytes = fs::read(&file).map_err(io_err("reading", &file))?;
        match manifest.get(path) {
            Some(entry) if entry.hash == content_hash(&bytes) => Ok(bytes),
            _ => Err(BuildError::AssetChanged(path.to_string())),
        }
    };

    let (pack_id, stored_bytes) = match options.kind {
        BuildKind::Development => {
            let data_dir = out_dir.join(DATA_DIR);
            log(format!("Copying assets to {}", data_dir.display()));
            for entry in manifest.entries() {
                let dest = data_dir.join(&entry.path);
                if let Some(parent) = dest.parent() {
                    fs::create_dir_all(parent).map_err(io_err("creating", parent))?;
                }
                fs::write(&dest, read_asset(&entry.path)?).map_err(io_err("writing", &dest))?;
            }
            (None, manifest.total_size())
        }
        BuildKind::Production => {
            let pack_path = out_dir.join(PACK_FILE);
            log(format!("Packing assets into {} (zstd level {})", pack_path.display(), options.compression_level));
            let mut writer =
                PackWriter::create(&pack_path, options.compression_level).map_err(io_err("creating", &pack_path))?;
            for entry in manifest.entries() {
                writer.add(&entry.path, &read_asset(&entry.path)?).map_err(io_err("packing", &pack_path))?;
            }
            let summary = writer.finish().map_err(io_err("writing", &pack_path))?;
            log(format!(
                "Packed {} → {} ({:.0}%)",
                format_bytes(summary.raw_bytes),
                format_bytes(summary.stored_bytes),
                100.0 * summary.stored_bytes as f64 / summary.raw_bytes.max(1) as f64
            ));
            (Some(summary.content_id), summary.stored_bytes)
        }
    };

    let staging = workspace_root().join("target").join("blackbird").join(project.slug()).join(options.kind.profile());
    fs::create_dir_all(&staging).map_err(io_err("creating", &staging))?;
    let manifest_path = staging.join("assets.manifest");
    fs::write(&manifest_path, manifest.to_text()).map_err(io_err("writing", &manifest_path))?;

    let built = cargo_build_player(options, &manifest_path, pack_id, log)?;
    let file_name = match built.extension() {
        Some(ext) => format!("{}.{}", project.slug(), ext.to_string_lossy()),
        None => project.slug(),
    };
    let executable = out_dir.join(file_name);
    fs::copy(&built, &executable).map_err(io_err("copying", &built))?;
    log(format!("Built {}", executable.display()));

    Ok(BuildOutput { out_dir, executable, assets: manifest.len(), raw_bytes: manifest.total_size(), stored_bytes })
}

/// Creates `out_dir`, or clears what a previous build left in it. Refuses to touch a non-empty
/// directory without the marker a build leaves behind.
fn prepare_out_dir(out_dir: &Path) -> Result<(), BuildError> {
    let marker = out_dir.join(MARKER_FILE);
    match fs::read_dir(out_dir) {
        Ok(mut entries) => {
            if !marker.exists() && entries.next().is_some() {
                return Err(BuildError::UnmanagedOutput(out_dir.to_path_buf()));
            }
            let data_dir = out_dir.join(DATA_DIR);
            if data_dir.exists() {
                fs::remove_dir_all(&data_dir).map_err(io_err("clearing", &data_dir))?;
            }
            let pack = out_dir.join(PACK_FILE);
            if pack.exists() {
                fs::remove_file(&pack).map_err(io_err("removing", &pack))?;
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            fs::create_dir_all(out_dir).map_err(io_err("creating", out_dir))?;
        }
        Err(e) => return Err(io_err("reading", out_dir)(e)),
    }
    fs::write(&marker, "Created by the Blackbird build tool. Its contents are replaced on every build.\n")
        .map_err(io_err("writing", &marker))
}

fn cargo_build_player(
    options: &BuildOptions,
    manifest_path: &Path,
    pack_id: Option<u64>,
    log: &mut dyn FnMut(String),
) -> Result<PathBuf, BuildError> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut cmd = Command::new(cargo);
    cmd.current_dir(workspace_root())
        .args(["build", "-p", "player", "--profile", options.kind.profile()])
        .args(["--message-format", "json-render-diagnostics"])
        .env(MANIFEST_ENV, manifest_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    match pack_id {
        Some(id) => cmd.env(PACK_ID_ENV, format!("{id:016x}")),
        None => cmd.env_remove(PACK_ID_ENV),
    };
    if options.kind == BuildKind::Production {
        cmd.args(["--no-default-features", "--features", "production"]);
    }
    if let Some(target) = &options.target {
        cmd.args(["--target", target]);
    }

    log(format!("Compiling player ({} profile)", options.kind.profile()));
    let mut child = cmd.spawn().map_err(BuildError::Spawn)?;
    let stdout = child.stdout.take().expect("stdout is piped");
    let artifacts = thread::spawn(move || {
        let mut executable = None;
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) else { continue };
            if msg["reason"] == "compiler-artifact"
                && msg["target"]["name"] == "player"
                && let Some(path) = msg["executable"].as_str()
            {
                executable = Some(PathBuf::from(path));
            }
        }
        executable
    });

    let stderr = child.stderr.take().expect("stderr is piped");
    for line in BufReader::new(stderr).lines().map_while(Result::ok) {
        log(line);
    }

    let status = child.wait().map_err(BuildError::Spawn)?;
    let executable = artifacts.join().ok().flatten();
    if !status.success() {
        return Err(BuildError::Cargo(status));
    }
    executable.ok_or(BuildError::NoExecutable)
}

pub fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 { format!("{bytes} B") } else { format!("{value:.1} {}", UNITS[unit]) }
}
