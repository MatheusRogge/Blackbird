//! Game-thread side of the asset browser. A background thread scans the project, works out
//! which files a build would ship, and decodes thumbnails and model stats on request, so
//! neither the game nor the render thread does file I/O or image decoding for the browser.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use engine::assets::path::{resolve_relative, to_logical};
use engine::gltf::external_uri;
use gltf_api::Gltf;
use project::Project;

const THUMBNAIL_SIZE: u32 = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AssetKind {
    Scene,
    Model,
    Texture,
    Buffer,
    Shader,
    Data,
    Other,
}

impl AssetKind {
    pub const ALL: [AssetKind; 7] = [
        AssetKind::Scene,
        AssetKind::Model,
        AssetKind::Texture,
        AssetKind::Buffer,
        AssetKind::Shader,
        AssetKind::Data,
        AssetKind::Other,
    ];

    fn from_path(path: &str) -> Self {
        let ext = path.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase()).unwrap_or_default();
        match ext.as_str() {
            "gltf" | "glb" if path.starts_with("scenes/") => AssetKind::Scene,
            "gltf" | "glb" => AssetKind::Model,
            "png" | "jpg" | "jpeg" => AssetKind::Texture,
            "bin" => AssetKind::Buffer,
            "wgsl" => AssetKind::Shader,
            "json" | "toml" | "txt" | "ron" => AssetKind::Data,
            _ => AssetKind::Other,
        }
    }

    pub fn icon(self) -> &'static str {
        match self {
            AssetKind::Scene => "🎬",
            AssetKind::Model => "📦",
            AssetKind::Texture => "🖼",
            AssetKind::Buffer => "💾",
            AssetKind::Shader => "✨",
            AssetKind::Data => "📄",
            AssetKind::Other => "❔",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            AssetKind::Scene => "Scene",
            AssetKind::Model => "Model",
            AssetKind::Texture => "Texture",
            AssetKind::Buffer => "Buffer",
            AssetKind::Shader => "Shader",
            AssetKind::Data => "Data",
            AssetKind::Other => "Other",
        }
    }

    /// Kinds worth inspecting in the background for a thumbnail or details.
    pub fn inspectable(self) -> bool {
        matches!(self, AssetKind::Scene | AssetKind::Model | AssetKind::Texture)
    }
}

#[derive(Clone, Debug)]
pub struct AssetEntry {
    /// Project-relative, `/`-separated.
    pub path: String,
    pub kind: AssetKind,
    pub size: u64,
}

impl AssetEntry {
    pub fn file_name(&self) -> &str {
        self.path.rsplit('/').next().unwrap_or(&self.path)
    }

    pub fn folder(&self) -> &str {
        self.path.rsplit_once('/').map(|(dir, _)| dir).unwrap_or("")
    }
}

pub struct Thumbnail {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

#[derive(Default)]
pub struct ModelStats {
    pub nodes: usize,
    pub meshes: usize,
    pub primitives: usize,
    pub vertices: usize,
    pub triangles: usize,
    pub materials: usize,
    pub textures: usize,
    pub cameras: usize,
    pub lights: usize,
    pub animations: usize,
    pub binary: bool,
    /// Project-relative files the model reads; `None` for one that points outside the project.
    pub dependencies: Vec<(String, Option<String>)>,
}

pub enum AssetDetails {
    Texture { width: u32, height: u32, color: String },
    Model(Box<ModelStats>),
    None,
    Error(String),
}

pub struct AssetInfo {
    pub details: AssetDetails,
    pub thumbnail: Option<Arc<Thumbnail>>,
}

/// Which files the current project's build would ship.
pub enum BuildSet {
    Pending,
    Ready(HashSet<String>),
    Error(String),
}

pub struct AssetBrowserState {
    pub entries: Vec<AssetEntry>,
    /// Bumped on every rescan; the UI drops its cached textures when it changes.
    pub generation: u64,
    pub info: HashMap<String, Arc<AssetInfo>>,
    /// Paths the UI wants inspected; the plugin hands them to the worker.
    pub wanted: Vec<String>,
    pub pending: HashSet<String>,
    pub build_set: BuildSet,
    pub refresh_requested: bool,
    pub folder: String,
    pub selected: Option<String>,
    pub search: String,
    pub hidden_kinds: HashSet<AssetKind>,
}

impl Default for AssetBrowserState {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            generation: 0,
            info: HashMap::new(),
            wanted: Vec::new(),
            pending: HashSet::new(),
            build_set: BuildSet::Pending,
            refresh_requested: false,
            folder: String::new(),
            selected: None,
            search: String::new(),
            hidden_kinds: HashSet::new(),
        }
    }
}

impl AssetBrowserState {
    pub fn in_build(&self, path: &str) -> Option<bool> {
        match &self.build_set {
            BuildSet::Ready(set) => Some(set.contains(path)),
            _ => None,
        }
    }

    pub fn entry(&self, path: &str) -> Option<&AssetEntry> {
        self.entries.iter().find(|e| e.path == path)
    }
}

enum Job {
    Scan { dir: PathBuf, generation: u64 },
    Inspect { dir: PathBuf, path: String, kind: AssetKind, generation: u64 },
}

enum Done {
    Scanned { generation: u64, entries: Vec<AssetEntry>, build_set: BuildSet },
    Inspected { generation: u64, path: String, info: AssetInfo },
}

pub struct AssetWorker {
    jobs: Sender<Job>,
    done: Receiver<Done>,
}

impl AssetWorker {
    pub fn spawn() -> Self {
        let (jobs, job_rx) = mpsc::channel::<Job>();
        let (done_tx, done) = mpsc::channel();
        thread::Builder::new()
            .name("asset-browser".into())
            .spawn(move || {
                for job in job_rx {
                    let result = match job {
                        Job::Scan { dir, generation } => {
                            let entries = scan(&dir);
                            let build_set = match build_tool::collect_manifest(&dir) {
                                Ok((_, manifest)) => BuildSet::Ready(manifest.entries().map(|e| e.path.clone()).collect()),
                                Err(e) => BuildSet::Error(e.to_string()),
                            };
                            Done::Scanned { generation, entries, build_set }
                        }
                        Job::Inspect { dir, path, kind, generation } => {
                            let info = inspect(&dir, &path, kind);
                            Done::Inspected { generation, path, info }
                        }
                    };
                    if done_tx.send(result).is_err() {
                        break;
                    }
                }
            })
            .expect("spawning asset browser thread");
        Self { jobs, done }
    }

    /// Starts a rescan and forgets everything inspected so far.
    pub fn rescan(&self, state: &mut AssetBrowserState, dir: &Path) {
        state.generation += 1;
        state.info.clear();
        state.pending.clear();
        state.wanted.clear();
        state.build_set = BuildSet::Pending;
        let _ = self.jobs.send(Job::Scan { dir: dir.to_path_buf(), generation: state.generation });
    }

    /// Forwards inspection requests and applies finished results.
    pub fn sync(&self, state: &mut AssetBrowserState, dir: Option<&Path>) {
        for done in self.done.try_iter() {
            match done {
                Done::Scanned { generation, entries, build_set } if generation == state.generation => {
                    if state.selected.as_ref().is_some_and(|s| !entries.iter().any(|e| &e.path == s)) {
                        state.selected = None;
                    }
                    state.entries = entries;
                    state.build_set = build_set;
                }
                Done::Inspected { generation, path, info } if generation == state.generation => {
                    state.pending.remove(&path);
                    state.info.insert(path, Arc::new(info));
                }
                _ => {}
            }
        }

        let Some(dir) = dir else {
            state.wanted.clear();
            return;
        };
        for path in std::mem::take(&mut state.wanted) {
            if state.info.contains_key(&path) || !state.pending.insert(path.clone()) {
                continue;
            }
            let kind = state.entry(&path).map(|e| e.kind).unwrap_or(AssetKind::Other);
            let _ = self.jobs.send(Job::Inspect { dir: dir.to_path_buf(), path, kind, generation: state.generation });
        }
    }
}

fn scan(dir: &Path) -> Vec<AssetEntry> {
    let skip = build_tool::builds_dir(dir);
    let mut entries = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(read) = std::fs::read_dir(&current) else { continue };
        for item in read.flatten() {
            let path = item.path();
            let hidden = item.file_name().to_string_lossy().starts_with('.');
            let Ok(meta) = item.metadata() else { continue };
            if hidden || path == skip {
                continue;
            }
            if meta.is_dir() {
                stack.push(path);
            } else if let Some(rel) = Project::relative_path(dir, &path) {
                entries.push(AssetEntry { kind: AssetKind::from_path(&rel), path: rel, size: meta.len() });
            }
        }
    }
    entries.sort_by(|a, b| a.path.cmp(&b.path));
    entries
}

fn inspect(dir: &Path, path: &str, kind: AssetKind) -> AssetInfo {
    let bytes = match std::fs::read(dir.join(path)) {
        Ok(b) => b,
        Err(e) => return AssetInfo { details: AssetDetails::Error(e.to_string()), thumbnail: None },
    };
    match kind {
        AssetKind::Texture => match image::load_from_memory(&bytes) {
            Ok(img) => AssetInfo {
                details: AssetDetails::Texture {
                    width: img.width(),
                    height: img.height(),
                    color: format!("{:?}", img.color()),
                },
                thumbnail: Some(Arc::new(thumbnail(&img))),
            },
            Err(e) => AssetInfo { details: AssetDetails::Error(e.to_string()), thumbnail: None },
        },
        AssetKind::Scene | AssetKind::Model => match Gltf::from_slice(&bytes) {
            Ok(gltf) => {
                let preview = model_preview(&gltf, dir, path);
                AssetInfo { details: AssetDetails::Model(Box::new(model_stats(&gltf, path))), thumbnail: preview }
            }
            Err(e) => AssetInfo { details: AssetDetails::Error(e.to_string()), thumbnail: None },
        },
        _ => AssetInfo { details: AssetDetails::None, thumbnail: None },
    }
}

fn thumbnail(img: &image::DynamicImage) -> Thumbnail {
    let scale = THUMBNAIL_SIZE as f32 / img.width().max(img.height()).max(1) as f32;
    let (w, h) = if scale < 1.0 {
        (((img.width() as f32 * scale) as u32).max(1), ((img.height() as f32 * scale) as u32).max(1))
    } else {
        (img.width(), img.height())
    };
    let rgba = image::imageops::thumbnail(&img.to_rgba8(), w, h);
    Thumbnail { width: rgba.width(), height: rgba.height(), rgba: rgba.into_raw() }
}

fn model_stats(gltf: &Gltf, path: &str) -> ModelStats {
    let doc = &gltf.document;
    let mut stats = ModelStats {
        nodes: doc.nodes().len(),
        meshes: doc.meshes().len(),
        materials: doc.materials().len(),
        textures: doc.textures().len(),
        cameras: doc.cameras().len(),
        lights: doc.lights().map_or(0, |l| l.len()),
        animations: doc.animations().len(),
        binary: gltf.blob.is_some(),
        ..ModelStats::default()
    };
    for primitive in doc.meshes().flat_map(|m| m.primitives()) {
        stats.primitives += 1;
        let vertices = primitive.get(&gltf_api::Semantic::Positions).map_or(0, |a| a.count());
        stats.vertices += vertices;
        if primitive.mode() == gltf_api::mesh::Mode::Triangles {
            stats.triangles += primitive.indices().map_or(vertices, |a| a.count()) / 3;
        }
    }

    let buffers = doc.buffers().filter_map(|b| match b.source() {
        gltf_api::buffer::Source::Uri(uri) => external_uri(uri),
        gltf_api::buffer::Source::Bin => None,
    });
    let images = doc.images().filter_map(|i| match i.source() {
        gltf_api::image::Source::Uri { uri, .. } => external_uri(uri),
        gltf_api::image::Source::View { .. } => None,
    });
    stats.dependencies = buffers
        .chain(images)
        .map(|uri| {
            let resolved = to_logical(&resolve_relative(Path::new(path), &uri));
            (uri.into_owned(), resolved)
        })
        .collect();
    stats
}

/// Thumbnail of the first material's base color texture, or of the first image.
fn model_preview(gltf: &Gltf, dir: &Path, path: &str) -> Option<Arc<Thumbnail>> {
    let doc = &gltf.document;
    let image = doc
        .materials()
        .find_map(|m| m.pbr_metallic_roughness().base_color_texture())
        .map(|info| info.texture().source())
        .or_else(|| doc.images().next())?;

    let read_relative = |uri: &str| {
        let rel = to_logical(&resolve_relative(Path::new(path), &external_uri(uri)?))?;
        std::fs::read(dir.join(rel)).ok()
    };
    let encoded = match image.source() {
        gltf_api::image::Source::Uri { uri, .. } => read_relative(uri)?,
        gltf_api::image::Source::View { view, .. } => {
            let buffer = match view.buffer().source() {
                gltf_api::buffer::Source::Bin => gltf.blob.clone()?,
                gltf_api::buffer::Source::Uri(uri) => read_relative(uri)?,
            };
            buffer.get(view.offset()..view.offset() + view.length())?.to_vec()
        }
    };
    image::load_from_memory(&encoded).ok().map(|img| Arc::new(thumbnail(&img)))
}
