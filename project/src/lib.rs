use std::{
    io,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const MANIFEST_FILE: &str = "project.json";
/// Pack a production build reads, next to its executable.
pub const PACK_FILE: &str = "data.bbpak";
/// Loose assets a development build reads, next to its executable.
pub const DATA_DIR: &str = "data";
const SCENES_DIR: &str = "scenes";
const ASSETS_DIR: &str = "assets";

#[derive(Error, Debug)]
pub enum ProjectError {
    #[error("{action}: {source}")]
    Io { action: &'static str, source: io::Error },

    #[error("invalid project.json: {0}")]
    Json(#[from] serde_json::Error),

    #[error("project has no default scene; set one in the editor's Project menu")]
    NoDefaultScene,
}

fn io_err(action: &'static str) -> impl FnOnce(io::Error) -> ProjectError {
    move |source| ProjectError::Io { action, source }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CameraSettings {
    /// World units per second for WASD/QE movement.
    pub move_speed: f32,
    /// Speed multiplier while Shift is held.
    pub fast_multiplier: f32,
    /// Exposure compensation in stops; the camera exposure is `2^exposure_stops`.
    pub exposure_stops: f32,
}

impl CameraSettings {
    pub const MIN_SPEED: f32 = 0.01;
    pub const MAX_SPEED: f32 = 1000.0;

    pub fn exposure(&self) -> f32 {
        self.exposure_stops.exp2()
    }
}

impl Default for CameraSettings {
    fn default() -> Self {
        Self { move_speed: 10.0, fast_multiplier: 4.0, exposure_stops: 0.0 }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BuildSettings {
    /// Project-relative assets to ship even though no scene references them.
    pub include: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Project {
    pub name: String,
    pub version: String,
    /// Scene opened with the project, relative to the project dir with `/` separators.
    pub default_scene: Option<String>,
    #[serde(default)]
    pub camera: CameraSettings,
    #[serde(default)]
    pub build: BuildSettings,
}

impl Project {
    pub fn create(dir: &Path, name: &str) -> Result<Self, ProjectError> {
        std::fs::create_dir_all(dir.join(SCENES_DIR)).map_err(io_err("creating scenes dir"))?;
        std::fs::create_dir_all(Self::textures_dir(dir)).map_err(io_err("creating assets/textures dir"))?;
        std::fs::create_dir_all(Self::models_dir(dir)).map_err(io_err("creating assets/models dir"))?;

        let project = Self {
            name: name.to_string(),
            version: "0.1.0".to_string(),
            default_scene: None,
            camera: CameraSettings::default(),
            build: BuildSettings::default(),
        };
        project.save(dir)?;
        Ok(project)
    }

    pub fn load(dir: &Path) -> Result<Self, ProjectError> {
        let bytes = std::fs::read(dir.join(MANIFEST_FILE)).map_err(io_err("reading project.json"))?;
        Self::parse(&bytes)
    }

    pub fn parse(bytes: &[u8]) -> Result<Self, ProjectError> {
        Ok(serde_json::from_slice(bytes)?)
    }

    pub fn save(&self, dir: &Path) -> Result<(), ProjectError> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(dir.join(MANIFEST_FILE), json).map_err(io_err("writing project.json"))
    }

    /// Assets a build starts its dependency walk from.
    pub fn build_roots(&self) -> Result<Vec<String>, ProjectError> {
        let scene = self.default_scene.clone().ok_or(ProjectError::NoDefaultScene)?;
        let mut roots = vec![MANIFEST_FILE.to_string(), scene];
        roots.extend(self.build.include.iter().cloned());
        Ok(roots)
    }

    /// File-name-safe form of the project name, for build outputs.
    pub fn slug(&self) -> String {
        let slug: String = self
            .name
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
            .collect();
        if slug.is_empty() { "game".to_string() } else { slug }
    }

    pub fn scenes_dir(dir: &Path) -> PathBuf {
        dir.join(SCENES_DIR)
    }

    pub fn models_dir(dir: &Path) -> PathBuf {
        dir.join(ASSETS_DIR).join("models")
    }

    pub fn textures_dir(dir: &Path) -> PathBuf {
        dir.join(ASSETS_DIR).join("textures")
    }

    pub fn default_scene_path(&self, project_dir: &Path) -> Option<PathBuf> {
        self.default_scene.as_ref().map(|s| project_dir.join(s))
    }

    /// `path` relative to the project dir in the form stored in `default_scene`,
    /// or `None` if it lies outside the project.
    pub fn relative_path(project_dir: &Path, path: &Path) -> Option<String> {
        let rel = path.strip_prefix(project_dir).ok()?;
        let parts: Option<Vec<&str>> = rel.components().map(|c| c.as_os_str().to_str()).collect();
        Some(parts?.join("/"))
    }

    /// Every `.gltf`/`.glb` in the scenes dir, as project-relative paths, sorted.
    pub fn list_scenes(project_dir: &Path) -> Vec<String> {
        let Ok(entries) = std::fs::read_dir(Self::scenes_dir(project_dir)) else {
            return Vec::new();
        };
        let mut scenes: Vec<String> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("gltf") || e.eq_ignore_ascii_case("glb")))
            .filter_map(|p| Self::relative_path(project_dir, &p))
            .collect();
        scenes.sort();
        scenes
    }
}
