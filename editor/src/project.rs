use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::state::CameraSettings;

const MANIFEST_FILE: &str = "project.json";
const SCENES_DIR: &str = "scenes";
const ASSETS_DIR: &str = "assets";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Project {
    pub name: String,
    pub version: String,
    /// Scene opened with the project, relative to the project dir with `/` separators.
    pub default_scene: Option<String>,
    #[serde(default)]
    pub camera: CameraSettings,
}

impl Project {
    pub fn create(dir: &Path, name: &str) -> Result<Self> {
        std::fs::create_dir_all(dir.join(SCENES_DIR))
            .context("creating scenes dir")?;
        std::fs::create_dir_all(Self::textures_dir(dir))
            .context("creating assets/textures dir")?;
        std::fs::create_dir_all(Self::models_dir(dir))
            .context("creating assets/models dir")?;

        let project = Self {
            name: name.to_string(),
            version: "0.1.0".to_string(),
            default_scene: None,
            camera: CameraSettings::default(),
        };
        project.save(dir)?;
        Ok(project)
    }

    pub fn load(dir: &Path) -> Result<Self> {
        let bytes = std::fs::read(dir.join(MANIFEST_FILE))
            .context("reading project.json")?;
        serde_json::from_slice(&bytes).context("parsing project.json")
    }

    pub fn save(&self, dir: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self).context("serializing project")?;
        std::fs::write(dir.join(MANIFEST_FILE), json).context("writing project.json")
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
