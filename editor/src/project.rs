use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const MANIFEST_FILE: &str = "project.json";
const SCENES_DIR: &str = "scenes";
const ASSETS_DIR: &str = "assets";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub name: String,
    pub version: String,
    pub default_scene: Option<String>,
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
}
