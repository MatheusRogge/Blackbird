use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const MAX_ENTRIES: usize = 10;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentEntry {
    pub name: String,
    pub path: PathBuf,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecentProjects {
    pub entries: Vec<RecentEntry>,
}

impl RecentProjects {
    pub fn load() -> Self {
        let path = match config_path() {
            Some(p) => p,
            None => return Self::default(),
        };
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => return Self::default(),
        };
        serde_json::from_slice(&bytes).unwrap_or_default()
    }

    pub fn push(&mut self, name: &str, project_dir: &Path) {
        self.entries.retain(|e| e.path != project_dir);
        self.entries.insert(0, RecentEntry {
            name: name.to_string(),
            path: project_dir.to_path_buf(),
        });
        self.entries.truncate(MAX_ENTRIES);
    }

    pub fn save(&self) -> Result<()> {
        let path = config_path().context("could not resolve config dir")?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).context("creating config dir")?;
        }
        let json = serde_json::to_string_pretty(self).context("serializing recents")?;
        std::fs::write(&path, json).context("writing recent_projects.json")
    }
}

fn config_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))?;
    Some(PathBuf::from(home).join(".blackbird").join("recent_projects.json"))
}
