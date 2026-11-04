use std::{
    fs, io,
    path::{Path, PathBuf},
};

/// Where asset bytes come from: loose files on disk or a `.bbpak`.
pub trait AssetSource: Send + Sync {
    fn read(&self, path: &Path) -> io::Result<Vec<u8>>;

    /// Human-readable description for logs and dev tools.
    fn describe(&self) -> String;
}

/// Loose files under `root`. Absolute paths bypass the root, which the editor relies on to open
/// scenes from anywhere.
pub struct DirectorySource {
    root: PathBuf,
}

impl DirectorySource {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Rooted at the directory holding the running executable.
    pub fn beside_executable() -> Self {
        let root = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(Path::to_path_buf))
            .unwrap_or_default();
        Self::new(root)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

impl AssetSource for DirectorySource {
    fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        fs::read(self.root.join(path))
    }

    fn describe(&self) -> String {
        format!("directory {}", self.root.display())
    }
}
