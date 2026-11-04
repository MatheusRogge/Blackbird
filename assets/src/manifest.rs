use std::{collections::BTreeMap, fmt::Write as _};

use thiserror::Error;

const HEADER: &str = "blackbird-manifest 1";

#[derive(Error, Debug)]
#[error("manifest line {line}: {message}")]
pub struct ManifestParseError {
    pub line: usize,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestEntry {
    /// Logical, `/`-separated path relative to the project root.
    pub path: String,
    pub size: u64,
    pub hash: u64,
}

/// A manifest entry baked into a binary by `codegen::generate`.
#[derive(Debug, Clone, Copy)]
pub struct StaticAsset {
    pub path: &'static str,
    pub size: u64,
    pub hash: u64,
}

/// Every asset a build may load. Produced by `AssetManager::collect` from a set of roots.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AssetManifest {
    entries: BTreeMap<String, ManifestEntry>,
}

impl AssetManifest {
    pub fn from_static(assets: &[StaticAsset]) -> Self {
        assets
            .iter()
            .map(|a| ManifestEntry { path: a.path.to_string(), size: a.size, hash: a.hash })
            .collect()
    }

    pub fn insert(&mut self, entry: ManifestEntry) {
        self.entries.insert(entry.path.clone(), entry);
    }

    pub fn get(&self, path: &str) -> Option<&ManifestEntry> {
        self.entries.get(path)
    }

    pub fn contains(&self, path: &str) -> bool {
        self.entries.contains_key(path)
    }

    /// Entries sorted by path.
    pub fn entries(&self) -> impl Iterator<Item = &ManifestEntry> {
        self.entries.values()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn total_size(&self) -> u64 {
        self.entries.values().map(|e| e.size).sum()
    }

    /// One `<hash> <size> <path>` line per entry; the path goes last so it may contain spaces.
    pub fn to_text(&self) -> String {
        let mut out = format!("{HEADER}\n");
        for e in self.entries.values() {
            let _ = writeln!(out, "{:016x} {} {}", e.hash, e.size, e.path);
        }
        out
    }

    pub fn parse(text: &str) -> Result<Self, ManifestParseError> {
        let err = |line: usize, message: &str| ManifestParseError { line, message: message.to_string() };
        let mut lines = text.lines().enumerate();
        match lines.next() {
            Some((_, header)) if header.trim() == HEADER => {}
            _ => return Err(err(1, "missing header")),
        }

        let mut manifest = Self::default();
        for (idx, line) in lines {
            if line.trim().is_empty() {
                continue;
            }
            let mut parts = line.splitn(3, ' ');
            let (Some(hash), Some(size), Some(path)) = (parts.next(), parts.next(), parts.next()) else {
                return Err(err(idx + 1, "expected `<hash> <size> <path>`"));
            };
            manifest.insert(ManifestEntry {
                path: path.to_string(),
                size: size.parse().map_err(|_| err(idx + 1, "invalid size"))?,
                hash: u64::from_str_radix(hash, 16).map_err(|_| err(idx + 1, "invalid hash"))?,
            });
        }
        Ok(manifest)
    }
}

impl FromIterator<ManifestEntry> for AssetManifest {
    fn from_iter<I: IntoIterator<Item = ManifestEntry>>(iter: I) -> Self {
        let mut manifest = Self::default();
        for entry in iter {
            manifest.insert(entry);
        }
        manifest
    }
}

/// FNV-1a. Detects changed or corrupted content; not a cryptographic hash.
pub fn content_hash(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_round_trip() {
        let manifest: AssetManifest = [
            ManifestEntry { path: "scenes/my scene.gltf".into(), size: 12, hash: 0xdead },
            ManifestEntry { path: "assets/a.png".into(), size: 3, hash: 1 },
        ]
        .into_iter()
        .collect();
        assert_eq!(AssetManifest::parse(&manifest.to_text()).unwrap(), manifest);
    }
}
