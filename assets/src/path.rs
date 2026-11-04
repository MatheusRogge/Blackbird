use std::path::{Component, Path, PathBuf};

/// Lexically resolves `.` and `..`. A `..` that would climb above the start is kept.
pub fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if matches!(out.components().next_back(), Some(Component::Normal(_))) {
                    out.pop();
                } else if !out.has_root() {
                    out.push("..");
                }
            }
            other => out.push(other),
        }
    }
    out
}

/// `relative` resolved against the directory containing `asset`.
pub fn resolve_relative(asset: &Path, relative: &str) -> PathBuf {
    let base = asset.parent().unwrap_or(Path::new(""));
    normalize(&base.join(relative))
}

/// The `/`-separated form used by manifests and packs, or `None` if `path` is absolute or
/// climbs out of its root.
pub fn to_logical(path: &Path) -> Option<String> {
    let normalized = normalize(path);
    let parts: Option<Vec<&str>> = normalized
        .components()
        .map(|c| match c {
            Component::Normal(part) => part.to_str(),
            _ => None,
        })
        .collect();
    parts.filter(|p| !p.is_empty()).map(|p| p.join("/"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_sibling_directories() {
        let resolved = resolve_relative(Path::new("scenes/main.gltf"), "../assets/models/a.bin");
        assert_eq!(to_logical(&resolved).as_deref(), Some("assets/models/a.bin"));
    }

    #[test]
    fn rejects_paths_outside_the_root() {
        assert_eq!(to_logical(Path::new("../secret.txt")), None);
        assert_eq!(to_logical(Path::new("/etc/passwd")), None);
    }
}
