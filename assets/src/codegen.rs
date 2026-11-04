//! Rust source for a build script, so a binary knows every asset it ships at compile time.

use std::{collections::HashSet, fmt::Write as _};

use crate::manifest::AssetManifest;

/// Generates a module body with:
/// - `paths::<NAME>: &str` for every asset (`scenes/main.gltf` → `SCENES_MAIN_GLTF`),
/// - `ASSETS: &[StaticAsset]` listing all of them with size and hash,
/// - `PACK_ID: Option<u64>`, the content id of the pack the binary expects.
///
/// `include!` it from a module that has `assets::manifest::StaticAsset` in scope.
pub fn generate(manifest: &AssetManifest, pack_id: Option<u64>) -> String {
    let mut used = HashSet::new();
    let names: Vec<String> = manifest
        .entries()
        .map(|e| {
            let base = const_name(&e.path);
            let mut name = base.clone();
            let mut n = 2;
            while !used.insert(name.clone()) {
                name = format!("{base}_{n}");
                n += 1;
            }
            name
        })
        .collect();

    let mut out = String::from("pub mod paths {\n");
    for (entry, name) in manifest.entries().zip(&names) {
        let _ = writeln!(out, "    pub const {name}: &str = {:?};", entry.path);
    }
    out.push_str("}\n\npub const ASSETS: &[StaticAsset] = &[\n");
    for (entry, name) in manifest.entries().zip(&names) {
        let _ = writeln!(
            out,
            "    StaticAsset {{ path: paths::{name}, size: {}, hash: 0x{:016x} }},",
            entry.size, entry.hash
        );
    }
    out.push_str("];\n\n");
    let _ = match pack_id {
        Some(id) => writeln!(out, "pub const PACK_ID: Option<u64> = Some(0x{id:016x});"),
        None => writeln!(out, "pub const PACK_ID: Option<u64> = None;"),
    };
    out
}

fn const_name(path: &str) -> String {
    let mut name: String = path
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_uppercase() } else { '_' })
        .collect();
    if name.starts_with(|c: char| c.is_ascii_digit()) || name.is_empty() {
        name.insert(0, '_');
    }
    name
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::ManifestEntry;

    #[test]
    fn disambiguates_colliding_names() {
        let manifest: AssetManifest = ["a-b.png", "a_b.png", "1.bin"]
            .into_iter()
            .map(|p| ManifestEntry { path: p.into(), size: 1, hash: 2 })
            .collect();
        let code = generate(&manifest, Some(7));
        assert!(code.contains("pub const _1_BIN: &str = \"1.bin\";"));
        assert!(code.contains("pub const A_B_PNG: &str"));
        assert!(code.contains("pub const A_B_PNG_2: &str"));
        assert!(code.contains("Some(0x0000000000000007)"));
    }
}
