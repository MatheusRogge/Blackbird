use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use gltf_api::buffer::{Data, Source};

use crate::project::Project;

/// Copies a glTF/GLB scene and everything it references into the project.
///
/// The scene is rewritten as `scenes/<name>.gltf`, every buffer is written to
/// `assets/models/<name>/` and every image (external, data URI or embedded in a
/// buffer view) to `assets/textures/<name>/`. Returns the imported scene path.
pub fn import_scene(source: &Path, project_dir: &Path) -> Result<PathBuf> {
    let bytes = std::fs::read(source).with_context(|| format!("reading {}", source.display()))?;
    let gltf = gltf_api::Gltf::from_slice(&bytes).context("parsing glTF")?;
    let base = source.parent().unwrap_or(Path::new("."));
    let buffers = gltf_api::import_buffers(&gltf.document, Some(base), gltf.blob)
        .context("reading buffers")?;
    let mut root = gltf.document.as_json().clone();

    let stem = sanitize(&source.file_stem().unwrap_or_default().to_string_lossy());
    let scenes_dir = Project::scenes_dir(project_dir);
    let name = unique_name(&stem, |n| {
        scenes_dir.join(format!("{n}.gltf")).exists()
            || Project::models_dir(project_dir).join(n).exists()
            || Project::textures_dir(project_dir).join(n).exists()
    });
    let models_dir = Project::models_dir(project_dir).join(&name);
    let textures_dir = Project::textures_dir(project_dir).join(&name);
    std::fs::create_dir_all(&scenes_dir).context("creating scenes dir")?;
    std::fs::create_dir_all(&models_dir).context("creating models dir")?;
    std::fs::create_dir_all(&textures_dir).context("creating textures dir")?;

    let mut taken = HashSet::new();
    for (i, (buffer, data)) in root.buffers.iter_mut().zip(&buffers).enumerate() {
        let file = file_name(buffer.uri.as_deref(), &format!("{name}_{i}"), "bin", &mut taken);
        std::fs::write(models_dir.join(&file), &data.0)
            .with_context(|| format!("writing buffer {file}"))?;
        buffer.uri = Some(format!("../assets/models/{name}/{file}"));
    }

    let mut taken = HashSet::new();
    for i in 0..root.images.len() {
        let image = &root.images[i];
        let mime = image.mime_type.as_ref().map(|m| m.0.as_str());
        let fallback = format!("{name}_{i}");
        let (data, file) = match (&image.uri, &image.buffer_view) {
            (Some(uri), _) => {
                let data = Data::from_source(Source::Uri(uri), Some(base))
                    .with_context(|| format!("reading image {uri}"))?
                    .0;
                let uri = (!uri.starts_with("data:")).then_some(uri.as_str());
                let ext = extension(mime, &data);
                (data, file_name(uri, &fallback, ext, &mut taken))
            }
            (None, Some(view)) => {
                let view = root.buffer_views.get(view.value()).context("image buffer view out of range")?;
                let buffer = &buffers[view.buffer.value()];
                let start = view.byte_offset.map_or(0, |o| o.0 as usize);
                let end = start + view.byte_length.0 as usize;
                let data = buffer.get(start..end).context("image buffer view out of bounds")?.to_vec();
                let ext = extension(mime, &data);
                (data, file_name(None, &fallback, ext, &mut taken))
            }
            (None, None) => bail!("image {i} has no source"),
        };
        std::fs::write(textures_dir.join(&file), data).with_context(|| format!("writing image {file}"))?;

        let image = &mut root.images[i];
        image.uri = Some(format!("../assets/textures/{name}/{file}"));
        image.buffer_view = None;
    }

    let scene_path = scenes_dir.join(format!("{name}.gltf"));
    let json = serde_json::to_string_pretty(&root).context("serializing scene")?;
    std::fs::write(&scene_path, json).context("writing scene")?;
    Ok(scene_path)
}

fn file_name(uri: Option<&str>, fallback: &str, ext: &str, taken: &mut HashSet<String>) -> String {
    let original = uri
        .and_then(|u| Path::new(u).file_name())
        .map(|n| sanitize(&n.to_string_lossy()))
        .filter(|n| !n.is_empty());
    let name = original.unwrap_or_else(|| format!("{fallback}.{ext}"));
    let (stem, ext) = match name.rsplit_once('.') {
        Some((s, e)) => (s.to_string(), format!(".{e}")),
        None => (name.clone(), String::new()),
    };
    let unique = unique_name(&stem, |s| taken.contains(&format!("{s}{ext}")));
    let file = format!("{unique}{ext}");
    taken.insert(file.clone());
    file
}

fn unique_name(stem: &str, exists: impl Fn(&str) -> bool) -> String {
    let mut name = stem.to_string();
    let mut n = 1;
    while exists(&name) {
        name = format!("{stem}_{n}");
        n += 1;
    }
    name
}

/// Keeps names URI-safe so rewritten references need no percent-encoding.
fn sanitize(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') { c } else { '_' })
        .collect();
    if s.is_empty() { "scene".to_string() } else { s }
}

fn extension(mime: Option<&str>, data: &[u8]) -> &'static str {
    match mime {
        Some("image/png") => "png",
        Some("image/jpeg") => "jpg",
        Some("image/webp") => "webp",
        Some("image/ktx2") => "ktx2",
        _ if data.starts_with(b"\x89PNG") => "png",
        _ if data.starts_with(b"\xFF\xD8") => "jpg",
        _ => "bin",
    }
}
