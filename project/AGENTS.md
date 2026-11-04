# project

The `project.json` manifest shared by `editor`, `player` and `build_tool`. Depends only on `serde`.

```rust
Project { name, version, default_scene: Option<String>, camera: CameraSettings, build: BuildSettings }
Project::create(dir, name) / load(dir) / save(dir) / parse(&[u8])
Project::build_roots() -> Result<Vec<String>, ProjectError>   // project.json + default scene + build.include
Project::relative_path(project_dir, path) / list_scenes(project_dir)
CameraSettings { move_speed, fast_multiplier, exposure_stops }   // exposure() = 2^stops
BuildSettings { include: Vec<String> }
```

- Every project path is relative to the project dir with `/` separators. That matches the logical paths used by `assets` manifests and packs, so a project path can be loaded unchanged from loose files or a `.bbpak`.
- `camera` and `build` are `#[serde(default)]`, so older manifests still load. Add new sections the same way.
- `build.include` lists assets the game loads by path at runtime that no scene references. Everything else ships only if a root reaches it through resolver dependencies.
