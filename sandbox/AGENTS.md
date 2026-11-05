# sandbox

Demo binaries and the **canonical location of the PBR WGSL shaders**. Depends on `engine` with `full, debug` and `rendering` with `rendering_2d`.

## Binaries

| Bin | Source | What it shows |
|-----|--------|---------------|
| `sandbox` | `src/main.rs` | Full PBR/DDGI pipeline. Async-loads a glTF (absolute path hard-coded per OS; edit it locally), spawns a `Camera` + `SkyLight`, WASD (`PlayerController`) + mouse orbit (`CameraController`, first-person), streams the scene with `GltfScene::load_batch(world, 10)`. |
| `sprites` | `src/bin/sprites.rs` | 2D path: `SpriteRenderGraphBuilder`, `Camera2D`, procedurally generated `SpriteTexture`s, z-ordering, rotation. |

Run with `cargo run -p sandbox` or `cargo run -p sandbox --bin sprites`.

## Shaders (`shaders/`)
Embedded at compile time with `include_str!` and passed to `RenderGraphPBRBuilder::new` in this order: `gbuffer`, `cluster_assignment`, `lighting`, `present`, `shadow`, `probe_trace`, `probe_update`, `debug_overlay`, `ssr`, `composite`.

`editor` and `player` also `include_str!` these same files through `../../sandbox/shaders/…`, so renaming or moving a shader breaks all of them.

- There is **no include/preprocessor system**. Shared code (octahedral encoding, `ProbeGridParams`, `CameraUniform`, …) is copy-pasted into every shader that needs it. `probe_helpers.wgsl` is not referenced by anything.
- `sdf_voxelize.wgsl` belongs to the unused `SdfVoxelizePass` (see `rendering/AGENTS.md`).
- Struct layouts and constants in WGSL must match the Rust `#[repr(C)]` structs and `const`s in `rendering/src/pass/*` (e.g. `PROBES_PER_ROW = 30`, `PROBES_PER_COL = 29`, `ProbeGridParams`, `ClusterParams`, `ShadowParams`, `DebugFlagsGpu`). Change both sides together.
- Bind group/binding indices are defined in the Rust pass that owns the pipeline. Look there before you renumber anything.

## Pattern to copy for a new demo
```rust
struct Game { /* state */ }
impl Plugin for Game {
    fn setup(&mut self, engine: &mut Engine) -> Result<(), EnginePluginError> { /* spawn entities */ Ok(()) }
    fn tick(&mut self, engine: &mut Engine, delta: f32) { /* poll AssetLoadHandle, controllers */ }
}
WindowedApplication::new(builder).add_plugin(Game::default()).run()?;
```
