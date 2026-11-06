# sandbox

Demo binaries and the **canonical location of the PBR WGSL shaders**. Depends on `engine` with `full, debug` and `rendering` with `rendering_2d`.

## Binaries

| Bin | Source | What it shows |
|-----|--------|---------------|
| `sandbox` | `src/main.rs` | Full PBR/DDGI pipeline. Async-loads a glTF (absolute path hard-coded per OS; edit it locally), spawns a `Camera` + `SkyLight`, WASD (`PlayerController`) + mouse orbit (`CameraController`, first-person), streams the scene with `GltfScene::load_batch(world, 10)`. |
| `capture` | `src/bin/capture.rs` | Headless screenshot tool: loads a glTF like the player (lights, all meshes, only the main camera), renders `--frames` frames (default 400, so GI converges) with `Renderer::new_headless`, writes a PNG. `capture <asset-root> <scene> <out.png> [--size WxH] [--frames N] [--eye x,y,z] [--target x,y,z] [--exposure E] [--no-specular\|--no-reflections\|--no-ssr\|--no-reflection-probe\|--no-emissive-lights\|--no-indirect\|--no-direct\|--no-shadows] [--view lit\|albedo\|normal\|material\|emissive\|shadows] [--reloads N] [--point x,y,z,I[,r]] [--stats]`. `--stats` prints per-pass timings every frame. Under WSL the only Vulkan device is usually llvmpipe (CPU), so GPU-heavy passes cost far more here than on real hardware; compare passes relative to each other. `--reloads N` unloads and reloads the scene N times first, in the same `World`, to reproduce editor hot-reload bugs. Reads shaders from `shaders/` at runtime, so shader edits don't need a rebuild. |
| `sprites` | `src/bin/sprites.rs` | 2D path: `SpriteRenderGraphBuilder`, `Camera2D`, procedurally generated `SpriteTexture`s, z-ordering, rotation. |

Run with `cargo run -p sandbox` or `cargo run -p sandbox --bin sprites`.

## Shaders (`shaders/`)
Embedded at compile time with `include_str!` and passed to `RenderGraphPBRBuilder::new` in this order: `gbuffer`, `cluster_assignment`, `lighting`, `present`, `shadow`, `probe_trace`, `probe_update`, `debug_overlay`, `ssr`, `composite`.

`editor` and `player` also `include_str!` these same files through `../../sandbox/shaders/…`, and `capture` reads them at runtime, so renaming or moving a shader breaks all of them.

- There is **no include/preprocessor system**. Shared code (octahedral encoding, `ProbeGridParams`, `CameraUniform`, …) is copy-pasted into every shader that needs it. `probe_helpers.wgsl` is not referenced by anything.
- `lighting.wgsl` is also the forward shader for blended meshes: the `transparent` pass reuses the same `ShaderAsset` with its `vs_forward` / `fs_forward` / `fs_background` / `fs_scene_copy` / `fs_scene_downsample` entry points, so there is no separate transparent shader. The `reflection_probe` pass reuses it too (`vs_probe` / `fs_probe_depth` / `fs_probe` / `vs_probe_sky` / `fs_probe_sky`); its prefilter shader lives in `rendering/src/pass/reflection_probe.wgsl`.
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
