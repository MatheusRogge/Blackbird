# Blackbird

Rust game engine on wgpu (Vulkan backend only). Cargo workspace. Each crate sets edition 2024 and `rust-version = 1.99` in its own manifest (there is no `[workspace.package]`).

Each member crate has its own `AGENTS.md` with architecture, public API and gotchas. Read the one for the crate you are touching.

## Crate map

| Crate dir | Package name | Role |
|-----------|--------------|------|
| `core/` | `engine_core` | `World`, `Entity`, `InputEvent`/`InputBuffer`. No GPU, no windowing. |
| `threading/` | `threading` | `Executor` trait + `DefaultExecutor` (one worker thread + named threads). |
| `assets/` | `assets` | `AssetManager`, extension-keyed `AssetResolver`s, `AssetSource` (dir / `.bbpak`), dependency walk, manifests, codegen. |
| `rendering/` | `rendering` | wgpu `Renderer`, `RenderGraph`, all passes, PBR + DDGI pipeline, 2D sprites. |
| `gltf/` | `gltf` | glTF/GLB resolver, `GltfScene` incremental loader into `World`. |
| `ui/` | `ui` | egui integration: `UiLayer` trait, `EguiRenderer`. |
| `engine/` | `engine` | Facade: `Engine`, `Plugin`, `HeadlessApplication`, `WindowedApplication`. Feature-gated re-exports. |
| `sandbox/` | `sandbox` | Demo binaries (`sandbox` PBR scene, `sprites` 2D). Owns the WGSL shaders. |
| `editor/` | `editor` | egui scene editor binary (projects, glTF load/hot-reload, lights, export, asset browser, Build menu). |
| `project/` | `project` | `project.json` model (`Project`, `CameraSettings`, `BuildSettings`) and build output layout consts. |
| `build_tool/` | `build_tool` | Library + `blackbird` CLI: collects a project's assets, packs them, compiles `player`. |
| `player/` | `player` | Standalone game runtime for a project; dev and production flavours. |

`application/` and `window/` are **not workspace members**. They are stale leftovers from before the `engine` facade (they use an old `InputEvent` shape and do not compile against current code). Do not add code there. Delete them or ignore them.

## Dependency graph

```
core ◄──────────────┬──────────┬────────────┐
threading ◄── assets ◄──┐       │            │
                     rendering ─┤ (debug feat → ui)
                        ▲       ui
                        gltf    ▲
                        ▲       │
                  engine (features pull in assets/gltf/rendering/ui)
                        ▲
          sandbox, player, build_tool ◄── editor
                (project is shared by editor, player, build_tool)
```

`player/build.rs` depends on `assets` with `default-features = false` (no zstd) only to parse the manifest and generate code.

`rendering` depends on `core` and `assets`. It does not depend on `engine`. Keep it that way, and don't make lower crates depend on `engine`.

## Runtime model (windowed)

Three threads, set up in `engine/src/windowed.rs`:

- **main**: winit event loop. Converts winit events into `engine_core::input::InputEvent` and batches them. On `RedrawRequested` it `try_send`s `GameMsg::Tick { inputs }`, so if the game thread is busy the frame is skipped.
- **game**: owns `Engine` and all `Plugin`s. Each tick: push inputs into `InputBuffer`, call `World::tick_all(delta)`, then `Plugin::tick` in registration order, then send the input batch on to the render thread.
- **render**: owns `Renderer`. Takes a **read** lock on `Arc<RwLock<World>>`, runs the render graph and the egui overlay, publishes `FrameStats`, and calls `window.request_redraw()`, which drives the next tick.

Channels are `sync_channel(1)`, so game and render run in lockstep with at most one frame in flight. A render pass that holds the world read lock blocks the game thread's write lock.

## Build / check

```bash
cargo check --workspace
cargo clippy --workspace          # workspace lints: clippy correctness=deny, others warn
cargo run -p sandbox              # PBR demo (hard-coded glTF path in sandbox/src/main.rs)
cargo run -p sandbox --bin sprites
cargo run -p editor
cargo test --workspace            # assets has unit tests (paths, manifest, pack, dependency walk)

cargo run -p build_tool -- build <project-dir>                # development build → <project>/builds/development
cargo run -p build_tool -- build <project-dir> --production   # production build → <project>/builds/production
cargo run -p player -- <project-dir>                          # run a project without building
```

## Standalone builds

```
project.json + default_scene + build.include
        │  AssetManager::collect (resolver dependencies, e.g. glTF → .bin/.png)
        ▼
   AssetManifest ──► development: <out>/data/** (loose)        ┐
        │        └─► production:  <out>/data.bbpak (zstd pack) ├─► cargo build -p player --profile <kind>
        └─ BLACKBIRD_ASSET_MANIFEST / BLACKBIRD_PACK_ID ───────┘     build.rs bakes manifest + pack id in
```

Only assets reachable from the roots ship. A resolver that reads extra files must report them from `AssetResolver::dependencies`, or production builds will be missing them.

Only `assets` has tests so far. CI (`.github/workflows/rust.yml`) runs `cargo build` and `cargo test` on `master`.

## Conventions

- Don't write comments unless the *why* is non-obvious.
- Delete dead code. Don't silence it with `#[allow(dead_code)]`.
- Commit messages are a single conventional-commit line (`feat: …`, `fix: …`, `chore: …`).
- Errors use `thiserror` in libraries and `anyhow` in binaries.
- Math uses `ultraviolet` (`Vec3`, `Mat4`). GPU structs are `#[repr(C)]` + `bytemuck::Pod`, padded by hand to WGSL alignment.
- Lock poisoning is handled with `.expect("<name> poisoned")`.
