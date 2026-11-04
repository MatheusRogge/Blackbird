# player

Standalone runtime for an editor project: boots `project.json`, streams the default scene, spawns its lights and makes sure there's a camera with the project's exposure. It is normally built by `build_tool`, not by hand.

## Content modes (decided at compile time by `build.rs`)
`build.rs` reads `BLACKBIRD_ASSET_MANIFEST` / `BLACKBIRD_PACK_ID` and writes `OUT_DIR/assets.rs` via `assets::codegen`: `paths::*` consts, `ASSETS: &[StaticAsset]`, `PACK_ID`. `Content::CURRENT` follows from that:

| Mode | When | Assets from |
|------|------|-------------|
| `Pack` | `PACK_ID` set (production) | `data.bbpak` beside the exe. Refused unless its content id matches `PACK_ID`. |
| `Loose` | manifest but no pack (development) | `data/` beside the exe. Loads outside the manifest log a warning. |
| `Unbound` | no manifest (`cargo run -p player -- <project-dir>`) | the project dir given as the first argument |

## Features
- `dev` (default): `engine/debug` (rendering stats overlay), log file at `<exe dir>/logs/player.log`, `dev_tools.rs`: Dev Tools window (F1) with build/asset/scene info and a log console, and a free camera (F2; WASD + LMB look).
- `production`: `log/release_max_level_warn`. With `dev` off there is no egui, no dev tools and no log file. Logs go to stderr at warn level. The Windows subsystem is GUI (no console).

Profiles live in the workspace `Cargo.toml`: `development` (dev, opt-level 1) and `production` (release + `panic = "abort"`, fat LTO, stripped).

## Gotchas
- Shaders are `include_str!`'d from `../../sandbox/shaders/`, as in the editor.
- `Camera::aspect` is fixed (glTF value, or 16:9 for the fallback camera). Nothing updates it on window resize yet.
- The scene's lights come from `engine::gltf::lights::read_lights`. Scene cameras come from `GltfScene` (node transforms). The fallback camera sits at (0, 2, 8) looking at the origin.
