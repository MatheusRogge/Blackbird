# build_tool

Turns an editor project into a standalone game. It is a library (used by the editor's Build menu and asset browser) plus the `blackbird` CLI. Depends on `engine` (`gltf` feature only), `project`, `serde_json`.

```bash
cargo run -p build_tool -- build <project-dir> [--production] [--out <dir>] [--target <triple>] [--level <1-22>]
cargo run -p build_tool -- assets <project-dir>      # what a build would ship
cargo run -p build_tool -- inspect <file.bbpak>      # pack contents, codecs, sizes
```

## Pipeline (`build`)
1. `collect_manifest`: `Project::build_roots()` (`project.json`, `default_scene`, `build.include`) walked by `AssetManager::collect` over a `DirectorySource` on the project, with `engine::register_asset_resolvers`. Only reachable files ship. A missing or out-of-project dependency fails the build.
2. `prepare_out_dir`: default `<project>/builds/<development|production>/`. A non-empty directory without the `.blackbird-build` marker is refused. Otherwise `data/` and `data.bbpak` from earlier builds are cleared.
3. Assets. **Development**: copied loose into `<out>/data/`. **Production**: written into `<out>/data.bbpak` (zstd, level 19 by default). Each file is re-hashed against the manifest, so files edited mid-build fail it.
4. The manifest text is written to `<workspace>/target/blackbird/<slug>/<kind>/assets.manifest`.
5. `cargo build -p player --profile <kind>` (production adds `--no-default-features --features production`) with `BLACKBIRD_ASSET_MANIFEST` and, for production, `BLACKBIRD_PACK_ID` (pack content id, hex). `player/build.rs` turns these into code. The executable path comes from cargo's JSON messages. Stderr streams to the log callback.
6. The executable is copied to `<out>/<project slug>[.exe]`.

## Gotchas
- `workspace_root()` is `CARGO_MANIFEST_DIR/..` at compile time. Builds only work on a machine with this engine checkout.
- `CARGO` is honoured when set (it is under `cargo run`). Otherwise `cargo` must be on `PATH`.
- Build output layout (`DATA_DIR`, `PACK_FILE`) lives in `project` so the player can use it without depending on this crate.
