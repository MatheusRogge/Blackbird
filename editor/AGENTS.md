# editor

Scene editor binary built on `engine` (`full, debug`) + egui. Opens projects, loads glTF scenes with hot reload, edits lights and camera live, and exports edited glTF. Extra deps: `gltf` 1.4 (renamed `gltf-api` to avoid clashing with the workspace `gltf` crate), `rfd` (native dialogs), `serde`/`serde_json`, `notify`.

Run with `cargo run -p editor`.

## Architecture

```
main.rs ── EditorState::new() ──► SharedState = Arc<Mutex<EditorState>>
                │                        ▲                ▲
                ▼                        │ lock()         │ try_lock()
   WindowedApplication(RenderGraphPBRBuilder.with_debug_flags(show_grid))
      .add_plugin(EditorPlugin)  ── game thread ──┘                │
      .add_ui_layer(EditorUiLayer) ── render thread ────────────────┘
```

- **`EditorUiLayer`** (`ui/`): runs on the render thread. It `try_lock`s the state and skips the frame if the lock is contended. It only *writes request flags and edit values*: no I/O, no `World` mutation.
- **`EditorPlugin`** (`plugin.rs`): runs on the game thread. Each `tick`, in order: `handle_signals` (new/open project, open recent, open scene, export), `poll_load`, `check_file_changes`, `handle_light_requests`, `apply_edits`, `GltfScene::load_batch(10)`, then camera controllers (skipped during camera preview).

This request/response pattern through `EditorState` is the rule. UI sets `*_requested` or `Option<PathBuf>` fields, the plugin consumes them with `.take()`, does the work, and writes `status`. Native `rfd` dialogs are opened **on the game thread** and block it while they're open.

## Modules

| File | Contents |
|------|----------|
| `state.rs` | `EditorState` (all shared state), `SceneNode`/`NodeKind` (hierarchy snapshot), `NodeEdit` (TRS + material factors + camera params), `Selection { GltfNode, Light, Mesh }`, `LightKind`, `LightData` (+ `default_for`), `SceneLight { key: EntityKey, data }`. `ensure_node_edit(idx)` seeds a `NodeEdit` from the glTF node. |
| `plugin.rs` | `EditorPlugin`: project/scene lifecycle, `notify` watcher on the scene file + external buffer/image URIs (300 ms debounce, full reload), light spawning, applying edits to entities. |
| `project.rs` | `Project { name, version, default_scene }` ⇄ `project.json`. `create` scaffolds `scenes/`, `assets/textures/`, `assets/models/`. |
| `recent.rs` | `RecentProjects`, max 10, stored at `$HOME/.blackbird/recent_projects.json` (falls back to `USERPROFILE`). |
| `export.rs` | `write_gltf(&EditorState, path)`: clones the document JSON, writes `node_edits` (TRS, and material factors to *every* material on the node's mesh), serialises **JSON only**. Buffers/images are not copied, so URIs must still resolve relative to the export location. |
| `ui/toolbar.rs` | File menu + status bar. |
| `ui/hierarchy.rs` | Scene node tree + lights list, selection, "add light". |
| `ui/inspector.rs` | Light editor, transform (euler ⇄ quat), material, camera params. |

## What edits actually affect

| Edit | Live in viewport | Exported |
|------|------------------|----------|
| Light properties | ✅ (`apply_edits` rewrites every light entity each tick from `LightData`) | ❌ (lights are editor-only) |
| Camera node fov/near/far | ✅ on the editor camera | ✅ |
| Camera preview (`camera_preview_node`) | ✅ editor camera snaps to the node's TRS | — |
| Node transform | ❌ (the `gltf` crate ignores node transforms) | ✅ |
| Material factors | ❌ (renderer ignores factors) | ✅ |

When you add live transform/material editing, the loaded `Mesh` entities would need to map back to glTF node indices. Nothing tracks that mapping today.

## Gotchas
- Loading a scene calls `reset_world`, which does `World::clear()`. That destroys lights too, and `start_load` clears `state.lights` to match. Hot reload therefore drops user-added lights.
- `node_edits` is keyed by glTF node index, and `ensure_node_edit` looks the node up with `doc.nodes().nth(idx)`.
- Shaders come from `../../sandbox/shaders/` through `include_str!`.
