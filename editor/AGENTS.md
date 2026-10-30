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
- **`EditorPlugin`** (`plugin.rs`): runs on the game thread. Each `tick`, in order: `handle_signals` (new/open project, open recent, open scene, export), `poll_load`, `check_file_changes`, `handle_light_requests`, `apply_edits`, `GltfScene::load_batch(10)`, `apply_node_transforms`, then camera controllers (skipped during camera preview).
- **`apply_node_transforms`**: when any `node_edits` TRS changes, or more meshes have streamed in, it recomputes `EditorState::node_world` for every node with entities (from `GltfScene::node_entities`), writes it to each `Mesh::transform`, and refreshes `state.entity_nodes` (mesh entity → node, used for picking).

This request/response pattern through `EditorState` is the rule. UI sets `*_requested` or `Option<PathBuf>` fields, the plugin consumes them with `.take()`, does the work, and writes `status`. Native `rfd` dialogs are opened **on the game thread** and block it while they're open.

## Modules

| File | Contents |
|------|----------|
| `state.rs` | `EditorState` (all shared state), `SceneNode`/`NodeKind` (hierarchy snapshot), `NodeEdit` (TRS + material factors + camera params), `Selection { GltfNode, Light, Mesh }`, `GizmoMode`, `LightKind`, `LightData` (+ `default_for`), `SceneLight { key: EntityKey, data }`. `ensure_node_edit(idx)` seeds a `NodeEdit` from the glTF node (matrix transforms are decomposed). `node_local` / `parent_world` / `node_world` give a node's matrices with edits applied, walking `node_parents`. |
| `math.rs` | glTF-style `[x, y, z, w]` quaternion helpers (`quat_mul`, `quat_from_axis_angle`, `quat_from_matrix`, `rotate`) and `trs`. |
| `plugin.rs` | `EditorPlugin`: project/scene lifecycle, `notify` watcher on the scene file + external buffer/image URIs (300 ms debounce, full reload), light spawning, applying edits to entities. |
| `project.rs` | `Project { name, version, default_scene }` ⇄ `project.json`. `create` scaffolds `scenes/`, `assets/textures/`, `assets/models/`. |
| `recent.rs` | `RecentProjects`, max 10, stored at `$HOME/.blackbird/recent_projects.json` (falls back to `USERPROFILE`). |
| `export.rs` | `write_gltf(&EditorState, path)`: clones the document JSON, writes `node_edits` (TRS, and material factors to *every* material on the node's mesh), serialises **JSON only**. Buffers/images are not copied, so URIs must still resolve relative to the export location. |
| `ui/toolbar.rs` | File menu + status bar. |
| `ui/hierarchy.rs` | Scene node tree + lights list, selection, "add light". |
| `ui/inspector.rs` | Light editor, transform (euler ⇄ quat), material, camera params. |
| `ui/gizmo.rs` | Viewport gizmo for the selected node: mode toolbar (Move/Rotate/Scale, keys 1/2/3), translate arrows and rotate rings on **world** axes, scale handles on the node's **own** axes plus a uniform centre square. Drag state lives in egui temp memory. Also click-to-select: ray vs. each mesh's world AABB, nearest hit selects its node; clicking empty space deselects. |

## Gizmo
- Drawn after the panels, clipped to the free rect left by `root_ui` (`viewport`). It projects with the editor camera's `view_proj_matrix()` onto the whole window (`ctx.content_rect()`), the same mapping the renderer uses.
- Pointer capture: an `Order::Middle` area becomes interactable only while a handle is hovered or dragged. That makes `egui_wants_pointer_input()` true, so the camera controller doesn't orbit during a gizmo drag but still orbits everywhere else.
- Translation is converted into the parent's space (`parent_world⁻¹`), and rotation is conjugated by the parent's rotation, so nested nodes move along world axes. Scale is clamped to ≥ 0.001.
- Hidden during camera preview.

## What edits actually affect

| Edit | Live in viewport | Exported |
|------|------------------|----------|
| Light properties | ✅ (`apply_edits` rewrites every light entity each tick from `LightData`) | ❌ (lights are editor-only) |
| Camera node fov/near/far | ✅ on the editor camera | ✅ |
| Camera preview (`camera_preview_node`) | ✅ editor camera snaps to the node's TRS | — |
| Node transform (inspector or gizmo) | ✅ (children follow their parent) | ✅ |
| Material factors | ❌ (renderer ignores factors) | ✅ |

Mesh entities map back to glTF nodes through `GltfScene::node_entities` / `state.entity_nodes`. Live material editing could use the same mapping.

## Gotchas
- Loading a scene calls `reset_world`, which does `World::clear()`. That destroys lights too, and `start_load` clears `state.lights` to match. Hot reload therefore drops user-added lights.
- `node_edits` is keyed by glTF node index, and `ensure_node_edit` looks the node up with `doc.nodes().nth(idx)`.
- Shaders come from `../../sandbox/shaders/` through `include_str!`.
