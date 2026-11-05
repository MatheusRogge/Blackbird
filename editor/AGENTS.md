# editor

Scene editor binary built on `engine` (`full, debug`) + egui. Opens projects, loads glTF scenes with hot reload, edits lights and camera live, and exports edited glTF. Extra deps: `gltf` 1.4 with `extensions` + `KHR_lights_punctual` (renamed `gltf-api` to avoid clashing with the workspace `gltf` crate), `rfd` (native dialogs), `serde`/`serde_json`, `notify`, `image` (thumbnails), and the workspace crates `project` (`project.json`) and `build_tool` (builds and asset collection).

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
- **`EditorPlugin`** (`plugin.rs`): runs on the game thread. Each `tick`, in order: `handle_signals` (new/open project, open recent, open scene, import, save, export), `poll_load`, `check_file_changes`, `handle_light_requests`, `apply_edits`, `sync_project`, `handle_build`, `sync_assets`, `GltfScene::load_batch(10)`, `apply_node_transforms`, then `EditorCameraController` (skipped during camera preview).
- **`apply_node_transforms`**: when any `node_edits` TRS changes, or more meshes have streamed in, it recomputes `EditorState::node_world` for every node with entities (from `GltfScene::node_entities`), writes it to each `Mesh::transform`, and refreshes `state.entity_nodes` (mesh entity → node, used for picking).

This request/response pattern through `EditorState` is the rule. UI sets `*_requested` or `Option<PathBuf>` fields, the plugin consumes them with `.take()`, does the work, and writes `status`. Native `rfd` dialogs are opened **on the game thread** and block it while they're open.

## Modules

| File | Contents |
|------|----------|
| `state.rs` | `EditorState` (all shared state), `SceneNode`/`NodeKind` (hierarchy snapshot), `NodeEdit` (TRS + material factors + camera params), `Selection { GltfNode, Light }`, `GizmoMode`, `LightKind`, `LightDataExt` (`icon`/`label`/`default_for` on the re-exported `engine::gltf::lights::LightData`), `CameraSettings` (re-exported from `project`), `SceneLight { key: EntityKey, data }`, `BuildState`, and `assets: AssetBrowserState`. `ensure_node_edit(idx)` seeds a `NodeEdit` from the glTF node (matrix transforms are decomposed). `node_local` / `parent_world` / `node_world` give a node's matrices with edits applied, walking `node_parents`. |
| `camera.rs` | `EditorCameraController`: editor viewport camera, replacing the engine's `CameraController`/`PlayerController` here. LMB drag looks, RMB drag orbits the pivot (`Camera::target`), MMB drag pans, scroll dollies (it pushes the pivot forward once it gets within 0.1). WASD moves along the view direction, Q/E move along world Y, Shift multiplies speed, LMB + scroll scales speed. Yaw/pitch are re-derived from the `Camera` every tick, and `up` is forced to +Y. Speeds come from `EditorState::camera_settings` (Camera menu, persisted in `project.json`). |
| `math.rs` | glTF-style `[x, y, z, w]` quaternion helpers (`quat_mul`, `quat_from_axis_angle`, `quat_from_matrix`, `rotate`) and `trs`. |
| `plugin.rs` | `EditorPlugin`: project/scene lifecycle (`project::Project`; `default_scene` is opened by `load_project`), `notify` watcher on the scene file + external buffer/image URIs (reloads 500 ms after the *last* event, so it waits for the exporter to finish; `start_reload` keeps the current scene, watcher and editor camera pose until the new load succeeds, and a failed reload is retried up to 3 times, 1 s apart), light spawning, applying edits to entities, builds, asset browser sync. `refresh_project_files` re-lists `project_scenes` and rescans the asset browser. It runs on project open/create, import, save, export, after a build and when `project.json` is written. |
| `asset_browser.rs` | Game-thread half of the asset browser. `AssetWorker` owns an `asset-browser` thread that scans the project (skipping `builds/` and dot-files), runs `build_tool::collect_manifest` to mark what a build would ship (`BuildSet`), and on request decodes 128 px thumbnails (textures; for models, the first base-color image) and model stats (`ModelStats`: counts, vertices/triangles, external files). `rescan` bumps `AssetBrowserState::generation`, and results from older generations are dropped. |
| `recent.rs` | `RecentProjects`, max 10, stored at `$HOME/.blackbird/recent_projects.json` (falls back to `USERPROFILE`). |
| `import.rs` | `import_scene(source, project_dir)`: copies a `.gltf`/`.glb` into the project as `scenes/<name>.gltf`, writing every buffer (incl. GLB `BIN` and data URIs) to `assets/models/<name>/` and every image (external, data URI or buffer-view embedded) to `assets/textures/<name>/`, with URIs rewritten to `../assets/...`. Names are sanitised to URI-safe characters and `<name>` is suffixed `_N` if taken. The plugin opens the imported scene afterwards. |
| `export.rs` | `write_gltf(&EditorState, path)`: clones the document JSON, writes `node_edits` (name, TRS, material factors to *every* material on the node's mesh, camera perspective + `blackbird_physical`), calls `lights::write_lights`, marks `main_camera` on the default scene, serialises **JSON only**. Buffers/images are not copied, so URIs must still resolve relative to the export location. |
| `lights.rs` | Writes scene lights into glTF; reading and the format live in `engine::gltf::lights` (also used by the standalone `player`). Sky/point/spot → `KHR_lights_punctual` (sky = `directional`, `radius` = `range`, light aims down the node's −Z, intensity copied 1:1 since the renderer uses the same physical units). Area lights → root `extras.blackbird_lights` (tagged `LightData`; other `extras` keys are kept). `write_lights` removes all existing punctual lights, then writes each light back onto `SceneLight::node` (setting its local TRS through `parent_world⁻¹`) or onto a new root node in the default scene. `poll_load` respawns whatever `gltf::lights::read_lights` returns, via `LightData::spawn`; `apply_edits` pushes edits with `LightData::apply`. |
| `ui/assets.rs` | `AssetPanel` (owned by `EditorUiLayer`, render thread): bottom "Assets" panel with folder list, kind toggles + search, a tile grid (thumbnails, a green dot = ships in builds, dimmed = left out of builds) and a details pane (preview, texture/model stats, files a model uses with missing/outside-project flags). Tiles push visible inspectable paths into `assets.wanted`; thumbnails become egui textures cached per path, cleared when `generation` changes. Double-click opens scenes/models; actions: Open, Default scene, Always ship (`project.build.include`), Copy path. |
| `ui/build.rs` | Build menu (Development / Production, Show Build Log) and the "Build" log window. Setting `build.requested` makes `handle_build` save pending `project.json` changes and run `build_tool::build` on a `build` thread, streaming its output into `build.log`. |
| `ui/toolbar.rs` | File, Project (default scene), Build, View (grid toggle, writes the shared `RenderDebugFlags`; gizmos toggle, `show_gizmos`) and Camera (move speed, Shift multiplier) menus + status bar (spinner while building). Owns the Ctrl+S (sets `save_requested`) and G (toggles grid) shortcuts. |
| `ui/hierarchy.rs` | One scene tree: the default scene's nodes (⏶ mesh, 📷 camera, light icon for `KHR_lights_punctual` nodes, ○ empty), then every light with no row of its own (added in the editor, area lights, lights on mesh/camera nodes or outside the default scene). The inspector shows a mesh node's mesh name and primitive count. Background context menu: "Default Scene" toggle (only for scenes inside the project) and "add light". |
| `ui/inspector.rs` | Light editor, node name (renames the hierarchy row too), transform (euler ⇄ quat), material, camera (`CameraEdit`: player camera, FOV, near, far/infinite, aspect viewport/preset/custom, physical exposure). |
| `ui/gizmo.rs` | Viewport gizmo for the selected node or light (`Target`): mode toolbar (Move/Rotate/Scale, keys 1/2/3), translate arrows and rotate rings on **world** axes, scale handles on the target's **own** axes plus a uniform centre square. Drag state lives in egui temp memory. Also draws light markers (only the selected light's when `show_gizmos` is off; dot; spot aim line; area outline; for sky lights a sun on the side the light comes from with an arrow into the origin, grey below the horizon) and does click-to-select: a light marker within reach wins, otherwise ray vs. each mesh's world AABB, nearest hit selects its node; clicking empty space deselects. |

## Gizmo
- Drawn after the panels, clipped to the free rect left by `root_ui` (`viewport`). It projects with the editor camera's `view_proj_matrix()` onto the whole window (`ctx.content_rect()`), the same mapping the renderer uses.
- Pointer capture: an `Order::Middle` area becomes interactable only while a handle is hovered or dragged. That makes `egui_wants_pointer_input()` true, so the camera controller doesn't start a drag on a gizmo handle but still does everywhere else.
- Translation is converted into the parent's space (`parent_world⁻¹`), and rotation is conjugated by the parent's rotation, so nested nodes move along world axes. Scale is clamped to ≥ 0.001.
- Hidden during camera preview.
- The gizmo edits a `Pose` (translation/rotation/scale in the parent's frame). Nodes read/write `node_edits`. Lights convert through `light_pose`/`apply_light_pose` with an identity parent: lights aim down local −Z (`math::quat_look`), and an area light's `right`/`up` are local X/Y scaled by their half-extents.
- `editable_axes` decides which modes and axes a target gets: point = move, spot = move + rotate, sky = rotate (drawn at the scene origin, since it has no position), area = move + rotate + scale X/Y. When the current mode doesn't apply, the gizmo switches to the first one that does, and the toolbar disables the others.

## What edits actually affect

| Edit | Live in viewport | Saved / exported |
|------|------------------|----------|
| Lights (add + properties) | ✅ (`apply_edits` rewrites every light entity each tick from `LightData`) | ✅ (`KHR_lights_punctual`, area lights in `extras`) |
| Node name | ✅ in the hierarchy | ✅ |
| Camera lens + physical exposure | ✅ during Assume Camera View | ✅ (glTF perspective; physical settings in the camera's `extras`) |
| Player camera (`main_camera`) | — | ✅ (default scene's `extras`) |
| Camera preview (`camera_preview_node`) | ✅ editor camera takes the node's world transform, lens and exposure; leaving restores the editor lens | — |
| Node transform (inspector or gizmo) | ✅ (children follow their parent) | ✅ |
| Material factors | ❌ (renderer ignores factors) | ✅ |

Mesh entities map back to glTF nodes through `GltfScene::node_entities` / `state.entity_nodes`. Live material editing could use the same mapping.

## Gotchas
- **Project settings** are edited in place on `state.project` (and `state.camera_settings`, copied into it). `sync_project` writes `project.json` 500 ms after the last change, so a DragValue drag doesn't write every frame.
- **Save** (`handle_save_scene`) runs `write_gltf` onto `state.file_path` and then ignores watcher events for 500 ms, so the editor's own write doesn't hot-reload (which would drop lights and selection). `.glb` scenes can't be saved in place; use Export.
- Loading a scene calls `clear_scene` → `reset_world`, which does `World::clear()`. That destroys lights too, and `clear_scene` clears `state.lights` to match (for a hot reload, only once the new file has loaded); `poll_load` respawns the ones saved in the file. Hot reload therefore drops lights added since the last save.
- A node carrying a `KHR_lights_punctual` light is `NodeKind::Light`. Clicking it in the hierarchy selects its `SceneLight` (`Selection::Light`), never the node, because on save the light's position/direction overwrite the node's TRS. A node that also has a mesh or camera stays selectable as a node, so moving it won't move the light.
- Unknown glTF extensions survive save/export only because of the gltf `extensions` feature (they land in each struct's `others` map). Don't drop that feature.
- Light intensities are physical (lux for sky, candela for point/spot, nits for area). Exposure lives in `CameraSettings::exposure_stops` (Camera menu, `project.json`) and `EditorCameraController` copies `2^stops` into `Camera::exposure` each tick.
- `node_edits` is keyed by glTF node index, and `ensure_node_edit` looks the node up with `doc.nodes().nth(idx)`.
- Shaders come from `../../sandbox/shaders/` through `include_str!`.
- Builds read the **saved** scene files, not the editor's in-memory edits. Save first.
- The asset browser's "ships in builds" marks come from the same `build_tool::collect_manifest` a real build uses, so an asset shown dimmed really is missing from builds. If the game loads it by path at runtime, mark it Always ship.
