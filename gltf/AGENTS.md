# gltf

glTF 2.0 / GLB support. Converts documents into `rendering` entities in an `engine_core::World`. `src/lib.rs` handles loading and scenes, `src/lights.rs` handles scene lights, `src/objects.rs` the editor's placed objects. Wraps the upstream `gltf` crate (same name, so it's referenced as `::gltf` inside).

Depends on `assets`, `engine_core`, `rendering`, plus `image`, `urlencoding` and `serde_json`.

## API

```rust
pub struct GLTFAsset { pub document: gltf::Document, pub buffers: Vec<buffer::Data>, pub images: Vec<image::Data> }
impl Asset for GLTFAsset {}

pub struct GLTFAssetResolver;            // AssetResolver<Asset = GLTFAsset>, reports buffer/image URIs and placed-model URIs as dependencies
pub enum GLTFAssetError { Asset, DocumentError, Image }   // Asset passes through, others -> AssetError::Other
gltf::external_uri(uri) -> Option<Cow<str>>   // percent-decoded relative URI, None for data: URIs

pub struct GltfScene;
GltfScene::new(Arc<GLTFAsset>)
scene.with_root(Mat4) -> GltfScene        // place the whole scene under a transform (a placed model)
scene.load_batch(&mut World, count) -> bool   // true while more mesh nodes remain
scene.is_complete() -> bool
scene.node_entities() -> &HashMap<usize, Vec<EntityKey>>   // glTF node index → its mesh entities
scene.node_emitters() -> &HashMap<usize, EntityKey>        // glTF node index → its ParticleEmitter
gltf::local_matrix(&node) -> Mat4             // node transform relative to its parent
```

`engine::register_asset_resolvers` registers the resolver for `"gltf"` and `"glb"`. The resolver eagerly imports all buffers and images. External URIs go through `LoadContext::read_relative`, so packs work. Images are always decoded to RGBA8. Don't go back to `gltf::import_buffers`/`import_images`: they read the filesystem directly.

### `camera.rs`
Camera settings in `extras`: `PhysicalCamera { aperture, shutter_speed (s), iso }` under a camera's `blackbird_physical` (`ev100()`, `exposure() = 1 / (1.2 · 2^EV100)`), and the main camera's node index under the default scene's `blackbird_main_camera`. The editor writes both in `editor/src/export.rs`.

### `lights.rs`
`LightData` (serde, tagged `type`: sky/point/spot/area) in physical units. `read_lights(doc)` returns `KHR_lights_punctual` node lights (world position/direction, missing `range` → cutoff radius; point/spot `source_radius` from the light's own `extras.blackbird_source_radius` (`SOURCE_RADIUS_KEY`), default 0) plus area lights from root `extras.blackbird_lights`. `LightData::spawn(world)` and `apply(world, key)` create and update the matching `rendering::light` entity. The editor writes lights back in `editor/src/lights.rs`. Keep both sides of the format in step.

### `animation.rs`
Keyframe animation (plan item 10). Two sources, both turned into `engine_core::animation::Clip<Target>`s by `load_clips` when `GltfScene::new` runs:
- **Standard glTF `animations`**: translation / rotation / scale channels (STEP, LINEAR, CUBICSPLINE; morph weights are skipped). The `gltf` crate can't parse `KHR_animation_pointer` (it requires `target.node`), so nothing else is read from them.
- **Root `extras.blackbird_animations`** (`EXTRAS_KEY`, `ClipData`): `{ name, loop (true), autoplay (true), speed (1), animation (optional standard animation index whose channels the clip also plays), tracks: [{ pointer, keys: [{ time, value (number or array), interpolation (step / linear / smooth / ease_in / ease_out / ease_in_out / cubic, default linear), in_tangent, out_tangent }] }] }`. Keys are inline JSON, so the editor can write them without buffers. Pointers follow `KHR_animation_pointer` paths (`Target::parse` / `pointer`): `/nodes/{i}/translation|rotation|scale`, `/materials/{i}/emissiveFactor`, `/materials/{i}/extensions/KHR_materials_emissive_strength/emissiveStrength`, `/extensions/KHR_lights_punctual/lights/{i}/intensity|color`, `/extras/blackbird_lights/{i}/intensity|color`, `/nodes/{i}/extras/blackbird_particles/{spawn_rate|lifetime|size|velocity|turbulence|color|opacity}`. Unknown pointers are skipped.
- Standard animations no clip includes become their own looping, autoplaying clip.

`GltfScene::animate(world, delta)` (game thread, every tick) advances playing clips and writes the sampled values as absolute values: material targets into every mesh entity using that material (`load_batch` records them), light targets into the entities handed over by `bind_lights` (`read_lights` order: punctual lights by their node's light index, `extras` lights by position), particle targets into the node's `ParticleEmitter`. Node transforms are kept per animated node (`Trs`); after a frame that moved one, the default scene is walked again and every node at or under an animated one gets its meshes', emitter's and punctual light's transforms re-derived (cameras aren't moved). Clips that aren't playing and haven't moved are not applied. `clips()` / `clips_mut()` expose them (playback control). Sandbox, player (scene and placed models) and capture call `animate` and `bind_lights`. For the editor's preview: `set_clip_data(&[ClipData])` rebuilds the clips (`build_clips`), `set_light_bindings(punctual, area)` maps light indices to entities directly, `pose(world, clip, time)` applies one clip at a time without advancing anything, and `reset_animation(world)` writes back every material / light / particle value animation overwrote since the last reset (recorded on first write in `originals`) and puts animated nodes back at their glTF transforms.

### `particles.rs`
`ParticleData` (serde): a node's `extras.blackbird_particles` (`EXTRAS_KEY`): `shape` (`box` with `half_extents` and optional `shear_along` direction, or `sphere` with `radius`), `count`, `spawn_rate`, `lifetime` / `size` ranges, `velocity`, `turbulence`, `color`, `opacity`, `blend` (only `additive`), `lighting` (`lit` / `unlit`), `visible_in_light_only`, `collision` (`none` / `settle`); unknown keys (`notes`) are ignored, an invalid object is logged and skipped. `ParticleData::read(node)`, `emitter(transform, seed)` → `rendering::particles::ParticleEmitter` (shear becomes XZ per unit Y, ranges are sorted). The editor keeps these extras on save because it never rewrites node `extras`.

### `objects.rs`
Objects the editor places besides glTF nodes, as root `extras.blackbird_objects` (`EXTRAS_KEY`): a list of `SceneObject { name, translation, rotation ([x, y, z, w]), scale, kind }` (serde; `kind` flattened, tagged `type`):
- `shape` (`Shape`: cube / sphere / plane / cylinder, unit-sized and centered, the plane in XZ facing +Y, plus a `ShapeMaterial`). Geometry isn't stored: `Shape::mesh(&material)` generates it (counter-clockwise outward triangles, UVs and tangents; unit-tested), so player and editor build the same mesh.
- `model { uri }`: another glTF file's default scene placed under the object's transform, URI relative to the scene file like a buffer. The resolver reports these URIs as dependencies, so `collect` ships each model and, transitively, its buffers and images. One level: a placed model's own objects aren't loaded.
- `player_start`: where the player stands (feet) and faces (local −Z heading); `PLAYER_EYE_HEIGHT` (1.7 m) above it.

`read_objects(doc)`, `SceneObject::matrix()`, `objects::trs`. The editor writes them in `editor/src/export.rs`; the player and `capture` spawn them. Keep both sides in step.

## Scene loading
`GltfScene` spreads the work across frames so a large scene doesn't stall the game thread:
1. On the first `load_batch`, every node with `blackbird_particles` becomes a `ParticleEmitter` at its world transform (× the root; seeded by node index), and every perspective camera node in the scene becomes a `rendering::camera::Camera` at the node's world transform, looking down its −Z with +Y up. A missing aspect ratio follows the viewport, a missing `zfar` becomes `camera::INFINITE_FAR`, and exposure comes from the camera's physical settings (else 1). `node_cameras()` maps node index → camera entity, and `main_camera()` resolves the scene's marked camera.
2. `GltfScene::new` walks the default scene (or the first scene, or every root node if there are no scenes) depth-first and records each node that has a mesh, together with its **world matrix** (parent × local).
3. Each call converts the meshes of up to `count` of those nodes. **Each primitive becomes one `rendering::mesh::Mesh` entity** with `transform` set to the node's world matrix. Vertices stay in mesh-local space. A mesh used by several nodes is created once per node.

Primitive conversion (`build_mesh_primitives`):
- Reads positions (a primitive without them is skipped), normals, `COLOR_0`, `TEXCOORD_0`, tangents. Missing attributes get defaults: color white, normal zero, tangent `+X`.
- Indices are widened to `u32`. Non-indexed primitives get `0..n`.
- Material: `baseColorTexture` → `albedo_texture`, `normalTexture` → `normal_texture`, `metallicRoughnessTexture` → `metallic_roughness_texture`, `emissiveTexture` → `emissive_texture` (rendered by `gbuffer`/`lighting` and fed into DDGI), all as `rendering::TextureAsset` (RGBA8). Base color, metallic, roughness and emissive factors plus `KHR_materials_emissive_strength` (crate feature on) go to `Mesh::material`. `KHR_materials_diffuse_transmission` isn't parsed by the `gltf` crate: its factor and color factor are read from the raw extension JSON (`extensions` crate feature) into `MaterialFactors::diffuse_transmission` / `diffuse_transmission_color`; its textures are ignored. `extras.blackbird_foliage` is not read. Only `TEXCOORD_0` is used, whatever `texCoord` a texture names, and `KHR_texture_transform` is ignored. RGB8 is expanded. Other image formats are skipped with a warning. Images are decoded once per `GltfScene` and shared by `Arc`, so primitives that use the same image share one GPU texture.

## Known limitations (intentional current scope)
- Occlusion textures are ignored. `COLOR_0` alpha is dropped (`Vertex::color` is RGB).
