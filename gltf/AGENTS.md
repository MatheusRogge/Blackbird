# gltf

glTF 2.0 / GLB support. Converts documents into `rendering` entities in an `engine_core::World`. `src/lib.rs` handles loading and scenes, `src/lights.rs` handles scene lights. Wraps the upstream `gltf` crate (same name, so it's referenced as `::gltf` inside).

Depends on `assets`, `engine_core`, `rendering`, plus `image`, `urlencoding` and `serde_json`.

## API

```rust
pub struct GLTFAsset { pub document: gltf::Document, pub buffers: Vec<buffer::Data>, pub images: Vec<image::Data> }
impl Asset for GLTFAsset {}

pub struct GLTFAssetResolver;            // AssetResolver<Asset = GLTFAsset>, reports buffer/image URIs as dependencies
pub enum GLTFAssetError { Asset, DocumentError, Image }   // Asset passes through, others -> AssetError::Other
gltf::external_uri(uri) -> Option<Cow<str>>   // percent-decoded relative URI, None for data: URIs

pub struct GltfScene;
GltfScene::new(Arc<GLTFAsset>)
scene.load_batch(&mut World, count) -> bool   // true while more mesh nodes remain
scene.is_complete() -> bool
scene.node_entities() -> &HashMap<usize, Vec<EntityKey>>   // glTF node index → its mesh entities
gltf::local_matrix(&node) -> Mat4             // node transform relative to its parent
```

`engine::register_asset_resolvers` registers the resolver for `"gltf"` and `"glb"`. The resolver eagerly imports all buffers and images. External URIs go through `LoadContext::read_relative`, so packs work. Images are always decoded to RGBA8. Don't go back to `gltf::import_buffers`/`import_images`: they read the filesystem directly.

### `lights.rs`
`LightData` (serde, tagged `type`: sky/point/spot/area) in physical units. `read_lights(doc)` returns `KHR_lights_punctual` node lights (world position/direction, missing `range` → cutoff radius) plus area lights from root `extras.blackbird_lights`. `LightData::spawn(world)` and `apply(world, key)` create and update the matching `rendering::light` entity. The editor writes lights back in `editor/src/lights.rs`. Keep both sides of the format in step.

## Scene loading
`GltfScene` spreads the work across frames so a large scene doesn't stall the game thread:
1. On the first `load_batch`, every perspective camera node in the scene becomes a `rendering::camera::Camera` at the node's world transform, looking down its −Z with +Y up. A missing aspect ratio falls back to 4:3.
2. `GltfScene::new` walks the default scene (or the first scene, or every root node if there are no scenes) depth-first and records each node that has a mesh, together with its **world matrix** (parent × local).
3. Each call converts the meshes of up to `count` of those nodes. **Each primitive becomes one `rendering::mesh::Mesh` entity** with `transform` set to the node's world matrix. Vertices stay in mesh-local space. A mesh used by several nodes is created once per node.

Primitive conversion (`build_mesh_primitives`):
- Reads positions (a primitive without them is skipped), normals, `COLOR_0`, `TEXCOORD_0`, tangents. Missing attributes get defaults: color white, normal zero, tangent `+X`.
- Indices are widened to `u32`. Non-indexed primitives get `0..n`.
- Material: only `baseColorTexture` → `albedo_texture` and `normalTexture` → `normal_texture`, both as `rendering::TextureAsset` (RGBA8). RGB8 is expanded. Other image formats are skipped with a warning. Images are decoded once per `GltfScene` and shared by `Arc`, so primitives that use the same image share one GPU texture.

## Known limitations (intentional current scope)
- PBR factors (base color, metallic, roughness, emissive), metallic-roughness texture, occlusion and emissive textures are ignored.
