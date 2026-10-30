# gltf

glTF 2.0 / GLB support. Converts documents into `rendering` entities in an `engine_core::World`. Single file, `src/lib.rs`. Wraps the upstream `gltf` crate (same name, so it's referenced as `::gltf` inside).

Depends on `assets`, `engine_core`, `rendering`.

## API

```rust
pub struct GLTFAsset { pub document: gltf::Document, pub buffers: Vec<buffer::Data>, pub images: Vec<image::Data> }
impl Asset for GLTFAsset {}

pub struct GLTFAssetResolver;            // AssetResolver<Asset = GLTFAsset>
pub enum GLTFAssetError { IOError, DocumentError }   // -> AssetError::Other

pub struct GltfScene;
GltfScene::new(Arc<GLTFAsset>)
scene.load_batch(&mut World, count) -> bool   // true while more meshes remain
scene.is_complete() -> bool
```

`engine`'s `GltfPlugin` registers the resolver for `"gltf"` and `"glb"`. The resolver eagerly imports all buffers and images (external URIs are relative to the file's directory).

## Scene loading
`GltfScene` spreads the work across frames so a large scene doesn't stall the game thread:
1. On the first `load_batch`, every perspective camera becomes a `rendering::camera::Camera` entity. Its eye/target are **hard-coded** (`(0,40,150)` → `(0,40,0)`) and the node transform is ignored.
2. Each call converts up to `count` glTF meshes. **Each primitive becomes one `rendering::mesh::Mesh` entity.**

Primitive conversion (`build_mesh_primitives`):
- Reads positions (a primitive without them is skipped), normals, `COLOR_0`, `TEXCOORD_0`, tangents. Missing attributes get defaults: color white, normal zero, tangent `+X`.
- Indices are widened to `u32`. Non-indexed primitives get `0..n`.
- Material: only `baseColorTexture` → `albedo_texture` and `normalTexture` → `normal_texture`, both as `rendering::TextureAsset` (RGBA8). RGB8 is expanded. Other image formats are skipped with a warning.

## Known limitations (intentional current scope)
- **Node transforms are not applied.** Vertices stay in mesh-local space, so scenes that rely on the node hierarchy render wrong.
- Meshes are walked from `document.meshes()`, not the scene graph. Instanced meshes appear once.
- PBR factors (base color, metallic, roughness, emissive), metallic-roughness texture, occlusion and emissive textures are ignored.
- Textures are decoded per primitive with no sharing: each primitive gets its own `Arc<TextureAsset>`, so the GBuffer pass's texture cache (keyed by `Arc` pointer) won't dedupe them.
- `aspect` fallback `(4 / 3) as f32` is integer division (= 1.0).
