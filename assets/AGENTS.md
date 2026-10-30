# assets

Type-erased asset loading keyed by file extension. Everything lives in `src/lib.rs`. Depends on `threading` for async loads.

## Core types

| Type | Purpose |
|------|---------|
| `trait Asset: Any + Downcast + Send + Sync` | Marker. Implement it on any loaded type (`GLTFAsset`, `ShaderAsset`, `TextureAsset`). |
| `trait AssetResolver { type Asset; type Error: Into<AssetError>; fn resolve(&self, base_path: &Path, file: File) -> Result<Self::Asset, Self::Error> }` | Parses an opened file. `base_path` is the file's parent dir, used for relative URIs. |
| `ErasedAssetResolver` | Object-safe wrapper, blanket-implemented for every `AssetResolver + Send + Sync`. |
| `ErasedLoadedAsset` / `LoadedAsset<A>` | Boxed `dyn Asset` and its downcast form. |
| `AssetError` | `IO(io::Error)` or `Other(String)`. |
| `AssetLoadHandle<A>` | Async result. `try_get() -> Option<Result<Arc<A>, Arc<AssetError>>>`: `None` while pending. Poll it each tick. |
| `AssetHandle` | `Arc<ErasedLoadedAsset>` wrapper. Currently unused. |

## `AssetManager`

```rust
add_resolver(ext: impl Into<String>, resolver)   // "gltf", "glb", ...; no leading dot
init_executor(Arc<dyn Executor>)                 // required before load_async
load_asset::<A>(path) -> Result<Arc<A>, AssetError>              // blocking
load_asset_from::<A>(path, base_path) -> Result<Arc<A>, ...>     // blocking, base_path.join(path)
load_async::<A>(path) -> AssetLoadHandle<A>                      // runs on executor
```

Resolution flow: resolve the path, look up the resolver by extension, open the `File`, call `resolver.resolve(parent, file)`, downcast to `A`. A type mismatch becomes `AssetError::Other`.

## Gotchas
- **Relative paths** resolve against `env::current_exe()` joined with the path, which means *the executable file itself*, not its directory. That is almost certainly a bug. Pass absolute paths until it is fixed.
- Extension lookup is case-sensitive (`.GLB` won't match `"glb"`).
- `load_async` without an executor returns an already-failed handle instead of panicking. In `engine`, `AssetsPlugin` (auto-added by `WindowedApplication::new`) sets the executor up.
- No caching or dedup: loading the same path twice parses it twice.
- Resolvers run on the executor thread, so they must be `Send + Sync` and must not touch the `World`.
