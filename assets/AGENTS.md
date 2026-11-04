# assets

Type-erased asset loading keyed by file extension, read through a swappable `AssetSource` (loose files or a `.bbpak` pack). It also has the build-time pieces: dependency walking, manifests, the pack format and codegen. Depends on `threading` for async loads, plus `zstd` behind the default `pack` feature.

## Modules

| Module | Contents |
|--------|----------|
| `lib.rs` | `Asset`, `AssetResolver`, `LoadContext`, `AssetManager`, `AssetLoadHandle`, `AssetError` |
| `source.rs` | `trait AssetSource { read(&Path) -> io::Result<Vec<u8>>; describe() }`, `DirectorySource` |
| `path.rs` | `normalize` (lexical `.`/`..`), `resolve_relative(asset, uri)`, `to_logical` (`/`-separated, `None` if absolute or escaping) |
| `manifest.rs` | `AssetManifest` (sorted `ManifestEntry { path, size, hash }`), text format, `StaticAsset`, `content_hash` (FNV-1a) |
| `pack.rs` (feature `pack`) | `.bbpak` format: `PackWriter`, `PackReader`, `PackSource` |
| `codegen.rs` | `generate(&AssetManifest, pack_id) -> String` for build scripts |

## Resolvers
```rust
trait AssetResolver {
    type Asset: Asset; type Error: Into<AssetError>;
    fn resolve(&self, ctx: &mut LoadContext) -> Result<Self::Asset, Self::Error>;
    fn dependencies(&self, ctx: &LoadContext) -> Result<Vec<String>, Self::Error> { Ok(vec![]) }
}
LoadContext::path() / bytes() / take_bytes() / read_relative(uri)
```
- Read sibling files **only** through `ctx.read_relative`, never `std::fs`. That is what lets the same resolver load from a pack.
- `dependencies` must list every file `resolve` reads through `read_relative` (relative to the asset's directory, percent-decoded). Builds ship exactly the files reached from the roots, so a missing entry shows up as a missing file in a production build, not in the editor.

## `AssetManager`
```rust
add_resolver(ext, resolver)            // case-insensitive, no leading dot
init_executor(Arc<dyn Executor>)       // Engine::new does this when engine has "assets"
set_source(Arc<dyn AssetSource>)       // default: DirectorySource beside the executable
set_manifest(AssetManifest)            // loads outside it log a warning
load_asset::<A>(path) / load_asset_from::<A>(path, base) / load_async::<A>(path)
collect(&roots) -> Result<AssetManifest, AssetError>   // BFS over resolver dependencies
```
- Relative paths resolve against the source root. `DirectorySource` passes absolute paths straight to the filesystem, which the editor relies on. `PackSource` rejects them.
- `collect` treats files without a resolver (`.bin`, `.png`) as leaves. It fails when a dependency is missing or climbs outside the root, and the error names the referencing asset.
- No caching or dedup: loading the same path twice parses it twice.
- Resolvers run on the executor thread, so they must be `Send + Sync` and must not touch the `World`.

## `.bbpak`
Little-endian. A 48-byte header (`BBPAK\0\0\0`, version, count, TOC offset/lengths, content id), then 16-byte-aligned blobs, then a zstd-compressed TOC sorted by path. Each entry is zstd-compressed, or stored as-is when compression saves less than 5% (PNG/JPEG). Reads decompress one entry on demand and verify its hash. The content id is the hash of the raw TOC. Builds bake it into the binary so a mismatched pack is refused. Bump `VERSION` on any layout change.

## Manifest text format
`blackbird-manifest 1` header, then `<hash hex> <size> <path>` per line. The path is last so it may contain spaces. Written by `build_tool`, read by `player/build.rs`.
