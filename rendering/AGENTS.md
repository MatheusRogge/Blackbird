# rendering

wgpu 30 renderer (Vulkan backend forced). Contains the render graph, every render/compute pass, the deferred PBR + clustered lighting + CSM shadows + DDGI pipeline, the 2D sprite renderer, camera/light entities and the GPU profiler. Depends on `engine_core` and `assets` only. Re-exports `wgpu`.

## Features
- `rendering_2d` (default): the `rendering_2d` module.
- `debug`: `DebugLayer` (an egui `UiLayer` with frame stats + `RenderDebugFlags` toggles/DDGI tuning). Pulls in `egui` + `ui`.

## Module map

| Module | Contents |
|--------|----------|
| `renderer` | `Renderer` (instance/surface/device/queue, resize, `render(world, overlay)`), `RenderGraphBuilder` trait, `OverlayPass` trait, `RendererError` |
| `graph` | `RenderGraph`, `GraphContext`, `SubGraphBuilder`, `GraphNode`, `NodeId` |
| `arena` | `BufferRange`, `UploadArena` (packed upload buffers), `GeometryArena` (shared mesh buffers), `FrameArena` (per-frame linear uploads), `texture_pool::assign` (screen texture aliasing), `ArenaStats` |
| `resource` | `ResourceId`, `ResourceDescriptor`, `BindingResource`, `AllocatedResource`, `GraphResource` |
| `pass` | `Pass` / `PassDesc` traits, `PassContext`, and every PBR pass (below) |
| `pbr` | `RenderGraphPBRBuilder`, which wires the full 3D pipeline |
| `rendering_2d` | `Sprite`, `SpriteTexture`, `Camera2D`, `SpritePass`, `SpriteRenderGraphBuilder` (shader embedded in-crate) |
| `camera` | `Camera` entity (`eye/target/up/fovy/aspect/near/far`). **Reversed-Z** projection. Implements `Controllable`. |
| `camera_controller` | `CameraController` (`FirstPerson` / `ThirdPerson` orbit + scroll zoom; respects `ui_wants_pointer`) |
| `light` | Entities `PointLight`, `SpotLight`, `AreaLight`, `SkyLight` + `Gpu*` structs, `LightCounts`, `MAX_*_LIGHTS` (256/64/32/4) |
| `mesh` | `Vertex` (pos, normal, color, uv, tangent), `Mesh` entity (+ optional albedo/normal `Arc<TextureAsset>`), `MeshBatch` |
| `texture` | `TextureAsset` (RGBA8) |
| `shader` | `ShaderAsset::from_raw(wgsl)`, `ShaderAssetResolver` |
| `bvh` | CPU BVH builder (`build(Vec<GpuTriangle>) -> BvhBuild`), `BvhNode`, `BvhInfo`, limits 1M nodes / 500k tris |
| `cache` | `PipelineCache`, `PipelineLayoutCache`, `BindGroupCache` helpers |
| `profiler` | `FrameProfiler` (CPU per-pass, 60-frame avg), `GpuProfiler` (timestamp queries, async readback), `FrameStats`, `PassStats` |
| `debug_flags` | `RenderDebugFlags` (shared `Arc<RwLock<_>>`), `DebugFlagsGpu` |
| `debug_layer` | `DebugLayer` (feature `debug`): egui frame stats, arena occupancy, debug flag toggles, and the "Texture aliasing" timeline (passes × physical textures, fed by `ArenaStats.screen_textures: Arc<AliasingReport>`) |

## Render graph

### Building
A `RenderGraphBuilder::build(self, device, surface_config) -> (RenderGraph, ResourceId)` returns the graph and the **output resource id** (the surface, created with `create_external_resource()`).

The pass constructor convention is `XPass::new(&mut graph, upstream_outputs…, shader) -> (XPass, XOutputs)`. The constructor allocates its output resources with `graph.alloc_resource_id(desc)` and returns the ids in an `XOutputs` struct for downstream passes. Then `graph.add_pass(device, pass)` registers it, creates a bind group layout from `layout_entries()` if there is one, and calls `bind_node_id`.

`ResourceDescriptor` variants:
- `ScreenTexture`: sized to the surface and reallocated on resize. **Transient**: screen textures with the same format + usage whose lifetimes in the pass order don't overlap share one physical texture. Contents are undefined after the last reader in a frame, and the first writer must clear or fully overwrite it. A screen texture some pass reads before any pass writes it is never aliased.
- `Texture`: has an explicit size but is *also* resized to the surface on resize.
- `FixedTexture`, `Fixed3DTexture`: never resized.
- `UploadBuffer`: CPU-written (`queue.write_buffer`), GPU **read-only**. Packed with every other `UploadBuffer` into a few `UploadArena` pages on the first `execute` (offsets aligned to the device's uniform/storage offset alignment). The arena is then sealed: adding an `UploadBuffer` later trips a `debug_assert!`. Use this for uniforms, params, light lists, BVH data.
- `Buffer`: a dedicated buffer. Use it for anything bound as `read_write` storage (`light_grid`, `light_indices`, `ray_radiance`, `ray_direction`). wgpu tracks usage per buffer, not per range, so a GPU-written range inside a shared page would conflict with the read-only ranges bound in the same pass.
- `ExternalView`: never allocated. Use `import_texture` / `update_imported`, or it's the surface.

### Executing (`RenderGraph::execute`, every frame)
1. **Compile** (cached, redone after `add_pass` / `add_subgraph`): walk backwards from the output resource through `writes()` → `reads()` to find the needed passes, then topologically sort them (Kahn, ties broken by lowest `NodeId`). `reads()` / `writes()` must therefore be static.
2. On the first frame build the `UploadArena`. Every frame, `GeometryArena::sync` uploads new meshes and frees removed ones, and `FrameArena::begin_frame` resets the frame cursor. When a screen texture is unallocated (first frame, resize, graph change), assign and allocate all screen textures through `texture_pool::assign`. Then lazily allocate every other unallocated non-external resource.
3. Rebuild the graph-managed bind group (from `binding_resources()` + `samplers()`) for **every** pass that has one. This happens every frame.
4. For each pass in order, build a `PassContext { views, textures, buffers, bind_group, upstream, surface_size, geometry, frame }` holding only that pass's declared reads + writes. `buffers` holds `BufferRange`s: bind with `.binding()`, write with `.write(queue, ..)` / `.write_at(..)`, slice with `.slice()`. Never use `as_entire_binding()` or offset 0 on a graph buffer, because it may share a page. The output id maps to the swapchain view. Then call `pass.execute(device, queue, encoder, &ctx, world)` wrapped in CPU and GPU timing.
5. Return one `CommandBuffer` and publish `ArenaStats` in `FrameStats.memory`. `Renderer` submits the buffer, plus the overlay buffer if there is one, then calls `schedule_gpu_readback`.

**Rules that follow from this:**
- A pass is **culled** unless something on the path to the output reads a resource it writes. Every resource a pass touches must appear in `reads()` / `writes()`, or the pass is dropped, misordered, or finds the id missing from `ctx`.
- There is one writer per resource (the `writer_of` map keeps the last one). Ping-pong patterns (DDGI atlases) declare both slots and track the active one with an `Arc<AtomicUsize>`.
- The order of independent passes is deterministic (lowest `NodeId` first), but it's an implementation detail. If one pass must run before another, add a read/write dependency. Texture aliasing is computed from this order.
- A pass-owned uniform should be an `UploadBuffer` declared in the constructor and listed in that pass's own `writes()`, so it lands in `ctx.buffers`. Shadow cascade VPs share one `UploadBuffer` at a 256-byte stride with a dynamic-offset bind group.
- `queue.write_buffer` calls made during `execute` all land before the frame's command buffer runs.
- Most passes build their pipelines **lazily on first execute** and own extra wgpu objects themselves (own bind groups, material texture caches). Graph bind groups are only for the simple case. Pass-cached bind groups that reference screen textures must be dropped in `on_resize`, which is when aliasing is reassigned.

### Arenas
- `GeometryArena` (`ctx.geometry`): one vertex + one index buffer for every `Mesh`, sub-allocated with `offset-allocator`, keyed by `EntityKey`. Bind `vertex_buffer()` / `index_buffer()` once per pass, then `draw_indexed(draw.indices.clone(), draw.base_vertex, ..)` with `ctx.geometry.get(key)`. When full it rebuilds bigger buffers and re-uploads every live mesh (`generation` bumps). Never hold a buffer across frames.
- `FrameArena` (`ctx.frame`): `alloc(device, queue, data, align) -> FrameSlice` for data that changes every frame (sprite instances). It resets each frame. On overflow that frame falls back to a one-off buffer and the arena grows at the next `begin_frame` (`generation` bumps). A bind group built on it must be rebuilt when `generation` changes.

### Writing a new pass
1. Create `pass/my_pass.rs` with `struct MyPass { node_id: Option<NodeId>, …ids, shader, pipeline: Option<…> }` and `MyOutputs`.
2. `impl PassDesc` (`name`, `reads`, `writes`, optionally `layout_entries` / `binding_resources` / `samplers`) and `impl Pass` (`bind_node_id`, `execute`, optionally `on_resize`).
3. Add `pub mod my_pass;` to `pass/mod.rs`, wire it in `pbr/builder.rs`, and make sure a downstream pass reads its output.
4. Add the WGSL to `sandbox/shaders/`, pass it through `RenderGraphPBRBuilder::new` (new parameter), and update **both** `sandbox/src/main.rs` and `editor/src/main.rs`.

## PBR pipeline (`pbr/builder.rs`)

| Pass | Kind | Reads (world / graph) | Writes |
|------|------|------------------------|--------|
| `camera` | upload | first `Camera` | camera uniform (view_proj, inv_view_proj, inv_view) |
| `light_upload` | upload | camera + all lights → **view space** | point/spot/area/sky buffers, counts |
| `bvh_upload` | CPU build + upload | all `Mesh` (rebuilt only when the mesh **count** changes) | BVH nodes/tris/info |
| `probe_atlas` | upload | camera eye, `RenderDebugFlags` DDGI params | probe grid params (grid snaps to `probe_spacing`, scrolls with the camera) |
| `probe_trace` | compute | BVH, probe params, sky lights | per-ray radiance + direction (864 probes × 128 rays) |
| `probe_update` | compute | rays, atlas[current] | atlas[1-current] (irradiance 8², visibility 16², 1px border), then flips `current_atlas` |
| `shadow` | raster | first `SkyLight`, camera, meshes | 4-cascade 2048² depth array + `ShadowParams` (PSSM λ=0.75, sphere-fit, texel-snapped) |
| `gbuffer` | raster | camera, meshes | albedo, normal, material, depth (reverse-Z, clear 0.0) |
| `cluster_assignment` | compute | camera, point + sky lights | 16×9×24 cluster grid, ≤128 lights/cluster |
| `lighting` | fullscreen | gbuffer, clusters, point/spot/area/sky lights + counts, shadows, probe atlas, debug flags | `lit` (HDR) |
| `debug_overlay` | raster | BVH, camera, probe params, debug flags | overlay texture (Rgba16Float): BVH boxes, probes, grid |
| `present` | fullscreen | `lit` + overlay | surface |

Coordinate conventions: right-handed, Y-up, camera looks down -Z. Light data is in **view space** on the GPU. Camera/GBuffer depth is reversed-Z (`GreaterEqual`, clear to 0). Shadow maps use standard Z (`LessEqual`).

## Gotchas
- **Mesh geometry is keyed by `EntityKey`** in `GeometryArena`, and gbuffer material textures by `Arc` pointer. Changing a `Mesh`'s vertices in place is never re-uploaded, but replacing the entity (new key) works. Removed meshes are freed on the next frame. Material textures are still never freed.
- The BVH only rebuilds when the mesh count changes. Moving or editing geometry at a constant count leaves DDGI tracing stale geometry.
- "Main camera" is `get_entities::<Camera>().first()` in several passes. With multiple cameras, which one wins depends on slotmap order.
- WGSL structs/constants are duplicated by hand (no shader includes). Keep the Rust `#[repr(C)]` structs (`ClusterParams`, `ProbeGridParams`, `ShadowParams`, `CameraUniform`, `DebugFlagsGpu`, `Gpu*Light`) and constants (`PROBE_GRID_DIM`, `PROBES_PER_ROW/COL`, `CLUSTER_*`, `NUM_CASCADES`) in sync with `sandbox/shaders/*.wgsl`.
- Only point lights are clustered. Spot (≤64) and area (≤32) lights are looped over for every pixel in `lighting.wgsl`, so keep their counts low. Area lights use a representative-point approximation, are two-sided, and treat `right` / `up` as **half-extent** vectors.
- The GBuffer stores **world-space** normals but local lights are in view space. `lighting.wgsl` converts the normal to view space (`N_vs`) for point/spot/area lights. Sky lights convert `L` to world space instead.
- `pass/sdf_voxelize.rs` is **not** in `pass/mod.rs`, so it is dead code. Wire it in or delete it.
- `Renderer::new` uses `adapter.limits()` and only the timestamp features the adapter supports. The GPU profiler activates only if `TIMESTAMP_QUERY | TIMESTAMP_QUERY_INSIDE_ENCODERS` are present.
- The surface is sRGB, so passes output linear color. The egui overlay gets a non-sRGB view of the same texture.
- `Renderer::resize` ignores 0×0 sizes and clamps to `max_texture_dimension_2d`.
- `RenderDebugFlags` is shared by `Arc<RwLock>` between `DebugLayer` (render-thread UI), `ProbeAtlasPass`, `LightingPass` and `DebugOverlayPass`. To control it from outside, inject your own with `RenderGraphPBRBuilder::with_debug_flags` (the editor does this).

## 2D path (`rendering_2d`)
`SpriteRenderGraphBuilder` builds a single `SpritePass` that writes straight to the surface. A `SpritePass` can also be added to another graph as an overlay (pass `None` as the clear color). It sorts sprites by ascending `z_order` (painter's order), uploads all instances in one `FrameArena` allocation, and draws each run of consecutive sprites sharing a texture as one instance range (the cache is keyed by `Arc<SpriteTexture>` pointer). Instanced quads have per-instance TRS, color and `uv_rect` for atlases. `Camera2D { position, zoom }` gives an orthographic, Y-up projection in pixel units. The first `Camera2D` is used, or the default if there is none.
