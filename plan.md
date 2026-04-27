# Clustered Shading Implementation Plan

## Overview

Implement GPU-based clustered light assignment for the deferred rendering pipeline. This introduces the engine's first compute pass, a scene light buffer, and upgrades the existing `LightingPass` to use per-cluster light lists instead of brute-force iteration.

---

## Phase 1: Compute Pass Support in the Render Graph

**Goal:** Extend the render graph to support compute dispatches alongside existing render passes.

### Tasks

1. **Audit the current `RenderPass` trait** — look at `RenderPassDesc` and `RenderPass` (the executable trait). Determine if the `execute` method is tightly coupled to `wgpu::RenderPass` or if it can accommodate `wgpu::ComputePass`.

2. **Introduce a `PassKind` enum** (or equivalent discriminant):
   ```rust
   enum PassKind {
       Render,
       Compute,
   }
   ```
   Add `fn kind(&self) -> PassKind` to `RenderPassDesc`. The graph executor uses this to decide whether to create a `wgpu::RenderPass` or `wgpu::ComputePass` on the `CommandEncoder`.

3. **Update `PassContext` (or execution context)** — the context passed to `execute()` needs to support both paths. Options:
   - A) `execute` receives an enum: `PassEncoder::Render(&mut wgpu::RenderPass)` / `PassEncoder::Compute(&mut wgpu::ComputePass)` — cleanest but every pass must match.
   - B) Two separate traits: `RenderPass::execute_render(...)` and `ComputePass::execute_compute(...)` with a unified `Pass` enum at the graph level — more type-safe.
   - C) `execute` receives the raw `&mut wgpu::CommandEncoder` and the pass creates its own render/compute pass internally — most flexible, least safe.

   **Recommendation:** Option B gives you the best type safety. The graph stores `enum PassNode { Render(Box<dyn RenderPass>), Compute(Box<dyn ComputePass>) }` and dispatches accordingly. Both share `RenderPassDesc` (or rename to `PassDesc`) for the declarative metadata.

4. **Update the graph executor** — in the topological traversal, check the pass kind and either:
   - Begin a render pass with the declared color/depth attachments, call `execute_render`
   - Begin a compute pass (no attachments), call `execute_compute`

5. **Update `impl_pass!` macro** — add an optional `kind: Compute` field that generates `fn kind() -> PassKind { PassKind::Compute }`. Default to `Render` for backward compat.

### Validation
- All existing passes (`CameraPass`, `DepthPrePass`, `GBufferPass`, `LightingPass`) compile and work unchanged.
- You can add a dummy no-op compute pass to the graph and it dispatches without crashing.

---

## Phase 2: Light Data Structures

**Goal:** Define the GPU-side light representation and the cluster grid parameters.

### Tasks

1. **Define `GpuPointLight` struct** (repr C, bytemuck):
   ```rust
   #[repr(C)]
   #[derive(Copy, Clone, Pod, Zeroable)]
   struct GpuPointLight {
       position_vs: [f32; 3],  // view-space position
       radius: f32,
       color: [f32; 3],
       intensity: f32,
   }
   ```
   Start with point lights only. Spot lights and directional can come later. View-space positions are computed CPU-side each frame (world position × view matrix) so the compute shader doesn't need the view matrix for intersection tests.

2. **Define cluster grid constants:**
   ```rust
   const CLUSTER_X: u32 = 16;
   const CLUSTER_Y: u32 = 9;
   const CLUSTER_Z: u32 = 24;
   const TOTAL_CLUSTERS: u32 = CLUSTER_X * CLUSTER_Y * CLUSTER_Z; // 3456
   const MAX_LIGHTS_PER_CLUSTER: u32 = 128;
   ```
   These can be `const` for now. Making them configurable is a later concern.

3. **Define `ClusterParams` uniform struct:**
   ```rust
   #[repr(C)]
   struct ClusterParams {
       // For converting fragment position to cluster index
       tile_size: [f32; 2],       // screen_size / [CLUSTER_X, CLUSTER_Y]
       z_near: f32,
       z_far: f32,
       grid_dim: [u32; 3],        // CLUSTER_X, CLUSTER_Y, CLUSTER_Z
       num_lights: u32,
       // For logarithmic Z slicing
       log_z_ratio: f32,          // 1.0 / log(z_far / z_near)
       _pad: [f32; 3],
   }
   ```

4. **Define output buffer layouts:**
   - **Light index list:** `Vec<u32>`, size = `TOTAL_CLUSTERS * MAX_LIGHTS_PER_CLUSTER * 4` bytes. Flat array where each cluster gets a fixed-size slot.
   - **Cluster lookup (light grid):** `Vec<[u32; 2]>`, size = `TOTAL_CLUSTERS * 8` bytes. Each entry is `(offset, count)`.
   
   > *Note:* A fixed-size-per-cluster approach wastes memory but avoids a two-pass prefix sum. Fine for a first implementation. Optimize with a global atomic counter + compacted list later.

---

## Phase 3: Light Buffer Upload

**Goal:** Get light data onto the GPU each frame.

### Tasks

1. **Create a `LightManager` (or `LightStore`)** — a CPU-side struct that holds the scene's lights and produces the GPU buffer. This lives outside the render graph (like your camera), since lights come from the scene/ECS.

2. **Each frame:**
   - Transform light positions to view space using the current view matrix
   - Pack into `Vec<GpuPointLight>`
   - Write to a `wgpu::Buffer` (usage `STORAGE | COPY_DST`)

3. **Import into the render graph** — use the same imported resource pattern as your swapchain: `create_imported()` / `update_imported()`. The `ClusterAssignmentPass` declares this as an input binding.

---

## Phase 4: Cluster Assignment Compute Pass

**Goal:** The core pass that assigns lights to clusters on the GPU.

### Tasks

1. **Create `ClusterAssignmentPass`** implementing your new `ComputePass` trait:
   - **Owned resources:**
     - `cluster_light_indices` — storage buffer, `TOTAL_CLUSTERS * MAX_LIGHTS_PER_CLUSTER * 4` bytes
     - `cluster_light_grid` — storage buffer, `TOTAL_CLUSTERS * 8` bytes
   - **Input bindings:**
     - Camera uniforms (from `CameraPass`)
     - Light buffer (imported)
     - Cluster params uniform
   - **Layout entries:** One bind group with the light buffer (read-only storage), cluster params (uniform), and the two output buffers (read-write storage)

2. **Write the WGSL compute shader** (`cluster_assignment.wgsl`):
   ```wgsl
   @group(0) @binding(0) var<uniform> cluster_params: ClusterParams;
   @group(0) @binding(1) var<storage, read> lights: array<PointLight>;
   @group(0) @binding(2) var<storage, read_write> light_grid: array<vec2<u32>>;
   @group(0) @binding(3) var<storage, read_write> light_indices: array<u32>;

   // One workgroup per cluster
   @compute @workgroup_size(64, 1, 1)  // 64 threads per cluster, each tests a subset of lights
   fn main(
       @builtin(workgroup_id) cluster_id: vec3<u32>,
       @builtin(local_invocation_index) thread_idx: u32,
   ) {
       // 1. Compute cluster AABB in view space
       //    - X/Y from screen tile → inverse projection
       //    - Z from logarithmic slice formula
       //
       // 2. Each thread tests lights[thread_idx], lights[thread_idx + 64], etc.
       //    - Sphere-AABB intersection test
       //    - If hit, atomically increment a shared counter and write to shared list
       //
       // 3. Barrier, then thread 0 writes the compacted list to global buffers
   }
   ```

   Key implementation details:
   - **Cluster AABB computation:** Convert cluster grid coords to view-space min/max. The X/Y bounds come from the tile's screen-space extent projected to the near/far of the Z slice. The Z bounds come from the log slice formula.
   - **Sphere-AABB test:** Standard closest-point-on-AABB-to-sphere-center distance test. If distance² < radius², the light intersects.
   - **Shared memory:** Use `var<workgroup>` for a local light count and local light index array. After all threads finish, thread 0 copies to global memory.

3. **`execute_compute` implementation:**
   ```rust
   fn execute_compute(&mut self, ctx: &PassContext, pass: &mut wgpu::ComputePass) {
       pass.set_pipeline(&self.pipeline);
       pass.set_bind_group(0, &self.bind_group, &[]);
       pass.dispatch_workgroups(CLUSTER_X, CLUSTER_Y, CLUSTER_Z);
   }
   ```
   One workgroup per cluster = `16 × 9 × 24 = 3456` workgroups.

---

## Phase 5: Update the Lighting Pass

**Goal:** Wire the cluster data into the existing lighting pass shader.

### Tasks

1. **Add new input bindings to `LightingPass`:**
   - Light buffer (read-only storage)
   - Cluster light grid (read-only storage)
   - Cluster light indices (read-only storage)
   - Cluster params (uniform)

   These come as `ResourceHandle`s from the `ClusterAssignmentPass` outputs and the imported light buffer.

2. **Update the lighting WGSL shader:**
   ```wgsl
   // Fragment shader
   fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
       // Read GBuffer
       let albedo = ...;
       let normal = ...;
       let depth = ...;

       // Reconstruct view-space position from depth
       let pos_vs = reconstruct_view_position(frag_coord.xy, depth);

       // Determine which cluster this fragment belongs to
       let tile = vec2<u32>(
           u32(frag_coord.x) / u32(cluster_params.tile_size.x),
           u32(frag_coord.y) / u32(cluster_params.tile_size.y),
       );
       let z_view = /* linear depth from reversed-Z */;
       let slice = u32(log(z_view / cluster_params.z_near) * cluster_params.log_z_ratio * f32(cluster_params.grid_dim.z));
       let cluster_idx = tile.x + tile.y * cluster_params.grid_dim.x
                       + slice * cluster_params.grid_dim.x * cluster_params.grid_dim.y;

       // Read light list for this cluster
       let grid = light_grid[cluster_idx]; // (offset, count)
       var color = vec3<f32>(0.0);
       for (var i = 0u; i < grid.y; i = i + 1u) {
           let light_idx = light_indices[grid.x + i];
           let light = lights[light_idx];
           color += evaluate_point_light(light, pos_vs, normal, albedo);
       }

       return vec4<f32>(color, 1.0);
   }
   ```

3. **Add the graph dependency** — `LightingPass` now connects to `ClusterAssignmentPass` outputs via `binding_resources()`. The graph's topological sort will automatically schedule cluster assignment before lighting.

---

## Phase 6: Debug Visualization

**Goal:** Verify the clustering is working correctly.

### Tasks

1. **Cluster heatmap debug mode** — a variant of the lighting shader that outputs `light_count / MAX_LIGHTS_PER_CLUSTER` as a color (cold blue → hot red). This immediately tells you if lights are being assigned correctly and where hotspots are.

2. **Cluster grid wireframe** (optional) — render the cluster boundaries as lines for a specific Z slice. Useful but lower priority.

---

## Execution Order (in the render graph)

```
CameraPass
    ↓
DepthPrePass
    ↓
GBufferPass
    ↓
ClusterAssignmentPass  ← NEW (compute, depends on CameraPass + imported light buffer)
    ↓
LightingPass           ← UPDATED (depends on GBuffer + ClusterAssignment outputs)
```

Note: `ClusterAssignmentPass` only depends on the camera (for projection info) and the light buffer — it does NOT depend on `GBufferPass`. The graph could theoretically schedule it in parallel with the GBuffer pass if you have async compute support later. For now, sequential is fine.

---

## Testing Strategy

1. **Single light, center of screen** — place one point light and verify it appears in the expected cluster(s). Use the heatmap debug view.
2. **Light at cluster boundary** — verify a light at the edge of a cluster correctly appears in adjacent clusters.
3. **Many overlapping lights** — stress test with 100+ lights in a small area. Check the heatmap shows high counts and rendering is correct.
4. **Camera movement** — verify clusters update correctly as the camera moves (lights should stay fixed in world space but their view-space positions update).
5. **Edge cases** — lights behind the camera, lights outside the far plane, lights exactly on a cluster boundary.

---

## Future Optimizations (NOT part of this plan)

- Two-pass prefix sum for compact light index lists (eliminate fixed max per cluster)
- Async compute overlap with GBuffer pass
- Spot light support (cone-AABB intersection)
- Directional light handling (always-present, no cluster assignment)
- Depth bounds optimization (use Hi-Z to tighten cluster Z ranges per tile)
- Light BVH for faster intersection with very high light counts
