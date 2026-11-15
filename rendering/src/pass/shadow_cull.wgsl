// GPU culling of the opaque shadow casters against each local shadow page rendered this frame
// (rendering::pass::shadow_cull). Every page gets a region of `instance_ids` (`casters` entries) and
// `groups` draw commands, compacted to the non-empty ones and counted in `draw_counts`.

// Mirrors rendering::pass::shadow_cull::GpuCaster.
struct Caster {
    min: vec3f,
    instance: u32,
    max: vec3f,
    group: u32,
}

// Mirrors rendering::pass::shadow_cull::GpuGroup: casters sharing geometry, drawn instanced.
struct Group {
    index_count: u32,
    first_index: u32,
    base_vertex: i32,
    // Where the group's instance ids start in a page's region.
    offset: u32,
}

// Mirrors rendering::pass::shadow_cull::Params.
struct Params {
    casters: u32,
    groups: u32,
    pages: u32,
    _pad: u32,
}

// wgpu::util::DrawIndexedIndirectArgs.
struct DrawArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> casters: array<Caster>;
@group(0) @binding(2) var<storage, read> groups: array<Group>;
// Six world-space planes per page (bounds::Frustum).
@group(0) @binding(3) var<storage, read> planes: array<vec4f>;
// Visible casters per (page, group); cleared before `cs_cull`.
@group(0) @binding(4) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> instance_ids: array<u32>;
@group(0) @binding(6) var<storage, read_write> draws: array<DrawArgs>;
// Draw commands per page; cleared before `cs_commands`.
@group(0) @binding(7) var<storage, read_write> draw_counts: array<atomic<u32>>;

// x: caster, y: page.
@compute @workgroup_size(64)
fn cs_cull(@builtin(global_invocation_id) id: vec3u) {
    let page = id.y;
    if id.x >= params.casters || page >= params.pages {
        return;
    }
    let caster = casters[id.x];
    for (var i = 0u; i < 6u; i++) {
        let plane = planes[page * 6u + i];
        let corner = select(caster.min, caster.max, plane.xyz >= vec3f(0.0));
        if plane.x * corner.x + plane.y * corner.y + plane.z * corner.z + plane.w < 0.0 {
            return;
        }
    }
    let n = atomicAdd(&counts[page * params.groups + caster.group], 1u);
    instance_ids[page * params.casters + groups[caster.group].offset + n] = caster.instance;
}

// x: group, y: page.
@compute @workgroup_size(64)
fn cs_commands(@builtin(global_invocation_id) id: vec3u) {
    let page = id.y;
    if id.x >= params.groups || page >= params.pages {
        return;
    }
    let n = atomicLoad(&counts[page * params.groups + id.x]);
    if n == 0u {
        return;
    }
    let group = groups[id.x];
    let k = atomicAdd(&draw_counts[page], 1u);
    draws[page * params.groups + k] =
        DrawArgs(group.index_count, n, group.first_index, group.base_vertex, page * params.casters + group.offset);
}
