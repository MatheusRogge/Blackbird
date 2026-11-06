// Hi-Z depth pyramid for SSR (rendering::pass::ssr). Reversed-Z: each texel holds the maximum, i.e.
// the nearest depth, of the level-0 texels it covers, so a ray in front of it is in front of all of them.

@group(0) @binding(0) var t_depth: texture_depth_2d;
@group(0) @binding(1) var t_prev: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_copy(@builtin(position) pos: vec4<f32>) -> @location(0) f32 {
    return textureLoad(t_depth, vec2<i32>(pos.xy), 0);
}

// 2×2 max of the level above. When that level has an odd size, the last row / column also takes
// the extra texel, so every source texel is covered.
@fragment
fn fs_reduce(@builtin(position) pos: vec4<f32>) -> @location(0) f32 {
    let src = vec2<i32>(textureDimensions(t_prev));
    let base = vec2<i32>(pos.xy) * 2;
    let last = src - 1;
    let dst = max(src / 2, vec2<i32>(1));
    let extra = vec2<i32>(
        select(1, 2, (src.x & 1) == 1 && i32(pos.x) == dst.x - 1),
        select(1, 2, (src.y & 1) == 1 && i32(pos.y) == dst.y - 1),
    );
    var d = 0.0;
    for (var y = 0; y <= extra.y; y++) {
        for (var x = 0; x <= extra.x; x++) {
            d = max(d, textureLoad(t_prev, min(base + vec2<i32>(x, y), last), 0).r);
        }
    }
    return d;
}
