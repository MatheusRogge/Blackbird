// Per-tile depth range for rendering::pass::depth_readback: each texel covers a TILE × TILE block
// of the depth buffer and holds its nearest (x) and farthest (y) non-sky depth, reversed-Z (0 =
// none).

const TILE: i32 = 16;

@group(0) @binding(0) var t_depth: texture_depth_2d;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_tiles(@builtin(position) pos: vec4<f32>) -> @location(0) vec2<f32> {
    let size = vec2<i32>(textureDimensions(t_depth));
    let base = vec2<i32>(pos.xy) * TILE;
    var nearest = 0.0;
    var farthest = 1.0;
    for (var y = 0; y < TILE; y++) {
        for (var x = 0; x < TILE; x++) {
            let p = base + vec2<i32>(x, y);
            if any(p >= size) {
                continue;
            }
            let d = textureLoad(t_depth, p, 0);
            if d > 0.0 {
                nearest = max(nearest, d);
                farthest = min(farthest, d);
            }
        }
    }
    return vec2f(nearest, select(0.0, farthest, nearest > 0.0));
}
