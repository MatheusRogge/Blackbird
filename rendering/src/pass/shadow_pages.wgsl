// Page requests for local light virtual shadow maps (rendering::pass::shadow_pages): every fourth
// visible pixel on each axis asks, for each shadowed light in range, for the page of its face at
// the mip whose texel matches the pixel's footprint, and the two coarser mips above it (the PCSS
// filter moves to coarser mips for wide penumbrae), widened by the filter's reach. The widening
// (KERNEL_TEXELS at that mip, at least 8 pixels) also covers the pixels in between: at half
// resolution the pass cost 4.4 ms in Osaka for the same images. Mirrors the page layout of
// rendering::light::virtual_shadow and the mip choice of `light_shadow` in lighting.wgsl.

const VIRTUAL_SIZE: f32 = 8192.0;
const PAGES0: u32 = 64u;
const MIPS: u32 = 7u;
const PAGES_PER_FACE: u32 = 5461u;
const MAX_SHADOWED_LIGHTS: u32 = 32u;
// Mirrors MAX_PENUMBRA_TEXELS in lighting.wgsl: the filter's reach around a lookup, in texels.
const KERNEL_TEXELS: f32 = 16.0;
// Mirrors PIXEL_STEP in rendering::pass::shadow_pages: one thread per this many pixels per axis.
const PIXEL_STEP: u32 = 4u;
// Requested mips: the footprint's and this many coarser ones.
const EXTRA_MIPS: u32 = 2u;

// Mirrors the prefix of rendering::pass::camera::CameraUniform.
struct CameraUniform {
    view_proj:     mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    inv_view:      mat4x4<f32>,
    depth_params:  vec4<f32>,
}

// Mirrors rendering::pass::shadow::LocalShadowFace / LocalShadowParams.
struct LocalShadowFace {
    view_proj: mat4x4<f32>,
    info:      vec4<u32>,
    light:     vec4<f32>,
    shape:     vec4<f32>,
}

struct LocalShadowParams {
    lod:   vec4<f32>,
    faces: array<LocalShadowFace, 192>,
}

@group(0) @binding(0) var t_depth: texture_depth_2d;
@group(0) @binding(1) var<uniform> camera: CameraUniform;
@group(0) @binding(2) var<uniform> local_shadow: LocalShadowParams;
@group(0) @binding(3) var<storage, read_write> requests: array<atomic<u32>>;

// Pages of the mips before `mip`: Σ (64 >> m)² = 4096 · (1 − 4^−mip) / (3/4), exact in integers
// (as `vsm_mip_offset` in lighting.wgsl).
fn mip_offset(mip: u32) -> u32 {
    return (16384u - (16384u >> (2u * mip))) / 3u;
}

fn request(face_index: u32, mip: u32, page: vec2<u32>) {
    let n = PAGES0 >> mip;
    let index = face_index * PAGES_PER_FACE + mip_offset(mip) + page.y * n + page.x;
    atomicOr(&requests[index >> 5u], 1u << (index & 31u));
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(t_depth);
    let pixel = id.xy * PIXEL_STEP;
    if any(pixel >= size) {
        return;
    }
    let depth = textureLoad(t_depth, vec2<i32>(pixel), 0);
    if depth == 0.0 {
        return;
    }
    let ndc = vec2f((f32(pixel.x) + 0.5) / f32(size.x) * 2.0 - 1.0, 1.0 - (f32(pixel.y) + 0.5) / f32(size.y) * 2.0);
    let h = camera.inv_view_proj * vec4f(ndc, depth, 1.0);
    let pos = h.xyz / h.w;
    let eye = camera.inv_view[3].xyz;
    let forward = -camera.inv_view[2].xyz;
    let footprint = dot(pos - eye, forward) * local_shadow.lod.x;

    for (var slot = 0u; slot < MAX_SHADOWED_LIGHTS; slot++) {
        let first = local_shadow.faces[slot * 6u];
        let shape = first.shape;
        if shape.x == 0.0 {
            continue;
        }
        let d = pos - first.light.xyz;
        if dot(d, d) > shape.w * shape.w {
            continue;
        }
        var face = 0u;
        if shape.x > 1.5 {
            let a = abs(d);
            if a.x >= a.y && a.x >= a.z {
                face = select(1u, 0u, d.x > 0.0);
            } else if a.y >= a.z {
                face = select(3u, 2u, d.y > 0.0);
            } else {
                face = select(5u, 4u, d.z > 0.0);
            }
        }
        let face_index = slot * 6u + face;
        let f = local_shadow.faces[face_index];
        if f.info.x == 0u {
            continue;
        }
        let clip = f.view_proj * vec4f(pos, 1.0);
        if clip.w <= 0.0 {
            continue;
        }
        let uv = (clip.xy / clip.w) * vec2f(0.5, -0.5) + 0.5;
        if any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) {
            continue;
        }
        let texel0 = 2.0 * clip.w * shape.y / VIRTUAL_SIZE;
        let mip = u32(clamp(floor(log2(max(footprint / texel0, 1.0))), 0.0, f32(MIPS - 1u)));
        for (var m = mip; m <= min(mip + EXTRA_MIPS, MIPS - 1u); m++) {
            let n = PAGES0 >> m;
            let reach = KERNEL_TEXELS / (VIRTUAL_SIZE / f32(1u << m));
            let lo = vec2<u32>(clamp((uv - reach) * f32(n), vec2f(0.0), vec2f(f32(n - 1u))));
            let hi = vec2<u32>(clamp((uv + reach) * f32(n), vec2f(0.0), vec2f(f32(n - 1u))));
            for (var y = lo.y; y <= hi.y; y++) {
                for (var x = lo.x; x <= hi.x; x++) {
                    request(face_index, m, vec2<u32>(x, y));
                }
            }
        }
    }
}
