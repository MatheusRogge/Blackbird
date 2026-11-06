// Mip chain and GGX prefilter for the camera-following reflection probe (rendering::pass::reflection_probe).
// Faces follow the Vulkan cube convention (+X −X +Y −Y +Z −Z), so `texture_cube` lookups see them seamlessly.

// Mirrors rendering::pass::reflection_probe::FilterParams.
struct FilterParams {
    face: u32,
    // Perceptual roughness of the prefiltered level being written.
    roughness: f32,
    // Face size of the capture (level 0), for the filtered-importance-sampling LOD.
    src_size: f32,
    // Face size of the level being written.
    dst_size: f32,
}

@group(0) @binding(0) var src_level: texture_2d_array<f32>;
@group(0) @binding(1) var src_cube: texture_cube<f32>;
@group(0) @binding(2) var src_sampler: sampler;
@group(0) @binding(3) var<uniform> params: FilterParams;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
    var out: VertexOutput;
    out.pos = vec4f(uv * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0);
    out.uv = uv;
    return out;
}

// One bilinear tap at the destination texel center: the 2×2 box of the level above. Alpha (the
// captured distance) is averaged the same way.
@fragment
fn fs_downsample(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSampleLevel(src_level, src_sampler, in.uv, i32(params.face), 0.0);
}

// The direction through `uv` of `face`: forward + sc·right − tc·up, with sc, tc in [−1, 1].
fn face_dir(face: u32, uv: vec2<f32>) -> vec3<f32> {
    let s = uv.x * 2.0 - 1.0;
    let t = uv.y * 2.0 - 1.0;
    switch face {
        case 0u: { return normalize(vec3f(1.0, -t, -s)); }
        case 1u: { return normalize(vec3f(-1.0, -t, s)); }
        case 2u: { return normalize(vec3f(s, 1.0, t)); }
        case 3u: { return normalize(vec3f(s, -1.0, -t)); }
        case 4u: { return normalize(vec3f(s, -t, 1.0)); }
        default: { return normalize(vec3f(-s, -t, -1.0)); }
    }
}

const PI: f32 = 3.14159265359;
const SAMPLES: u32 = 64u;

fn hammersley(i: u32, n: u32) -> vec2<f32> {
    var bits = i;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return vec2f(f32(i) / f32(n), f32(bits) * 2.3283064365386963e-10);
}

// GGX prefilter with N = V = R (split-sum assumption), filtered importance sampling (Křivánek &
// Colbert): each sample reads the capture mip whose texel solid angle matches the sample's, so 64
// samples don't alias. Level 0 (roughness 0) is a plain resample. Alpha keeps the captured distance
// along the texel's direction, for parallax correction.
@fragment
fn fs_prefilter(in: VertexOutput) -> @location(0) vec4<f32> {
    let N = face_dir(params.face, in.uv);
    let base_lod = log2(params.src_size / params.dst_size);
    let distance = textureSampleLevel(src_cube, src_sampler, N, base_lod).a;
    if params.roughness <= 0.0 {
        return vec4f(textureSampleLevel(src_cube, src_sampler, N, base_lod).rgb, distance);
    }

    let alpha = params.roughness * params.roughness;
    let a2 = alpha * alpha;
    let up = select(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), abs(N.z) > 0.999);
    let T = normalize(cross(up, N));
    let B = cross(N, T);
    let texel_solid_angle = 4.0 * PI / (6.0 * params.src_size * params.src_size);

    var color = vec3f(0.0);
    var weight = 0.0;
    for (var i = 0u; i < SAMPLES; i++) {
        let xi = hammersley(i, SAMPLES);
        let phi = 2.0 * PI * xi.x;
        let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a2 - 1.0) * xi.y));
        let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
        let H = normalize(T * (sin_theta * cos(phi)) + B * (sin_theta * sin(phi)) + N * cos_theta);
        let L = 2.0 * dot(N, H) * H - N;
        let n_dot_l = dot(N, L);
        if n_dot_l <= 0.0 {
            continue;
        }
        // pdf of L = D(H)·(N·H) / (4·V·H), with V = N.
        let d = (cos_theta * cos_theta) * (a2 - 1.0) + 1.0;
        let D = a2 / (PI * d * d);
        let pdf = D / 4.0;
        let sample_solid_angle = 1.0 / (f32(SAMPLES) * pdf + 1e-6);
        let lod = max(0.5 * log2(sample_solid_angle / texel_solid_angle) + 1.0, 0.0);
        color += textureSampleLevel(src_cube, src_sampler, L, lod).rgb * n_dot_l;
        weight += n_dot_l;
    }
    return vec4f(color / max(weight, 1e-4), distance);
}
