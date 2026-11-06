// Mirrors rendering::pass::shadow::ShadowView.
struct LightUniform {
    view_proj: mat4x4<f32>,
    // w = 0: xyz is the direction light travels (cascades); w = 1: xyz is the light origin (cube faces).
    light: vec4<f32>,
}

// Mirrors rendering::arena::GpuModel.
struct Model {
    model: mat4x4<f32>,
    normal: mat4x4<f32>,
    // x: alpha cutoff (0 = never discard), y: 1 if double-sided, z: metallic, w: roughness.
    material: vec4<f32>,
    base_color: vec4<f32>,
    emissive: vec4<f32>,
    // x: transmission, y: IOR, z: volume thickness (mesh-local), w: attenuation distance (0 = none).
    transmission: vec4<f32>,
    // rgb: volume attenuation color.
    attenuation: vec4<f32>,
}

@group(0) @binding(0) var<uniform> light: LightUniform;
@group(1) @binding(0) var<storage, read> models: array<Model>;

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @builtin(instance_index) instance: u32,
) -> @builtin(position) vec4<f32> {
    return light.view_proj * models[instance].model * vec4<f32>(position, 1.0);
}

// Alpha-tested and double-sided meshes: drawn without culling, discarding cut-out texels.
@group(2) @binding(0) var albedo_tex: texture_2d<f32>;
@group(2) @binding(1) var albedo_sampler: sampler;

struct MaskedOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) cutoff: f32,
    @location(2) @interpolate(flat) alpha: f32,
}

@vertex
fn vs_masked(
    @location(0) position: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @builtin(instance_index) instance: u32,
) -> MaskedOutput {
    let m = models[instance];
    var out: MaskedOutput;
    out.position = light.view_proj * m.model * vec4<f32>(position, 1.0);
    out.uv = uv;
    out.cutoff = m.material.x;
    out.alpha = m.base_color.a;
    return out;
}

@fragment
fn fs_masked(in: MaskedOutput) {
    if textureSample(albedo_tex, albedo_sampler, in.uv).a * in.alpha < in.cutoff {
        discard;
    }
}

// Blended meshes (glass) into the glass tint map, cleared to (1, 1, 1, GLASS_TINT_CLEAR_DEPTH).
// rgb multiplies (dst × src), so stacked layers multiply their transmittance; a keeps the min, the
// depth of the glass nearest the light: clip z for the ortho cascades, clip w (distance along the
// face axis) for the perspective cube faces, which keeps it linear in a 16-bit float.
struct GlassOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) depth: f32,
    @location(3) @interpolate(flat) instance: u32,
    @location(4) world_pos: vec3<f32>,
    @location(5) world_normal: vec3<f32>,
}

@vertex
fn vs_glass(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @builtin(instance_index) instance: u32,
) -> GlassOutput {
    let m = models[instance];
    let world = m.model * vec4<f32>(position, 1.0);
    let clip = light.view_proj * world;
    var out: GlassOutput;
    out.position = clip;
    out.uv = uv;
    out.color = color;
    out.depth = select(clip.z, clip.w, light.light.w > 0.5);
    out.instance = instance;
    out.world_pos = world.xyz / world.w;
    out.world_normal = (m.normal * vec4f(normal, 0.0)).xyz;
    return out;
}

// The fraction of light one surface lets through: the uncovered part, plus the covered part's
// transmission tinted by the base color, minus what Fresnel reflects at the light's actual angle
// of incidence. That angle is what makes clear glass cast a visible shadow: curved walls seen
// edge-on by the light reflect most of it, leaving a dark rim. A volume is drawn twice (entry and
// exit faces), so each face takes the square root of its tint and Beer–Lambert absorption over the
// thickness; a thin-walled surface is drawn once but has two interfaces. Metallic is ignored: only its factor is bound here,
// and glTF glass that relies on a metallic-roughness texture often leaves the factor at its default 1.
@fragment
fn fs_glass(in: GlassOutput) -> @location(0) vec4<f32> {
    let m = models[in.instance];
    let base = textureSample(albedo_tex, albedo_sampler, in.uv) * m.base_color * vec4f(in.color, 1.0);
    let transmission = m.transmission.x;
    let ior = m.transmission.y;
    let f0 = pow((ior - 1.0) / (ior + 1.0), 2.0);

    let scale = (length(m.model[0].xyz) + length(m.model[1].xyz) + length(m.model[2].xyz)) / 3.0;
    let thickness = m.transmission.z * scale;
    var tint = base.rgb;
    if thickness > 0.0 {
        if m.transmission.w > 0.0 {
            tint *= pow(m.attenuation.rgb, vec3f(thickness / m.transmission.w));
        }
        tint = sqrt(tint);
    }
    let L = select(light.light.xyz, in.world_pos - light.light.xyz, light.light.w > 0.5);
    let cos_i = abs(dot(normalize(in.world_normal), normalize(L)));
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - cos_i, 5.0);
    let interfaces = select(1.0, 2.0, thickness <= 0.0);
    let through = (1.0 - base.a) + base.a * transmission * pow(1.0 - fresnel, interfaces) * tint;
    return vec4f(through, in.depth);
}
