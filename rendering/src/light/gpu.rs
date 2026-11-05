#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuPointLight {
    pub position_vs: [f32; 3],
    pub radius: f32,
    pub color: [f32; 3],
    pub intensity: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSpotLight {
    pub position_vs: [f32; 3],
    pub radius: f32,
    pub direction_vs: [f32; 3],
    pub inner_cos: f32,
    pub color: [f32; 3],
    pub outer_cos: f32,
    pub intensity: f32,
    pub _pad: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuAreaLight {
    pub position_vs: [f32; 3],
    pub intensity: f32,
    pub right_vs: [f32; 3],
    /// 1 = emits from both faces, 0 = only toward `right × up`.
    pub two_sided: f32,
    pub up_vs: [f32; 3],
    pub _pad1: f32,
    pub color: [f32; 3],
    pub _pad2: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSkyLight {
    pub direction_vs: [f32; 3],
    pub intensity: f32,
    pub color: [f32; 3],
    pub _pad: f32,
}

/// Local lights that get a cube shadow map, one slot each. An entry is `kind << 16 | index` into
/// that kind's light buffer (`SHADOW_KIND_*`). `lights[slot]` is the shadow origin in world space
/// (xyz) and the light's half size (w, 0 for point and spot lights), which widens the PCF. Slots
/// can be empty (`SHADOW_ENTRY_NONE`); `count` is one past the last used slot.
#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowedLights {
    pub entries: [u32; MAX_SHADOWED_LIGHTS],
    pub count: u32,
    pub _pad: [u32; 3],
    pub lights: [[f32; 4]; MAX_SHADOWED_LIGHTS],
}

/// What the shadow pass needs to render a slot's cube map. Shared from `light_upload` to `shadow`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ShadowCaster {
    pub entry: u32,
    pub origin: ultraviolet::Vec3,
    /// Far plane of the cube faces: the light's range.
    pub range: f32,
}

pub type SharedShadowCasters = std::sync::Arc<std::sync::Mutex<[Option<ShadowCaster>; MAX_SHADOWED_LIGHTS]>>;

pub const MAX_SHADOWED_LIGHTS: usize = 4;
pub const SHADOW_KIND_POINT: u32 = 0;
pub const SHADOW_KIND_SPOT: u32 = 1;
pub const SHADOW_KIND_AREA: u32 = 2;
pub const SHADOW_ENTRY_NONE: u32 = u32::MAX;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightCounts {
    pub num_point: u32,
    pub num_spot: u32,
    pub num_area: u32,
    pub num_sky: u32,
}
