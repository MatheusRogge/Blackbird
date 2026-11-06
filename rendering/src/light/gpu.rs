use super::shadow_atlas::Tile;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuPointLight {
    pub position_vs: [f32; 3],
    pub radius: f32,
    pub color: [f32; 3],
    pub intensity: f32,
    pub source_radius: f32,
    /// Index into `ShadowedLights`, or `SHADOW_SLOT_NONE`.
    pub shadow_slot: u32,
    pub _pad: [f32; 2],
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
    pub source_radius: f32,
    /// Index into `ShadowedLights`, or `SHADOW_SLOT_NONE`.
    pub shadow_slot: u32,
    pub _pad: f32,
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
    /// Index into `ShadowedLights`, or `SHADOW_SLOT_NONE`.
    pub shadow_slot: u32,
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

/// Local lights with a shadow map in the atlas, one slot each; lights point at theirs through
/// `shadow_slot`. Everything about a slot's map (origin, planes, tiles) is in the `shadow` pass's
/// `LocalShadowParams`, written when the map is rendered, since rendering can lag the light.
/// `count` is one past the last used slot.
#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowedLights {
    pub count: u32,
    pub _pad: [u32; 3],
}

/// A single-tile spot light shadow: the light's direction and tan of half the field of view.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SpotShadow {
    pub direction: ultraviolet::Vec3,
    pub tan_half_fov: f32,
}

/// What the shadow pass needs to render a slot. Shared from `light_upload` to `shadow`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ShadowCaster {
    /// `kind << 16 | index` into that kind's light list (`SHADOW_KIND_*`), CPU-side identity.
    pub entry: u32,
    pub origin: ultraviolet::Vec3,
    /// Far plane of the faces: the light's range.
    pub range: f32,
    /// `None`: a cube, six faces in `tiles`. `Some`: one face in `tiles[0]`.
    pub spot: Option<SpotShadow>,
    pub tiles: [Tile; 6],
    /// The light's half size (half extent for area lights, source radius for point and spot
    /// lights), which sets the penumbra.
    pub half_size: f32,
    /// Rank among this frame's shadowed lights, 0 the most important. Orders re-renders.
    pub rank: u32,
}

/// Near plane of every local light face.
const LOCAL_SHADOW_NEAR: f32 = 0.02;

impl ShadowCaster {
    pub fn faces(&self) -> usize {
        if self.spot.is_some() { 1 } else { 6 }
    }

    /// Near and far planes of its faces' projections.
    pub fn planes(&self) -> (f32, f32) {
        (LOCAL_SHADOW_NEAR, self.range.max(LOCAL_SHADOW_NEAR * 2.0))
    }

    /// Whether this light's maps would be drawn differently from `other`'s (rank aside).
    pub fn renders_like(&self, other: &Self) -> bool {
        Self { rank: 0, ..*self } == Self { rank: 0, ..*other }
    }
}

pub type SharedShadowCasters = std::sync::Arc<std::sync::Mutex<[Option<ShadowCaster>; MAX_SHADOWED_LIGHTS]>>;

/// Mirrored by `MAX_SHADOWED_LIGHTS` in `lighting.wgsl` (with `LocalShadowParams` holding 6 faces
/// per slot). Lights share the tiles of one atlas (`shadow_atlas`), so this only bounds the
/// uniforms and the per-frame selection.
pub const MAX_SHADOWED_LIGHTS: usize = 32;
pub const SHADOW_KIND_POINT: u32 = 0;
pub const SHADOW_KIND_SPOT: u32 = 1;
pub const SHADOW_KIND_AREA: u32 = 2;
pub const SHADOW_SLOT_NONE: u32 = u32::MAX;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightCounts {
    pub num_point: u32,
    pub num_spot: u32,
    pub num_area: u32,
    pub num_sky: u32,
}
