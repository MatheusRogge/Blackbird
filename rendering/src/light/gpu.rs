
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
    /// `light_upload::cull_range`: GI (`probe_trace`) ignores the light past it.
    pub cull_range: f32,
    pub _pad: f32,
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
    /// As `GpuPointLight::cull_range`.
    pub cull_range: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuAreaLight {
    pub position_vs: [f32; 3],
    pub intensity: f32,
    pub right_vs: [f32; 3],
    /// `AREA_TWO_SIDED` (else it emits only toward `right × up`) | `AREA_EMISSIVE`.
    pub flags: u32,
    pub up_vs: [f32; 3],
    /// Index into `ShadowedLights`, or `SHADOW_SLOT_NONE`.
    pub shadow_slot: u32,
    pub color: [f32; 3],
    /// Distance from the rectangle beyond which shading ignores it, fading it out over the last
    /// quarter; `f32::MAX` for none.
    pub range: f32,
}

pub const AREA_TWO_SIDED: u32 = 1;
/// The light of an emissive mesh. These get a range: unranged, their highlights washed over the
/// reflections of Osaka's signs, which the reflection passes already show.
pub const AREA_EMISSIVE: u32 = 2;

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
/// `count` is one past the last used slot. Slots get a quality tier by importance, for `lighting`'s
/// deferred pass: the `SHADOW_MASK_LAYERS` most important a layer of its half-resolution shadow
/// mask (PCSS and contact shadows), the next `FILTERED_SHADOW_LIGHTS` a fixed small PCF kernel, the
/// rest one bilinear tap; neither of the latter marches contact shadows. `mask_slots[layer]` is the
/// slot in a layer and `tier[slot / 4][slot % 4]` a slot's tier: its layer, `SHADOW_TIER_FILTERED`,
/// or `SHADOW_SLOT_NONE`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowedLights {
    pub count: u32,
    pub _pad: [u32; 3],
    pub mask_slots: [u32; SHADOW_MASK_LAYERS],
    pub tier: [[u32; 4]; MAX_SHADOWED_LIGHTS / 4],
}

impl ShadowedLights {
    /// Tiers `slots` in order (most important first).
    pub fn new(count: u32, slots: impl IntoIterator<Item = u32>) -> Self {
        let mut lights = Self {
            count,
            _pad: [0; 3],
            mask_slots: [SHADOW_SLOT_NONE; SHADOW_MASK_LAYERS],
            tier: [[SHADOW_SLOT_NONE; 4]; MAX_SHADOWED_LIGHTS / 4],
        };
        for (rank, slot) in slots.into_iter().take(SHADOW_MASK_LAYERS + FILTERED_SHADOW_LIGHTS).enumerate() {
            let tier = if rank < SHADOW_MASK_LAYERS {
                lights.mask_slots[rank] = slot;
                rank as u32
            } else {
                SHADOW_TIER_FILTERED
            };
            lights.tier[slot as usize / 4][slot as usize % 4] = tier;
        }
        lights
    }
}

impl Default for ShadowedLights {
    fn default() -> Self {
        Self::new(0, [])
    }
}

/// A single-face spot light shadow: the light's direction and tan of half the field of view.
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
    /// Far plane of the faces: the light's range, rounded up to a power of two
    /// (`shadow_slots::shadow_range`).
    pub range: f32,
    /// `None`: a cube, six faces. `Some`: one face.
    pub spot: Option<SpotShadow>,
    /// The emitting side of a one-sided light: cube faces wholly behind it get no map.
    pub front: Option<ultraviolet::Vec3>,
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

    /// Which of its six face slots get maps.
    pub fn faces_enabled(&self) -> [bool; 6] {
        std::array::from_fn(|f| {
            f < self.faces() && !self.front.is_some_and(|n| self.spot.is_none() && super::virtual_shadow::face_behind(f, n))
        })
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
/// per slot). Lights share the physical pages of one atlas (`virtual_shadow`), so this only bounds
/// the uniforms, the page table and the per-frame selection.
pub const MAX_SHADOWED_LIGHTS: usize = 32;
pub const SHADOW_KIND_POINT: u32 = 0;
pub const SHADOW_KIND_SPOT: u32 = 1;
pub const SHADOW_KIND_AREA: u32 = 2;
pub const SHADOW_SLOT_NONE: u32 = u32::MAX;
/// Layers of `lighting`'s half-resolution shadow mask (one shadowed light each). Mirrored in
/// `lighting.wgsl`.
pub const SHADOW_MASK_LAYERS: usize = 4;
/// Shadowed lights after the masked ones that get a small PCF kernel instead of one tap
/// (`ShadowedLights`).
pub const FILTERED_SHADOW_LIGHTS: usize = 8;
/// Mirrored in `lighting.wgsl`.
pub const SHADOW_TIER_FILTERED: u32 = u32::MAX - 1;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightCounts {
    pub num_point: u32,
    pub num_spot: u32,
    pub num_area: u32,
    pub num_sky: u32,
}
