use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DebugFlagsGpu {
    pub disable_albedo:   u32,
    pub disable_indirect: u32,
    pub disable_direct:   u32,
    pub disable_shadows:  u32,
    pub view:             u32,
    pub disable_specular: u32,
    pub disable_reflections: u32,
    pub disable_ssr:      u32,
}

/// What the `lighting` pass outputs: the lit image, or one G-buffer target as-is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GBufferView {
    #[default]
    Lit,
    Albedo,
    Normal,
    /// R = roughness, G = metallic, B = ambient occlusion.
    Material,
    /// Emitted radiance, exposed.
    Emissive,
    /// Cube shadow map visibility of the first three shadowed local lights, one per channel.
    LocalShadows,
}

impl GBufferView {
    pub const ALL: [Self; 6] =
        [Self::Lit, Self::Albedo, Self::Normal, Self::Material, Self::Emissive, Self::LocalShadows];

    pub fn label(self) -> &'static str {
        match self {
            Self::Lit => "Lit",
            Self::Albedo => "Albedo",
            Self::Normal => "Normal",
            Self::Material => "Roughness / metallic / AO",
            Self::Emissive => "Emissive",
            Self::LocalShadows => "Local light shadows",
        }
    }
}

pub struct RenderDebugFlags {
    pub albedo:   bool,
    pub indirect: bool,
    pub direct:   bool,
    pub shadows:  bool,
    pub specular: bool,
    /// Area lights generated from flat emissive meshes (`light_upload`).
    pub emissive_lights: bool,
    /// SSR and the probe fallback. Direct highlights stay.
    pub reflections: bool,
    /// Screen-space reflections only: with this off, reflections come from the probe fallback alone.
    pub ssr: bool,
    /// The camera-following reflection probe (`reflection_probe` pass): with this off, reflections
    /// SSR misses fall back to DDGI irradiance around the reflection vector.
    pub reflection_probe: bool,
    pub view:     GBufferView,

    pub show_bvh:        bool,
    pub bvh_leaves_only: bool,
    pub show_probes:     bool,
    pub show_grid:       bool,
    pub show_texture_aliasing: bool,

    pub ddgi_hysteresis:    f32,
    pub ddgi_normal_bias:   f32,
    pub ddgi_view_bias:     f32,
    pub ddgi_probe_spacing: f32,
    pub ddgi_grid_anchor:   [f32; 3],
}

impl Default for RenderDebugFlags {
    fn default() -> Self {
        Self {
            albedo:   true,
            indirect: true,
            direct:   true,
            shadows:  true,
            specular: true,
            emissive_lights: true,
            reflections: true,
            ssr: true,
            reflection_probe: true,
            view:     GBufferView::Lit,

            show_bvh:        false,
            bvh_leaves_only: false,
            show_probes:     false,
            show_grid:       false,
            show_texture_aliasing: false,

            ddgi_hysteresis:    0.97,
            ddgi_normal_bias:   0.2,
            ddgi_view_bias:     0.1,
            ddgi_probe_spacing: 5.0,
            ddgi_grid_anchor:      [0.0, 0.0, 0.0],
        }
    }
}

impl RenderDebugFlags {
    pub fn to_gpu(&self) -> DebugFlagsGpu {
        DebugFlagsGpu {
            disable_albedo:   !self.albedo as u32,
            disable_indirect: !self.indirect as u32,
            disable_direct:   !self.direct as u32,
            disable_shadows:  !self.shadows as u32,
            view:             self.view as u32,
            disable_specular: !self.specular as u32,
            disable_reflections: !self.reflections as u32,
            disable_ssr:      !self.ssr as u32,
        }
    }
}
