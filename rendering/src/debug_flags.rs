use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DebugFlagsGpu {
    pub disable_albedo:   u32,
    pub disable_indirect: u32,
    pub disable_direct:   u32,
    pub disable_shadows:  u32,
}

pub struct RenderDebugFlags {
    pub albedo:   bool,
    pub indirect: bool,
    pub direct:   bool,
    pub shadows:  bool,

    pub show_bvh:        bool,
    pub bvh_leaves_only: bool,
    pub show_probes:     bool,
    pub show_grid:       bool,

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

            show_bvh:        false,
            bvh_leaves_only: false,
            show_probes:     false,
            show_grid:       false,

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
        }
    }
}
