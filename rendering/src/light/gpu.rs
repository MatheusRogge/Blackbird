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
    pub _pad0: f32,
    pub up_vs: [f32; 3],
    pub _pad1: f32,
    pub color: [f32; 3],
    pub _pad2: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightCounts {
    pub num_point: u32,
    pub num_spot: u32,
    pub num_area: u32,
    pub _pad: u32,
}
