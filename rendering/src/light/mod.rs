pub mod emissive;
mod entity;
mod gpu;

pub use entity::{AreaLight, PointLight, SkyLight, SpotLight};
pub use gpu::{
    GpuAreaLight, GpuPointLight, GpuSkyLight, GpuSpotLight, LightCounts, MAX_SHADOWED_LIGHTS, SHADOW_KIND_AREA,
    SHADOW_KIND_POINT, SHADOW_KIND_SPOT, ShadowedLights,
};

pub const MAX_POINT_LIGHTS: usize = 256;
pub const MAX_SPOT_LIGHTS: usize = 64;
pub const MAX_AREA_LIGHTS: usize = 32;
pub const MAX_SKY_LIGHTS: usize = 4;
