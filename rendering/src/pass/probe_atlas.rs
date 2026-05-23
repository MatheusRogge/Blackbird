use bytemuck::{Pod, Zeroable};
use engine_core::world::World;

use crate::{
    graph::{NodeId, RenderGraph},
    pass::{Pass, PassContext, PassDesc},
    resource::{ResourceDescriptor, ResourceId},
};

pub const PROBE_GRID_DIM: [u32; 3] = [12, 6, 12];
pub const PROBE_COUNT: u32 = 12 * 6 * 12; // 864
pub const RAYS_PER_PROBE: u32 = 128;
pub const IRRADIANCE_TEXEL_SIZE: u32 = 8;
pub const VISIBILITY_TEXEL_SIZE: u32 = 16;

const PROBES_PER_ROW: u32 = 30;
const PROBES_PER_COL: u32 = 29; // ceil(864 / 30) = 29, 30×29 = 870 ≥ 864

pub const IRRADIANCE_ATLAS_WIDTH: u32 = PROBES_PER_ROW * (IRRADIANCE_TEXEL_SIZE + 2);
pub const IRRADIANCE_ATLAS_HEIGHT: u32 = PROBES_PER_COL * (IRRADIANCE_TEXEL_SIZE + 2);
pub const VISIBILITY_ATLAS_WIDTH: u32 = PROBES_PER_ROW * (VISIBILITY_TEXEL_SIZE + 2);
pub const VISIBILITY_ATLAS_HEIGHT: u32 = PROBES_PER_COL * (VISIBILITY_TEXEL_SIZE + 2);

pub struct ProbeGridConfig {
    pub grid_dim: [u32; 3],
    pub grid_origin: [f32; 3],
    pub probe_spacing: f32,
}

impl Default for ProbeGridConfig {
    fn default() -> Self {
        Self {
            grid_dim: PROBE_GRID_DIM,
            // Covers a 24m×12m×24m volume centered near the world origin.
            grid_origin: [-12.0, 0.0, -12.0],
            probe_spacing: 2.0,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ProbeGridParams {
    pub grid_dim: [u32; 3],              // offset  0
    pub probe_count: u32,                // offset 12
    pub grid_origin: [f32; 3],           // offset 16
    pub max_ray_distance: f32,           // offset 28
    pub probe_spacing: f32,              // offset 32
    pub rays_per_probe: u32,             // offset 36
    pub irradiance_texel_size: u32,      // offset 40
    pub visibility_texel_size: u32,      // offset 44
    pub hysteresis: f32,                 // offset 48
    pub normal_bias: f32,                // offset 52
    pub view_bias: f32,                  // offset 56
    pub _pad: f32,                       // offset 60
}

pub struct ProbeAtlasOutputs {
    pub irradiance_atlas_id: ResourceId,
    pub visibility_atlas_id: ResourceId,
    pub probe_params_id: ResourceId,
}

pub struct ProbeAtlasPass {
    node_id: Option<NodeId>,
    config: ProbeGridConfig,
    probe_params_id: ResourceId,
}

impl ProbeAtlasPass {
    pub fn new(graph: &mut RenderGraph, config: ProbeGridConfig) -> (Self, ProbeAtlasOutputs) {
        let irradiance_atlas_id = graph.alloc_resource_id(ResourceDescriptor::FixedTexture {
            size: wgpu::Extent3d {
                width: IRRADIANCE_ATLAS_WIDTH,
                height: IRRADIANCE_ATLAS_HEIGHT,
                depth_or_array_layers: 1,
            },
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        });

        let visibility_atlas_id = graph.alloc_resource_id(ResourceDescriptor::FixedTexture {
            size: wgpu::Extent3d {
                width: VISIBILITY_ATLAS_WIDTH,
                height: VISIBILITY_ATLAS_HEIGHT,
                depth_or_array_layers: 1,
            },
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        });

        let probe_params_id = graph.alloc_resource_id(ResourceDescriptor::Buffer {
            size: std::mem::size_of::<ProbeGridParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let outputs = ProbeAtlasOutputs {
            irradiance_atlas_id,
            visibility_atlas_id,
            probe_params_id,
        };

        (
            Self {
                node_id: None,
                config,
                probe_params_id,
            },
            outputs,
        )
    }
}

impl PassDesc for ProbeAtlasPass {
    fn name(&self) -> &'static str {
        "probe_atlas"
    }

    fn reads(&self) -> Vec<ResourceId> {
        vec![]
    }

    // Only claims the params buffer; the atlas textures are allocated but
    // owned/written by ProbeUpdatePass (Phase 4).
    fn writes(&self) -> Vec<ResourceId> {
        vec![self.probe_params_id]
    }
}

impl Pass for ProbeAtlasPass {
    fn bind_node_id(&mut self, node_id: NodeId) {
        self.node_id = Some(node_id);
    }

    fn execute(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _encoder: &mut wgpu::CommandEncoder,
        ctx: &PassContext<'_>,
        _world: &World,
    ) {
        let Some(&params_buf) = ctx.buffers.get(&self.probe_params_id) else {
            return;
        };

        let cfg = &self.config;
        let probe_count = cfg.grid_dim[0] * cfg.grid_dim[1] * cfg.grid_dim[2];

        let params = ProbeGridParams {
            grid_dim: cfg.grid_dim,
            probe_count,
            grid_origin: cfg.grid_origin,
            max_ray_distance: 20.0,
            probe_spacing: cfg.probe_spacing,
            rays_per_probe: 128,
            irradiance_texel_size: IRRADIANCE_TEXEL_SIZE,
            visibility_texel_size: VISIBILITY_TEXEL_SIZE,
            hysteresis: 0.97,
            normal_bias: 0.2,
            view_bias: 0.1,
            _pad: 0.0,
        };

        queue.write_buffer(params_buf, 0, bytemuck::bytes_of(&params));
    }
}
