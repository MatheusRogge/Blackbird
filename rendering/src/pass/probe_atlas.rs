use std::sync::{Arc, RwLock};
use std::sync::atomic::AtomicUsize;

use bytemuck::{Pod, Zeroable};
use engine_core::world::World;

use crate::{
    bounds::Aabb,
    camera::Camera,
    mesh::Mesh,
    debug_flags::RenderDebugFlags,
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
            grid_origin: [-28.0, -2.0, -28.0],
            probe_spacing: 5.0,
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
    pub indirect_intensity: f32,         // offset 60
    /// `grid_origin` in probe cells from the anchor. Probes are stored in the atlas by world cell
    /// modulo `grid_dim`, so a probe keeps its slot while the grid slides past it.
    pub cell_origin: [i32; 3],           // offset 64
    pub bounce: f32,                     // offset 76
}

pub struct ProbeAtlasOutputs {
    /// Ping-pong pair; index 0 is current on startup.
    pub irradiance_atlas_ids: [ResourceId; 2],
    pub visibility_atlas_ids: [ResourceId; 2],
    pub probe_params_id: ResourceId,
    /// Which slot (0 or 1) was written most recently — read by LightingPass.
    pub current_atlas: Arc<AtomicUsize>,
}

pub struct ProbeAtlasPass {
    node_id: Option<NodeId>,
    config: ProbeGridConfig,
    probe_params_id: ResourceId,
    debug_flags: Arc<RwLock<RenderDebugFlags>>,
}

impl ProbeAtlasPass {
    pub fn new(
        graph: &mut RenderGraph,
        config: ProbeGridConfig,
        debug_flags: Arc<RwLock<RenderDebugFlags>>,
    ) -> (Self, ProbeAtlasOutputs) {
        let irradiance_atlas_ids = [
            graph.alloc_resource_id(ResourceDescriptor::FixedTexture {
                size: wgpu::Extent3d { width: IRRADIANCE_ATLAS_WIDTH, height: IRRADIANCE_ATLAS_HEIGHT, depth_or_array_layers: 1 },
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            }),
            graph.alloc_resource_id(ResourceDescriptor::FixedTexture {
                size: wgpu::Extent3d { width: IRRADIANCE_ATLAS_WIDTH, height: IRRADIANCE_ATLAS_HEIGHT, depth_or_array_layers: 1 },
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            }),
        ];
        let visibility_atlas_ids = [
            graph.alloc_resource_id(ResourceDescriptor::FixedTexture {
                size: wgpu::Extent3d { width: VISIBILITY_ATLAS_WIDTH, height: VISIBILITY_ATLAS_HEIGHT, depth_or_array_layers: 1 },
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            }),
            graph.alloc_resource_id(ResourceDescriptor::FixedTexture {
                size: wgpu::Extent3d { width: VISIBILITY_ATLAS_WIDTH, height: VISIBILITY_ATLAS_HEIGHT, depth_or_array_layers: 1 },
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            }),
        ];

        let probe_params_id = graph.alloc_resource_id(ResourceDescriptor::UploadBuffer {
            size: std::mem::size_of::<ProbeGridParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let current_atlas = Arc::new(AtomicUsize::new(0));
        let outputs = ProbeAtlasOutputs {
            irradiance_atlas_ids,
            visibility_atlas_ids,
            probe_params_id,
            current_atlas,
        };

        (
            Self {
                node_id: None,
                config,
                probe_params_id,
                debug_flags,
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
        world: &World,
    ) {
        let Some(&params_buf) = ctx.buffers.get(&self.probe_params_id) else {
            return;
        };

        let cfg = &self.config;
        let probe_count = cfg.grid_dim[0] * cfg.grid_dim[1] * cfg.grid_dim[2];

        let flags = self.debug_flags.read().expect("debug_flags poisoned");
        let (hysteresis, normal_bias, view_bias, probe_spacing, anchor) = (
            flags.ddgi_hysteresis,
            flags.ddgi_normal_bias,
            flags.ddgi_view_bias,
            flags.ddgi_probe_spacing,
            flags.ddgi_grid_anchor,
        );
        let (indirect_intensity, bounce) = (flags.ddgi_indirect_intensity, flags.ddgi_bounce);
        drop(flags);

        let max_ray_distance = {
            let dim = self.config.grid_dim;
            dim[0].max(dim[1]).max(dim[2]) as f32 * probe_spacing
        };

        // Probes sit at world-aligned multiples of probe_spacing (offset from anchor). The grid
        // centres on the camera, kept inside the scene's bounds: a scene that fits gets one fixed
        // grid covering all of it, a larger one a grid that slides only as far as its edges.
        // Following the camera alone left far parts of a fitting scene outside the grid, without
        // indirect light, until the camera walked towards them.
        let cell_origin: [i32; 3] = {
            let cameras = world.get_entities::<Camera>();
            let eye = cameras.first().map(|c| c.eye).unwrap_or_default();
            let mut scene = Aabb::empty();
            for handle in world.get_entity_handles::<Mesh>() {
                if let Some(draw) = ctx.geometry.get(handle.id) {
                    let bounds = draw.bounds.transformed(handle.entity.transform);
                    if !bounds.is_empty() {
                        scene.extend(bounds.min);
                        scene.extend(bounds.max);
                    }
                }
            }
            let dim = self.config.grid_dim;
            std::array::from_fn(|i| {
                let steps = dim[i] as i32 - 1;
                let centred = ((eye[i] - anchor[i]) / probe_spacing - steps as f32 * 0.5).floor() as i32;
                if scene.is_empty() {
                    return centred;
                }
                let first = ((scene.min[i] - anchor[i]) / probe_spacing).floor() as i32;
                let last = ((scene.max[i] - anchor[i]) / probe_spacing).ceil() as i32 - steps;
                if last <= first {
                    (first + last).div_euclid(2)
                } else {
                    centred.clamp(first, last)
                }
            })
        };
        let grid_origin: [f32; 3] = std::array::from_fn(|i| anchor[i] + cell_origin[i] as f32 * probe_spacing);

        let params = ProbeGridParams {
            grid_dim: cfg.grid_dim,
            probe_count,
            grid_origin,
            max_ray_distance,
            probe_spacing,
            rays_per_probe: 128,
            irradiance_texel_size: IRRADIANCE_TEXEL_SIZE,
            visibility_texel_size: VISIBILITY_TEXEL_SIZE,
            hysteresis,
            normal_bias,
            view_bias,
            indirect_intensity,
            cell_origin,
            bounce,
        };

        params_buf.write(queue, bytemuck::bytes_of(&params));
    }
}
