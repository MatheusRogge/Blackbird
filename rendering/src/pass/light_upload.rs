use std::mem::size_of;

use engine_core::world::World;
use ultraviolet::Vec4;

use crate::{
    camera::Camera,
    graph::{NodeId, RenderGraph},
    light::{
        AreaLight, GpuAreaLight, GpuPointLight, GpuSkyLight, GpuSpotLight, LightCounts,
        MAX_AREA_LIGHTS, MAX_POINT_LIGHTS, MAX_SKY_LIGHTS, MAX_SPOT_LIGHTS, PointLight, SkyLight,
        SpotLight,
    },
    pass::{PassContext, Pass, PassDesc},
    resource::{ResourceDescriptor, ResourceId},
};

pub struct LightBuffers {
    pub point_buffer_id: ResourceId,
    pub spot_buffer_id: ResourceId,
    pub area_buffer_id: ResourceId,
    pub sky_buffer_id: ResourceId,
    pub counts_buffer_id: ResourceId,
}

pub struct LightUploadPass {
    node_id: Option<NodeId>,
    pub buffers: LightBuffers,
}

impl LightUploadPass {
    pub fn new(graph: &mut RenderGraph) -> (Self, LightBuffers) {
        let point_buffer_id = graph.alloc_resource_id(ResourceDescriptor::Buffer {
            size: (MAX_POINT_LIGHTS * size_of::<GpuPointLight>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let spot_buffer_id = graph.alloc_resource_id(ResourceDescriptor::Buffer {
            size: (MAX_SPOT_LIGHTS * size_of::<GpuSpotLight>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let area_buffer_id = graph.alloc_resource_id(ResourceDescriptor::Buffer {
            size: (MAX_AREA_LIGHTS * size_of::<GpuAreaLight>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let sky_buffer_id = graph.alloc_resource_id(ResourceDescriptor::Buffer {
            size: (MAX_SKY_LIGHTS * size_of::<GpuSkyLight>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let counts_buffer_id = graph.alloc_resource_id(ResourceDescriptor::Buffer {
            size: size_of::<LightCounts>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let buffers = LightBuffers {
            point_buffer_id,
            spot_buffer_id,
            area_buffer_id,
            sky_buffer_id,
            counts_buffer_id,
        };

        let pass = Self {
            node_id: None,
            buffers: LightBuffers {
                point_buffer_id,
                spot_buffer_id,
                area_buffer_id,
                sky_buffer_id,
                counts_buffer_id,
            },
        };

        (pass, buffers)
    }
}

impl PassDesc for LightUploadPass {
    fn name(&self) -> &'static str {
        "light_upload"
    }

    fn reads(&self) -> Vec<ResourceId> {
        vec![]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![
            self.buffers.point_buffer_id,
            self.buffers.spot_buffer_id,
            self.buffers.area_buffer_id,
            self.buffers.sky_buffer_id,
            self.buffers.counts_buffer_id,
        ]
    }
}

impl Pass for LightUploadPass {
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
        let cameras = world.get_entities::<Camera>();
        let Some(camera) = cameras.first() else {
            return;
        };
        let view = camera.view_matrix();

        let to_vs = |p: ultraviolet::Vec3| -> [f32; 3] {
            let v = view * Vec4::new(p.x, p.y, p.z, 1.0);
            [v.x, v.y, v.z]
        };

        let dir_to_vs = |d: ultraviolet::Vec3| -> [f32; 3] {
            let v = view * Vec4::new(d.x, d.y, d.z, 0.0);
            [v.x, v.y, v.z]
        };

        let point_lights: Vec<GpuPointLight> = world
            .get_entities::<PointLight>()
            .iter()
            .take(MAX_POINT_LIGHTS)
            .map(|l| GpuPointLight {
                position_vs: to_vs(l.position),
                radius: l.radius,
                color: l.color.into(),
                intensity: l.intensity,
            })
            .collect();

        let spot_lights: Vec<GpuSpotLight> = world
            .get_entities::<SpotLight>()
            .iter()
            .take(MAX_SPOT_LIGHTS)
            .map(|l| GpuSpotLight {
                position_vs: to_vs(l.position),
                radius: l.radius,
                direction_vs: dir_to_vs(l.direction),
                inner_cos: l.inner_angle.cos(),
                color: l.color.into(),
                outer_cos: l.outer_angle.cos(),
                intensity: l.intensity,
                _pad: [0.0; 3],
            })
            .collect();

        let area_lights: Vec<GpuAreaLight> = world
            .get_entities::<AreaLight>()
            .iter()
            .take(MAX_AREA_LIGHTS)
            .map(|l| GpuAreaLight {
                position_vs: to_vs(l.position),
                intensity: l.intensity,
                right_vs: dir_to_vs(l.right),
                _pad0: 0.0,
                up_vs: dir_to_vs(l.up),
                _pad1: 0.0,
                color: l.color.into(),
                _pad2: 0.0,
            })
            .collect();

        let sky_lights: Vec<GpuSkyLight> = world
            .get_entities::<SkyLight>()
            .iter()
            .take(MAX_SKY_LIGHTS)
            .map(|l| GpuSkyLight {
                direction_vs: dir_to_vs(l.direction.normalized()),
                intensity: l.intensity,
                color: l.color.into(),
                _pad: 0.0,
            })
            .collect();

        let counts = LightCounts {
            num_point: point_lights.len() as u32,
            num_spot: spot_lights.len() as u32,
            num_area: area_lights.len() as u32,
            num_sky: sky_lights.len() as u32,
        };

        if !point_lights.is_empty() {
            queue.write_buffer(
                ctx.buffers[&self.buffers.point_buffer_id],
                0,
                bytemuck::cast_slice(&point_lights),
            );
        }

        if !spot_lights.is_empty() {
            queue.write_buffer(
                ctx.buffers[&self.buffers.spot_buffer_id],
                0,
                bytemuck::cast_slice(&spot_lights),
            );
        }

        if !area_lights.is_empty() {
            queue.write_buffer(
                ctx.buffers[&self.buffers.area_buffer_id],
                0,
                bytemuck::cast_slice(&area_lights),
            );
        }

        if !sky_lights.is_empty() {
            queue.write_buffer(
                ctx.buffers[&self.buffers.sky_buffer_id],
                0,
                bytemuck::cast_slice(&sky_lights),
            );
        }

        queue.write_buffer(
            ctx.buffers[&self.buffers.counts_buffer_id],
            0,
            bytemuck::bytes_of(&counts),
        );
    }
}
