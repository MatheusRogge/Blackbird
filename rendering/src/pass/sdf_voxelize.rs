use bytemuck::{Pod, Zeroable};
use engine_core::world::World;
use wgpu::util::DeviceExt;

use crate::{
    graph::{NodeId, RenderGraph},
    mesh::Mesh,
    pass::{Pass, PassContext, PassDesc},
    resource::{ResourceDescriptor, ResourceId},
    shader::ShaderAsset,
};

pub const SDF_RESOLUTION: u32 = 128;
const MAX_SDF_TRIANGLES: u32 = 8_000;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SdfParams {
    pub world_min: [f32; 3],
    pub voxel_size: f32,
    pub world_max: [f32; 3],
    pub triangle_count: u32,
    pub resolution: [u32; 3],
    pub _pad: u32,
}

pub struct SdfOutputs {
    pub sdf_volume_id: ResourceId,
    pub sdf_params_id: ResourceId,
}

pub struct SdfVoxelizePass {
    node_id: Option<NodeId>,
    sdf_volume_id: ResourceId,
    sdf_params_id: ResourceId,
    vertex_buf: Option<wgpu::Buffer>,
    index_buf: Option<wgpu::Buffer>,
    pipeline: Option<wgpu::ComputePipeline>,
    bind_group: Option<wgpu::BindGroup>,
    dirty: bool,
    last_mesh_count: usize,
    shader: ShaderAsset,
}

impl SdfVoxelizePass {
    pub fn new(graph: &mut RenderGraph, shader: ShaderAsset) -> (Self, SdfOutputs) {
        let sdf_volume_id = graph.alloc_resource_id(ResourceDescriptor::Fixed3DTexture {
            size: [SDF_RESOLUTION, SDF_RESOLUTION, SDF_RESOLUTION],
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        let sdf_params_id = graph.alloc_resource_id(ResourceDescriptor::Buffer {
            size: std::mem::size_of::<SdfParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let outputs = SdfOutputs { sdf_volume_id, sdf_params_id };

        (
            Self {
                node_id: None,
                sdf_volume_id,
                sdf_params_id,
                vertex_buf: None,
                index_buf: None,
                pipeline: None,
                bind_group: None,
                dirty: true,
                last_mesh_count: 0,
                shader,
            },
            outputs,
        )
    }
}

impl PassDesc for SdfVoxelizePass {
    fn name(&self) -> &'static str {
        "sdf_voxelize"
    }

    fn reads(&self) -> Vec<ResourceId> {
        vec![]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![self.sdf_volume_id, self.sdf_params_id]
    }
}

impl Pass for SdfVoxelizePass {
    fn bind_node_id(&mut self, node_id: NodeId) {
        self.node_id = Some(node_id);
    }

    fn execute(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &PassContext<'_>,
        world: &World,
    ) {
        let Some(&sdf_view) = ctx.views.get(&self.sdf_volume_id) else {
            return;
        };
        let Some(&params_buf) = ctx.buffers.get(&self.sdf_params_id) else {
            return;
        };

        let meshes = world.get_entities::<Mesh>();
        if meshes.is_empty() {
            return;
        }

        if meshes.len() != self.last_mesh_count {
            self.last_mesh_count = meshes.len();
            self.dirty = true;
        }

        if self.dirty {
            let positions: Vec<f32> = meshes
                .iter()
                .flat_map(|m| m.vertices.iter().flat_map(|v| v.position))
                .collect();

            let mut all_indices: Vec<u32> = Vec::new();
            let mut base_vertex: u32 = 0;
            for mesh in &meshes {
                for &idx in &mesh.indices {
                    all_indices.push(idx + base_vertex);
                }
                base_vertex += mesh.vertices.len() as u32;
            }

            let triangle_count = (all_indices.len() / 3) as u32;

            let mut world_min = [f32::MAX; 3];
            let mut world_max = [f32::MIN; 3];
            for chunk in positions.chunks(3) {
                for i in 0..3 {
                    world_min[i] = world_min[i].min(chunk[i]);
                    world_max[i] = world_max[i].max(chunk[i]);
                }
            }
            for i in 0..3 {
                let margin = (world_max[i] - world_min[i]) * 0.1 + 1.0;
                world_min[i] -= margin;
                world_max[i] += margin;
            }

            let max_extent = (0..3)
                .map(|i| world_max[i] - world_min[i])
                .fold(0.0f32, f32::max);
            let voxel_size = max_extent / SDF_RESOLUTION as f32;

            let sdf_params = SdfParams {
                world_min,
                voxel_size,
                world_max,
                triangle_count,
                resolution: [SDF_RESOLUTION; 3],
                _pad: 0,
            };
            queue.write_buffer(params_buf, 0, bytemuck::bytes_of(&sdf_params));

            self.vertex_buf = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&positions),
                usage: wgpu::BufferUsages::STORAGE,
            }));
            self.index_buf = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&all_indices),
                usage: wgpu::BufferUsages::STORAGE,
            }));

            self.bind_group = None;

            if triangle_count > MAX_SDF_TRIANGLES {
                log::warn!(
                    "sdf_voxelize: {} triangles exceeds limit ({}), skipping SDF — probes will use sky-only indirect",
                    triangle_count, MAX_SDF_TRIANGLES
                );
                // voxel_size=0 signals probe_trace to skip geometry occlusion
                queue.write_buffer(params_buf, 0, bytemuck::bytes_of(&SdfParams {
                    world_min: [0.0; 3],
                    voxel_size: 0.0,
                    world_max: [0.0; 3],
                    triangle_count: 0,
                    resolution: [SDF_RESOLUTION; 3],
                    _pad: 0,
                }));
                self.dirty = false;
                return;
            }

            log::info!(
                "sdf_voxelize: {} meshes, {} triangles",
                meshes.len(),
                triangle_count
            );
        }

        let (Some(vertex_buf), Some(index_buf)) = (&self.vertex_buf, &self.index_buf) else {
            return;
        };

        if self.pipeline.is_none() {
            let module = self
                .shader
                .compile(device)
                .expect("sdf_voxelize shader compile failed");

            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sdf_voxelize_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D3,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            let pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("sdf_voxelize_pipeline_layout"),
                    bind_group_layouts: &[Some(&bgl)],
                    immediate_size: 0,
                });

            self.pipeline = Some(device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("sdf_voxelize_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                },
            ));
        }

        if self.bind_group.is_none() {
            let layout = self.pipeline.as_ref().unwrap().get_bind_group_layout(0);
            self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(sdf_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: vertex_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: index_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            }));
        }

        if self.dirty {
            let pipeline = self.pipeline.as_ref().unwrap();
            let bg = self.bind_group.as_ref().unwrap();

            const GROUP_SIZE: u32 = 4;
            let wg = SDF_RESOLUTION / GROUP_SIZE;

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sdf_voxelize"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, bg, &[]);
            compute_pass.dispatch_workgroups(wg, wg, wg);

            self.dirty = false;
        }
    }
}
