use bytemuck::bytes_of;
use engine_core::world::World;

use crate::{
    bvh::{self, BvhInfo, BvhNode, GpuTriangle, MAX_BVH_NODES, MAX_BVH_TRIS},
    graph::{NodeId, RenderGraph},
    mesh::Mesh,
    pass::{Pass, PassContext, PassDesc},
    resource::{ResourceDescriptor, ResourceId},
};

pub struct BvhOutputs {
    pub nodes_id: ResourceId,
    pub tris_id: ResourceId,
    pub info_id: ResourceId,
}

pub struct BvhUploadPass {
    node_id: Option<NodeId>,
    nodes_id: ResourceId,
    tris_id: ResourceId,
    info_id: ResourceId,
    last_mesh_count: usize,
    dirty: bool,
}

impl BvhUploadPass {
    pub fn new(graph: &mut RenderGraph) -> (Self, BvhOutputs) {
        let nodes_id = graph.alloc_resource_id(ResourceDescriptor::Buffer {
            size: MAX_BVH_NODES * std::mem::size_of::<BvhNode>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let tris_id = graph.alloc_resource_id(ResourceDescriptor::Buffer {
            size: MAX_BVH_TRIS * std::mem::size_of::<GpuTriangle>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let info_id = graph.alloc_resource_id(ResourceDescriptor::Buffer {
            size: std::mem::size_of::<BvhInfo>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let outputs = BvhOutputs { nodes_id, tris_id, info_id };
        (
            Self {
                node_id: None,
                nodes_id,
                tris_id,
                info_id,
                last_mesh_count: 0,
                dirty: true,
            },
            outputs,
        )
    }
}

impl PassDesc for BvhUploadPass {
    fn name(&self) -> &'static str {
        "bvh_upload"
    }

    fn reads(&self) -> Vec<ResourceId> {
        vec![]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![self.nodes_id, self.tris_id, self.info_id]
    }
}

impl Pass for BvhUploadPass {
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
        let Some(&nodes_buf) = ctx.buffers.get(&self.nodes_id) else { return };
        let Some(&tris_buf) = ctx.buffers.get(&self.tris_id) else { return };
        let Some(&info_buf) = ctx.buffers.get(&self.info_id) else { return };

        let meshes = world.get_entities::<Mesh>();

        if meshes.len() != self.last_mesh_count {
            self.last_mesh_count = meshes.len();
            self.dirty = true;
        }

        if !self.dirty {
            return;
        }
        self.dirty = false;

        if meshes.is_empty() {
            queue.write_buffer(info_buf, 0, bytes_of(&BvhInfo { node_count: 0, tri_count: 0, enabled: 0, _pad: 0 }));
            return;
        }

        // Collect triangles from all meshes.
        let mut gpu_tris: Vec<GpuTriangle> = Vec::new();
        for mesh in &meshes {
            let verts = &mesh.vertices;
            for chunk in mesh.indices.chunks(3) {
                if chunk.len() < 3 { continue; }
                let v0 = verts[chunk[0] as usize].position;
                let v1 = verts[chunk[1] as usize].position;
                let v2 = verts[chunk[2] as usize].position;
                gpu_tris.push(GpuTriangle { v0, _pad0: 0, v1, _pad1: 0, v2, _pad2: 0 });
            }
        }

        let tri_count = gpu_tris.len() as u64;
        if tri_count > MAX_BVH_TRIS {
            log::warn!("bvh_upload: {} triangles exceeds limit ({MAX_BVH_TRIS}), skipping", tri_count);
            queue.write_buffer(info_buf, 0, bytes_of(&BvhInfo { node_count: 0, tri_count: 0, enabled: 0, _pad: 0 }));
            return;
        }

        log::info!("bvh_upload: building BVH for {} triangles", tri_count);
        let built = bvh::build(gpu_tris);
        let node_count = built.nodes.len() as u64;

        if node_count > MAX_BVH_NODES {
            log::warn!("bvh_upload: {} nodes exceeds limit ({MAX_BVH_NODES}), skipping", node_count);
            queue.write_buffer(info_buf, 0, bytes_of(&BvhInfo { node_count: 0, tri_count: 0, enabled: 0, _pad: 0 }));
            return;
        }

        queue.write_buffer(nodes_buf, 0, bytemuck::cast_slice(&built.nodes));
        queue.write_buffer(tris_buf, 0, bytemuck::cast_slice(&built.triangles));
        queue.write_buffer(
            info_buf,
            0,
            bytes_of(&BvhInfo {
                node_count: built.nodes.len() as u32,
                tri_count: built.triangles.len() as u32,
                enabled: 1,
                _pad: 0,
            }),
        );
        log::info!("bvh_upload: {} nodes uploaded", node_count);
    }
}
