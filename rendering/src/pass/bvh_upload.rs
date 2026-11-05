use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use bytemuck::bytes_of;
use engine_core::world::World;
use ultraviolet::{Mat4, Vec3};

use crate::{
    bvh::{self, BvhInfo, BvhNode, GpuTriangle, MAX_BVH_NODES, MAX_BVH_TRIS, pack_emissive},
    graph::{NodeId, RenderGraph},
    mesh::Mesh,
    pass::{Pass, PassContext, PassDesc},
    resource::{ResourceDescriptor, ResourceId},
    texture::TextureAsset,
};

/// Frames a transform change must stay unchanged before the BVH is rebuilt, so
/// dragging a mesh doesn't rebuild it every frame.
const SETTLE_FRAMES: u32 = 10;

pub struct BvhOutputs {
    pub nodes_id: ResourceId,
    pub tris_id: ResourceId,
    pub info_id: ResourceId,
    pub node_count: Arc<AtomicU32>,
}

pub struct BvhUploadPass {
    node_id: Option<NodeId>,
    nodes_id: ResourceId,
    tris_id: ResourceId,
    info_id: ResourceId,
    last_mesh_count: usize,
    dirty: bool,
    built_transforms: Vec<Mat4>,
    seen_transforms: Vec<Mat4>,
    settled_frames: u32,
    node_count: Arc<AtomicU32>,
    /// `TextureAsset::mean` per (`Arc` pointer, sRGB).
    texture_means: HashMap<(usize, bool), [f32; 4]>,
}

fn pack_unorm4x8(v: [f32; 4]) -> u32 {
    v.iter()
        .enumerate()
        .map(|(i, c)| ((c.clamp(0.0, 1.0) * 255.0).round() as u32) << (8 * i))
        .sum()
}

impl BvhUploadPass {
    pub fn new(graph: &mut RenderGraph) -> (Self, BvhOutputs) {
        let nodes_id = graph.alloc_resource_id(ResourceDescriptor::UploadBuffer {
            size: MAX_BVH_NODES * std::mem::size_of::<BvhNode>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let tris_id = graph.alloc_resource_id(ResourceDescriptor::UploadBuffer {
            size: MAX_BVH_TRIS * std::mem::size_of::<GpuTriangle>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let info_id = graph.alloc_resource_id(ResourceDescriptor::UploadBuffer {
            size: std::mem::size_of::<BvhInfo>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let node_count = Arc::new(AtomicU32::new(0));
        let outputs = BvhOutputs { nodes_id, tris_id, info_id, node_count: Arc::clone(&node_count) };
        (
            Self {
                node_id: None,
                nodes_id,
                tris_id,
                info_id,
                last_mesh_count: 0,
                dirty: true,
                built_transforms: Vec::new(),
                seen_transforms: Vec::new(),
                settled_frames: 0,
                node_count,
                texture_means: HashMap::new(),
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

        let transforms: Vec<Mat4> = meshes.iter().map(|m| m.transform).collect();
        if transforms != self.built_transforms {
            if transforms == self.seen_transforms {
                self.settled_frames += 1;
            } else {
                self.settled_frames = 0;
                self.seen_transforms.clone_from(&transforms);
            }
            if self.settled_frames >= SETTLE_FRAMES {
                self.dirty = true;
            }
        }

        if !self.dirty {
            return;
        }
        self.dirty = false;
        self.settled_frames = 0;
        self.built_transforms = transforms;

        if meshes.is_empty() {
            self.node_count.store(0, Ordering::Relaxed);
            info_buf.write(queue, bytes_of(&BvhInfo { node_count: 0, tri_count: 0, enabled: 0, _pad: 0 }));
            return;
        }

        // Collect triangles from all meshes. Probe rays can't sample textures, so each triangle
        // carries its material factors times the average of its textures.
        let mut gpu_tris: Vec<GpuTriangle> = Vec::new();
        for mesh in &meshes {
            let mut mean = |texture: &Option<Arc<TextureAsset>>, srgb: bool| {
                texture.as_ref().map_or([1.0; 4], |t| {
                    *self.texture_means.entry((Arc::as_ptr(t) as usize, srgb)).or_insert_with(|| t.mean(srgb))
                })
            };
            let albedo_mean = mean(&mesh.albedo_texture, true);
            let mr_mean = mean(&mesh.metallic_roughness_texture, false);
            let factors = &mesh.material;
            let albedo: [f32; 3] = std::array::from_fn(|c| factors.base_color[c] * albedo_mean[c]);
            let metallic = factors.metallic * mr_mean[2];
            let roughness = factors.roughness * mr_mean[1];
            let emissive_mean = mean(&mesh.emissive_texture, true);
            let emissive =
                pack_emissive(std::array::from_fn(|c| factors.emissive[c] * factors.emissive_strength * emissive_mean[c]));

            let world_pos = |i: usize| -> [f32; 3] {
                mesh.transform.transform_point3(Vec3::from(mesh.vertices[i].position)).into()
            };
            let verts = &mesh.vertices;
            for chunk in mesh.indices.chunks(3) {
                if chunk.len() < 3 { continue; }
                let (i0, i1, i2) = (chunk[0] as usize, chunk[1] as usize, chunk[2] as usize);
                if i0 >= verts.len() || i1 >= verts.len() || i2 >= verts.len() { continue; }
                let (v0, v1, v2) = (world_pos(i0), world_pos(i1), world_pos(i2));
                let color = |c: usize| (verts[i0].color[c] + verts[i1].color[c] + verts[i2].color[c]) / 3.0;
                let albedo_metallic = pack_unorm4x8([albedo[0] * color(0), albedo[1] * color(1), albedo[2] * color(2), metallic]);
                gpu_tris.push(GpuTriangle { v0, albedo_metallic, v1, roughness, v2, emissive });
            }
        }

        let tri_count = gpu_tris.len() as u64;
        if tri_count > MAX_BVH_TRIS {
            log::warn!("bvh_upload: {} triangles exceeds limit ({MAX_BVH_TRIS}), skipping", tri_count);
            info_buf.write(queue, bytes_of(&BvhInfo { node_count: 0, tri_count: 0, enabled: 0, _pad: 0 }));
            return;
        }

        log::info!("bvh_upload: building BVH for {} triangles", tri_count);
        let built = bvh::build(gpu_tris);
        let node_count = built.nodes.len() as u64;

        if node_count > MAX_BVH_NODES {
            log::warn!("bvh_upload: {} nodes exceeds limit ({MAX_BVH_NODES}), skipping", node_count);
            self.node_count.store(0, Ordering::Relaxed);
            info_buf.write(queue, bytes_of(&BvhInfo { node_count: 0, tri_count: 0, enabled: 0, _pad: 0 }));
            return;
        }

        let n = built.nodes.len() as u32;
        nodes_buf.write(queue, bytemuck::cast_slice(&built.nodes));
        tris_buf.write(queue, bytemuck::cast_slice(&built.triangles));
        info_buf.write(queue, bytes_of(&BvhInfo {
                node_count: n,
                tri_count: built.triangles.len() as u32,
                enabled: 1,
                _pad: 0,
            }),
        );
        self.node_count.store(n, Ordering::Relaxed);
        log::info!("bvh_upload: {} nodes uploaded", node_count);
    }
}
