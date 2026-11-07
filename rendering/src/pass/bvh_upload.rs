use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use bytemuck::bytes_of;
use engine_core::world::World;
use ultraviolet::{Mat4, Vec3};

use engine_core::entity::EntityKey;

use crate::{
    arena::BufferRange,
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

/// A mesh in the uploaded BVH, so a change to its emissive (e.g. an animated material) is
/// written into its triangles without a rebuild.
struct BuiltMesh {
    key: EntityKey,
    /// The packed emissive its triangles carry.
    emissive: u32,
    /// Its triangles' indices in the uploaded (BVH-ordered) triangle buffer.
    triangles: Vec<u32>,
}

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
    /// `TextureAsset::mean` per (`TextureAsset::id`, sRGB), pruned to live textures on each rebuild.
    texture_means: HashMap<(u64, bool), [f32; 4]>,
    built_meshes: Vec<BuiltMesh>,
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
                built_meshes: Vec::new(),
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

        let handles = world.get_entity_handles::<Mesh>();
        let meshes: Vec<&Mesh> = handles.iter().map(|h| h.entity).collect();

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
            self.refresh_emissive(queue, tris_buf, world);
            return;
        }
        self.dirty = false;
        self.settled_frames = 0;
        self.built_transforms = transforms;

        self.built_meshes.clear();
        if meshes.is_empty() {
            self.node_count.store(0, Ordering::Relaxed);
            info_buf.write(queue, bytes_of(&BvhInfo { node_count: 0, tri_count: 0, enabled: 0, _pad: 0 }));
            return;
        }

        // Collect triangles from all meshes. Probe rays can't sample textures, so each triangle
        // carries its material factors times the average of its textures.
        let mut gpu_tris: Vec<GpuTriangle> = Vec::new();
        // Index into `built_meshes` per triangle.
        let mut tri_mesh: Vec<u32> = Vec::new();
        // Blended surfaces would block GI rays as if they were solid.
        for handle in handles.iter().filter(|h| !h.entity.is_blended()) {
            let mesh = handle.entity;
            let mut mean = |texture: &Option<Arc<TextureAsset>>, srgb: bool| {
                texture.as_ref().map_or([1.0; 4], |t| {
                    *self.texture_means.entry((t.id(), srgb)).or_insert_with(|| t.mean(srgb))
                })
            };
            let albedo_mean = mean(&mesh.albedo_texture, true);
            let mr_mean = mean(&mesh.metallic_roughness_texture, false);
            let factors = &mesh.material;
            let albedo: [f32; 3] = std::array::from_fn(|c| factors.base_color[c] * albedo_mean[c]);
            let metallic = factors.metallic * mr_mean[2];
            let roughness = factors.roughness * mr_mean[1];
            let emissive = emissive_of(mesh, &mut self.texture_means);
            let mesh_index = self.built_meshes.len() as u32;
            self.built_meshes.push(BuiltMesh { key: handle.id, emissive, triangles: Vec::new() });

            let world_pos = |i: usize| -> [f32; 3] {
                mesh.transform.transform_point3(Vec3::from(mesh.vertices[i].position)).into()
            };
            // Cut-out texels pass `alpha × base color alpha >= cutoff`.
            let cutout = mesh.alpha_cutoff().map(|cutoff| (cutoff, mesh.material.base_color[3]));
            let verts = &mesh.vertices;
            for chunk in mesh.indices.chunks(3) {
                if chunk.len() < 3 { continue; }
                let (i0, i1, i2) = (chunk[0] as usize, chunk[1] as usize, chunk[2] as usize);
                if i0 >= verts.len() || i1 >= verts.len() || i2 >= verts.len() { continue; }
                let (v0, v1, v2) = (world_pos(i0), world_pos(i1), world_pos(i2));
                let color = |c: usize| (verts[i0].color[c] + verts[i1].color[c] + verts[i2].color[c]) / 3.0;
                let coverage = match cutout {
                    None => 1.0,
                    Some((_, base_alpha)) if base_alpha <= 0.0 => 0.0,
                    Some((cutoff, base_alpha)) => mesh.albedo_texture.as_ref().map_or(
                        if base_alpha >= cutoff { 1.0 } else { 0.0 },
                        |texture| texture.coverage([i0, i1, i2].map(|i| verts[i].uv), cutoff / base_alpha),
                    ),
                };
                if coverage <= 0.0 { continue; }
                let albedo_metallic = pack_unorm4x8([albedo[0] * color(0), albedo[1] * color(1), albedo[2] * color(2), metallic]);
                let roughness_coverage = pack_unorm4x8([roughness, coverage, 0.0, 0.0]);
                gpu_tris.push(GpuTriangle { v0, albedo_metallic, v1, roughness_coverage, v2, emissive });
                tri_mesh.push(mesh_index);
            }
        }

        let live: HashSet<u64> = meshes
            .iter()
            .flat_map(|m| [&m.albedo_texture, &m.metallic_roughness_texture, &m.emissive_texture])
            .filter_map(|t| t.as_ref().map(|t| t.id()))
            .collect();
        self.texture_means.retain(|(id, _), _| live.contains(id));

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

        for (index, &source) in built.order.iter().enumerate() {
            self.built_meshes[tri_mesh[source as usize] as usize].triangles.push(index as u32);
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

impl BvhUploadPass {
    /// Rewrites the emissive word of the triangles of each mesh whose emissive changed since the build.
    fn refresh_emissive(&mut self, queue: &wgpu::Queue, tris: BufferRange<'_>, world: &World) {
        const EMISSIVE_OFFSET: u64 = std::mem::offset_of!(GpuTriangle, emissive) as u64;
        for built in &mut self.built_meshes {
            let Some(mesh) = world.get_entity::<Mesh>(built.key) else { continue };
            let emissive = emissive_of(mesh, &mut self.texture_means);
            if emissive == built.emissive {
                continue;
            }
            built.emissive = emissive;
            for &index in &built.triangles {
                let offset = u64::from(index) * std::mem::size_of::<GpuTriangle>() as u64 + EMISSIVE_OFFSET;
                tris.write_at(queue, offset, bytes_of(&emissive));
            }
        }
    }
}

/// The mesh's packed emitted radiance: factor × strength × its emissive texture's mean.
fn emissive_of(mesh: &Mesh, texture_means: &mut HashMap<(u64, bool), [f32; 4]>) -> u32 {
    let mean = mesh
        .emissive_texture
        .as_ref()
        .map_or([1.0; 4], |t| *texture_means.entry((t.id(), true)).or_insert_with(|| t.mean(true)));
    let factors = &mesh.material;
    pack_emissive(std::array::from_fn(|c| factors.emissive[c] * factors.emissive_strength * mean[c]))
}
