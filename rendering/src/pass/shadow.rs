
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use bytemuck::{Pod, Zeroable};
use engine_core::world::{EntityHandle, World};
use ultraviolet::{Mat4, Vec3, Vec4};

use crate::{
    arena::{GeometryArena, MeshDraw},
    bounds::Aabb,
    camera::Camera,
    graph::{NodeId, RenderGraph},
    light::{
        MAX_SHADOWED_LIGHTS, ShadowCaster, SharedShadowCasters, SkyLight,
        shadow_atlas::{ATLAS_SIZE, Tile},
    },
    mesh::Mesh,
    pass::{
        Pass, PassContext, PassDesc,
        material::albedo_alpha_cutoff,
        mipmap::{MipGenerator, TextureKind},
    },
    resource::{ResourceDescriptor, ResourceId},
    shader::ShaderAsset,
    texture::TextureAsset,
};

pub const NUM_CASCADES: usize = 4;
pub const SHADOW_MAP_SIZE: u32 = 2048;

const CASCADE_LAMBDA: f32 = 0.75;
const CASCADE_VP_STRIDE: u64 = 256;
const VP_SIZE: u64 = std::mem::size_of::<ShadowView>() as u64;

/// One shadow view, at `CASCADE_VP_STRIDE` in the cascade / cube face uniform buffers. Mirrors
/// `LightUniform` in `shadow.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ShadowView {
    view_proj: [[f32; 4]; 4],
    /// w = 0: xyz is the direction light travels (cascades); w = 1: xyz is the light's origin
    /// (cube faces).
    light: [f32; 4],
}

/// Face slots in `LocalShadowParams`: six per shadowed light (spot lights use the first).
pub const LOCAL_SHADOW_FACES: usize = MAX_SHADOWED_LIGHTS * 6;
const LOCAL_SHADOW_NEAR: f32 = 0.02;

/// Glass tint maps (`fs_glass` in `shadow.wgsl`), one layer per cascade / cube face: rgb is the
/// transmittance of every blended surface along the light ray, a the depth of the one nearest the
/// light. 16-bit float so that depth keeps its precision. Glass shadows are soft, so they use a
/// lower resolution than the depth maps.
const GLASS_TINT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const GLASS_TINT_SIZE: u32 = 1024;
/// Clear value: no tint, and a depth beyond any receiver.
const GLASS_TINT_CLEAR: wgpu::Color = wgpu::Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0e4 };
const GLASS_TINT_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Dst,
        dst_factor: wgpu::BlendFactor::Zero,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Min,
    },
};
/// Cube faces in the order `lighting.wgsl` picks them by dominant axis: +X, −X, +Y, −Y, +Z, −Z.
const CUBE_FACES: [(Vec3, Vec3); 6] = [
    (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0)),
    (Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0)),
    (Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
    (Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 0.0, -1.0)),
    (Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, -1.0, 0.0)),
    (Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, -1.0, 0.0)),
];

/// One local light face: its view-projection and where its tile sits in the atlas.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LocalShadowFace {
    pub view_proj: [[f32; 4]; 4],
    /// `Tile::rect`: atlas UV offset (xy) and scale (z), tile size in texels (w).
    pub rect: [f32; 4],
}

/// Every local light face, slot-major (`slot * 6 + face`). Mirrors `LocalShadowParams` in
/// `lighting.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LocalShadowParams {
    pub faces: [LocalShadowFace; LOCAL_SHADOW_FACES],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ShadowParams {
    pub view_to_shadow: [[[f32; 4]; 4]; NUM_CASCADES],
    pub cascade_splits: [f32; NUM_CASCADES],
    pub bias: f32,
    pub inv_shadow_map_size: f32,
    pub _pad: [f32; 2],
}

pub struct ShadowOutputs {
    pub shadow_map_id: ResourceId,
    pub shadow_params_id: ResourceId,
    /// Depth32Float `ATLAS_SIZE`² atlas of local light faces (`light::shadow_atlas`).
    pub local_shadow_map_id: ResourceId,
    /// Uniform `LocalShadowParams`.
    pub local_shadow_params_id: ResourceId,
    /// `GLASS_TINT_FORMAT`, `NUM_CASCADES` layers of `GLASS_TINT_SIZE`².
    pub glass_tint_id: ResourceId,
    /// `GLASS_TINT_FORMAT` `ATLAS_SIZE`² atlas, laid out like the depth atlas.
    pub local_glass_tint_id: ResourceId,
}

type AlbedoCache = HashMap<u64, (wgpu::Texture, wgpu::BindGroup)>;

/// Uploads mesh albedo textures (with mips) into bind groups for `shadow.wgsl`'s group 2.
struct AlbedoUpload {
    mips: MipGenerator,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl AlbedoUpload {
    /// Uploads the albedo of every mesh in `meshes` that `cache` doesn't hold yet (plus the white
    /// fallback at key 0) as `kind(mesh)`, and drops the ones no mesh uses any more.
    fn sync<'m>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        cache: &mut AlbedoCache,
        meshes: impl Iterator<Item = &'m Mesh>,
        kind: impl Fn(&Mesh) -> TextureKind,
    ) {
        let mut upload = |asset: &TextureAsset, kind| {
            let (texture, view) = self.mips.upload(device, queue, encoder, asset, kind);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shadow_albedo_texture"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            });
            (texture, bind_group)
        };
        cache.entry(0).or_insert_with(|| {
            let white = TextureAsset::new(1, 1, vec![255; 4]);
            upload(&white, TextureKind::Linear { alpha_cutoff: None })
        });
        let mut live = HashSet::from([0]);
        for mesh in meshes {
            let Some(texture) = &mesh.albedo_texture else { continue };
            live.insert(texture.id());
            cache.entry(texture.id()).or_insert_with(|| upload(texture, kind(mesh)));
        }
        cache.retain(|key, _| live.contains(key));
    }
}

/// Draws `glass` into one layer of a glass tint map, clearing it first.
#[allow(clippy::too_many_arguments)]
fn draw_glass(
    encoder: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    (vp_bg, vp_offset): (&wgpu::BindGroup, u32),
    instance_bg: &wgpu::BindGroup,
    textures: &AlbedoCache,
    geometry: &GeometryArena,
    glass: &[EntityHandle<'_, Mesh>],
    view_proj: Mat4,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("shadow_glass_tint"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(GLASS_TINT_CLEAR), store: wgpu::StoreOp::Store },
        })],
        ..Default::default()
    });
    if glass.is_empty() {
        return;
    }
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, vp_bg, &[vp_offset]);
    pass.set_bind_group(1, instance_bg, &[]);
    pass.set_vertex_buffer(0, geometry.vertex_buffer().slice(..));
    pass.set_index_buffer(geometry.index_buffer().slice(..), wgpu::IndexFormat::Uint32);
    for (handle, draw) in visible(glass.iter(), geometry, view_proj) {
        let Some((_, texture_bg)) = textures.get(&albedo_key(handle.entity)).or(textures.get(&0)) else {
            continue;
        };
        pass.set_bind_group(2, texture_bg, &[]);
        pass.draw_indexed(draw.indices.clone(), draw.base_vertex, draw.instances());
    }
}

pub struct ShadowPass {
    node_id: Option<NodeId>,
    shader: ShaderAsset,
    pipeline: Option<wgpu::RenderPipeline>,
    /// Unculled pipeline that discards cut-out texels, for `Mesh::is_masked` meshes.
    masked_pipeline: Option<wgpu::RenderPipeline>,
    albedo: Option<AlbedoUpload>,
    /// Albedo textures of masked meshes (alpha only), keyed by `TextureAsset::id`; key 0 is white.
    masked_textures: AlbedoCache,
    /// Tints blended meshes into the glass tint maps, unculled, multiplying (see `GLASS_TINT_BLEND`).
    glass_pipeline: Option<wgpu::RenderPipeline>,
    /// Albedo textures of blended meshes (sRGB color), keyed like `masked_textures`.
    glass_textures: AlbedoCache,
    bgl: Option<wgpu::BindGroupLayout>,
    instance_layout: Option<wgpu::BindGroupLayout>,
    instance_bind_group: Option<(u64, wgpu::BindGroup)>,

    shadow_map_id: ResourceId,
    shadow_params_id: ResourceId,
    cascade_vp_id: ResourceId,

    cascade_views: Vec<wgpu::TextureView>,
    glass_tint_id: ResourceId,
    glass_tint_views: Vec<wgpu::TextureView>,
    cascade_bind_group: Option<wgpu::BindGroup>,

    /// Read only to run after `light_upload`, which fills `casters` this frame.
    shadowed_buffer_id: ResourceId,
    casters: SharedShadowCasters,
    local_map_id: ResourceId,
    local_params_id: ResourceId,
    local_vp_id: ResourceId,
    local_glass_tint_id: ResourceId,
    local_bind_group: Option<wgpu::BindGroup>,
    /// Unculled, depth-biased variants of the two pipelines: cube faces see both sides of walls.
    local_pipeline: Option<wgpu::RenderPipeline>,
    local_masked_pipeline: Option<wgpu::RenderPipeline>,
    /// Clear one atlas tile (`vs_clear`): far depth, and the glass tint clear value.
    clear_depth_pipeline: Option<wgpu::RenderPipeline>,
    clear_tint_pipeline: Option<wgpu::RenderPipeline>,
    /// What each slot's faces were last rendered for, with the scene geometry signature then.
    rendered: [Option<(ShadowCaster, u64)>; MAX_SHADOWED_LIGHTS],
    local_params: LocalShadowParams,
}

/// Key of a mesh's albedo texture in `ShadowPass::masked_textures`: its `TextureAsset::id`, 0 for none.
fn albedo_key(mesh: &Mesh) -> u64 {
    mesh.albedo_texture.as_ref().map_or(0, |t| t.id())
}

impl ShadowPass {
    pub fn new(
        graph: &mut RenderGraph,
        shader: ShaderAsset,
        shadowed_buffer_id: ResourceId,
        casters: SharedShadowCasters,
    ) -> (Self, ShadowOutputs) {
        let shadow_map_id = graph.alloc_resource_id(ResourceDescriptor::FixedTexture {
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: NUM_CASCADES as u32,
            },
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        let shadow_params_id = graph.alloc_resource_id(ResourceDescriptor::UploadBuffer {
            size: std::mem::size_of::<ShadowParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let cascade_vp_id = graph.alloc_resource_id(ResourceDescriptor::UploadBuffer {
            size: CASCADE_VP_STRIDE * NUM_CASCADES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let local_map_id = graph.alloc_resource_id(ResourceDescriptor::FixedTexture {
            size: wgpu::Extent3d { width: ATLAS_SIZE, height: ATLAS_SIZE, depth_or_array_layers: 1 },
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        let local_params_id = graph.alloc_resource_id(ResourceDescriptor::UploadBuffer {
            size: std::mem::size_of::<LocalShadowParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let local_vp_id = graph.alloc_resource_id(ResourceDescriptor::UploadBuffer {
            size: CASCADE_VP_STRIDE * LOCAL_SHADOW_FACES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let tint_map = |size, layers| ResourceDescriptor::FixedTexture {
            size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: layers },
            format: GLASS_TINT_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        };
        let glass_tint_id = graph.alloc_resource_id(tint_map(GLASS_TINT_SIZE, NUM_CASCADES as u32));
        let local_glass_tint_id = graph.alloc_resource_id(tint_map(ATLAS_SIZE, 1));

        let outputs = ShadowOutputs {
            shadow_map_id,
            shadow_params_id,
            local_shadow_map_id: local_map_id,
            local_shadow_params_id: local_params_id,
            glass_tint_id,
            local_glass_tint_id,
        };

        (
            Self {
                node_id: None,
                shader,
                pipeline: None,
                masked_pipeline: None,
                albedo: None,
                masked_textures: HashMap::new(),
                glass_pipeline: None,
                glass_textures: HashMap::new(),
                bgl: None,
                instance_layout: None,
                instance_bind_group: None,
                shadow_map_id,
                shadow_params_id,
                cascade_vp_id,
                cascade_views: Vec::new(),
                glass_tint_id,
                glass_tint_views: Vec::new(),
                cascade_bind_group: None,
                shadowed_buffer_id,
                casters,
                local_map_id,
                local_params_id,
                local_vp_id,
                local_glass_tint_id,
                local_bind_group: None,
                local_pipeline: None,
                local_masked_pipeline: None,
                clear_depth_pipeline: None,
                clear_tint_pipeline: None,
                rendered: [None; MAX_SHADOWED_LIGHTS],
                local_params: LocalShadowParams::zeroed(),
            },
            outputs,
        )
    }
}

impl ShadowPass {
    /// Renders the faces of every shadowed light whose caster (light, shape, tiles) or the scene
    /// geometry changed since its last render into their atlas tiles, and uploads all faces.
    fn render_local_shadows(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        ctx: &PassContext<'_>,
        meshes: &[EntityHandle<'_, Mesh>],
        masked: &[&EntityHandle<'_, Mesh>],
        glass: &[EntityHandle<'_, Mesh>],
    ) {
        let (Some(&vp_buf), Some(&params_buf)) = (ctx.buffers.get(&self.local_vp_id), ctx.buffers.get(&self.local_params_id))
        else {
            return;
        };
        let (Some(&depth_view), Some(&tint_view)) = (ctx.views.get(&self.local_map_id), ctx.views.get(&self.local_glass_tint_id))
        else {
            return;
        };
        let (
            Some(pipeline),
            Some(masked_pipeline),
            Some(glass_pipeline),
            Some(clear_depth),
            Some(clear_tint),
            Some(bind_group),
            Some((_, instance_bg)),
        ) = (
            &self.local_pipeline,
            &self.local_masked_pipeline,
            &self.glass_pipeline,
            &self.clear_depth_pipeline,
            &self.clear_tint_pipeline,
            &self.local_bind_group,
            &self.instance_bind_group,
        )
        else {
            return;
        };

        let mut hasher = std::hash::DefaultHasher::new();
        ctx.geometry.generation().hash(&mut hasher);
        for handle in meshes.iter().chain(glass) {
            handle.id.hash(&mut hasher);
            for value in handle.entity.transform.as_array() {
                value.to_bits().hash(&mut hasher);
            }
        }
        let geometry = hasher.finish();

        // (face index, tile, view-projection) of every face to re-render.
        let mut faces: Vec<(usize, Tile, Mat4)> = Vec::new();
        let casters = *self.casters.lock().expect("shadow_casters poisoned");
        for (slot, caster) in casters.iter().enumerate() {
            let Some(caster) = *caster else {
                self.rendered[slot] = None;
                continue;
            };
            if self.rendered[slot] == Some((caster, geometry)) {
                continue;
            }
            self.rendered[slot] = Some((caster, geometry));
            let far = caster.range.max(LOCAL_SHADOW_NEAR * 2.0);
            for (face, &tile) in caster.tiles[..caster.faces()].iter().enumerate() {
                let vp = match caster.spot {
                    Some(spot) => {
                        let fov = 2.0 * spot.tan_half_fov.atan();
                        let up = if spot.direction.y.abs() > 0.99 { Vec3::unit_x() } else { Vec3::unit_y() };
                        ultraviolet::projection::perspective_wgpu_dx(fov, 1.0, LOCAL_SHADOW_NEAR, far)
                            * Mat4::look_at(caster.origin, caster.origin + spot.direction, up)
                    }
                    None => {
                        let (dir, up) = CUBE_FACES[face];
                        ultraviolet::projection::perspective_wgpu_dx(std::f32::consts::FRAC_PI_2, 1.0, LOCAL_SHADOW_NEAR, far)
                            * Mat4::look_at(caster.origin, caster.origin + dir, up)
                    }
                };
                let index = slot * 6 + face;
                let cols = vp.cols;
                let view_proj: [[f32; 4]; 4] = [cols[0].into(), cols[1].into(), cols[2].into(), cols[3].into()];
                self.local_params.faces[index] = LocalShadowFace { view_proj, rect: tile.rect() };
                let view = ShadowView { view_proj, light: [caster.origin.x, caster.origin.y, caster.origin.z, 1.0] };
                vp_buf.write_at(queue, CASCADE_VP_STRIDE * index as u64, bytemuck::bytes_of(&view));
                faces.push((index, tile, vp));
            }
        }
        params_buf.write(queue, bytemuck::bytes_of(&self.local_params));
        if faces.is_empty() {
            return;
        }

        let viewport = |pass: &mut wgpu::RenderPass<'_>, tile: Tile| {
            pass.set_viewport(tile.x as f32, tile.y as f32, tile.size as f32, tile.size as f32, 0.0, 1.0);
            pass.set_scissor_rect(tile.x, tile.y, tile.size, tile.size);
        };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shadow_local_faces"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        for &(index, tile, vp) in &faces {
            let offset = (CASCADE_VP_STRIDE * index as u64) as u32;
            viewport(&mut pass, tile);
            pass.set_pipeline(clear_depth);
            pass.draw(0..3, 0..1);

            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[offset]);
            pass.set_bind_group(1, instance_bg, &[]);
            pass.set_vertex_buffer(0, ctx.geometry.vertex_buffer().slice(..));
            pass.set_index_buffer(ctx.geometry.index_buffer().slice(..), wgpu::IndexFormat::Uint32);
            for (_, draw) in visible(meshes.iter().filter(|h| !h.entity.is_masked()), ctx.geometry, vp) {
                pass.draw_indexed(draw.indices.clone(), draw.base_vertex, draw.instances());
            }

            pass.set_pipeline(masked_pipeline);
            pass.set_bind_group(0, bind_group, &[offset]);
            pass.set_bind_group(1, instance_bg, &[]);
            let mut current_texture = None;
            for (handle, draw) in visible(masked.iter().copied(), ctx.geometry, vp) {
                let key = albedo_key(handle.entity);
                if current_texture != Some(key) {
                    let Some((_, texture_bg)) = self.masked_textures.get(&key).or(self.masked_textures.get(&0)) else {
                        continue;
                    };
                    pass.set_bind_group(2, texture_bg, &[]);
                    current_texture = Some(key);
                }
                pass.draw_indexed(draw.indices.clone(), draw.base_vertex, draw.instances());
            }
        }
        drop(pass);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shadow_local_glass_tint"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: tint_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        for &(index, tile, vp) in &faces {
            viewport(&mut pass, tile);
            pass.set_pipeline(clear_tint);
            pass.draw(0..3, 0..1);
            if glass.is_empty() {
                continue;
            }
            pass.set_pipeline(glass_pipeline);
            pass.set_bind_group(0, bind_group, &[(CASCADE_VP_STRIDE * index as u64) as u32]);
            pass.set_bind_group(1, instance_bg, &[]);
            pass.set_vertex_buffer(0, ctx.geometry.vertex_buffer().slice(..));
            pass.set_index_buffer(ctx.geometry.index_buffer().slice(..), wgpu::IndexFormat::Uint32);
            for (handle, draw) in visible(glass.iter(), ctx.geometry, vp) {
                let Some((_, texture_bg)) = self.glass_textures.get(&albedo_key(handle.entity)).or(self.glass_textures.get(&0))
                else {
                    continue;
                };
                pass.set_bind_group(2, texture_bg, &[]);
                pass.draw_indexed(draw.indices.clone(), draw.base_vertex, draw.instances());
            }
        }
    }
}

/// The meshes with geometry whose bounds reach into the clip volume of `view_proj`.
fn visible<'m, 'w: 'm, 'g>(
    meshes: impl Iterator<Item = &'m EntityHandle<'w, Mesh>>,
    geometry: &'g GeometryArena,
    view_proj: Mat4,
) -> impl Iterator<Item = (&'m EntityHandle<'w, Mesh>, &'g MeshDraw)> {
    meshes.filter_map(move |h| {
        let draw = geometry.get(h.id)?;
        (!draw.bounds.outside_clip(view_proj * h.entity.transform)).then_some((h, draw))
    })
}

fn ortho_rh_wgpu(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> Mat4 {
    let rml = r - l;
    let tmb = t - b;
    let fmn = f - n;
    Mat4::new(
        Vec4::new(2.0 / rml, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 2.0 / tmb, 0.0, 0.0),
        Vec4::new(0.0, 0.0, -1.0 / fmn, 0.0),
        Vec4::new(-(r + l) / rml, -(t + b) / tmb, -n / fmn, 1.0),
    )
}

/// Builds a stable light-space VP matrix for one cascade slice.
///
/// Uses a bounding sphere (not an AABB) to fix the XY extents so the ortho
/// projection size is rotation-invariant — the camera rotating no longer changes
/// the projection bounds, eliminating the warping artefact. The sphere centre is
/// then snapped to the nearest shadow texel so the grid only shifts by whole-
/// texel steps during camera translation. `depth_range` is the light-space
/// near/far of the whole scene, so every caster lands in every cascade.
fn cascade_vp(camera: &Camera, aspect: f32, light_view: &Mat4, near_z: f32, far_z: f32, depth_range: (f32, f32)) -> Mat4 {
    let th = (camera.fovy * 0.5).tan();
    let nh = near_z * th;
    let nw = nh * aspect;
    let fh = far_z * th;
    let fw = fh * aspect;

    let corners_vs = [
        Vec3::new(-nw, -nh, -near_z), Vec3::new( nw, -nh, -near_z),
        Vec3::new(-nw,  nh, -near_z), Vec3::new( nw,  nh, -near_z),
        Vec3::new(-fw, -fh, -far_z),  Vec3::new( fw, -fh, -far_z),
        Vec3::new(-fw,  fh, -far_z),  Vec3::new( fw,  fh, -far_z),
    ];

    let inv_cam_view = camera.view_matrix().inversed();

    // Convert corners to world space.
    let mut corners_ws = [Vec3::zero(); 8];
    let mut center_ws  = Vec3::zero();
    for (i, c) in corners_vs.iter().enumerate() {
        let w4 = inv_cam_view * Vec4::new(c.x, c.y, c.z, 1.0);
        corners_ws[i] = Vec3::new(w4.x, w4.y, w4.z);
        center_ws += corners_ws[i];
    }
    center_ws /= 8.0;

    // Bounding sphere radius — constant for a given (near_z, far_z, fov, aspect)
    // regardless of camera orientation, so the XY projection bounds never change
    // size as the camera rotates.
    let radius = corners_ws
        .iter()
        .map(|c| (*c - center_ws).mag())
        .fold(0f32, f32::max);

    // Project sphere centre into light space and snap to whole-texel steps.
    let c_ls = *light_view * Vec4::new(center_ws.x, center_ws.y, center_ws.z, 1.0);
    let texel = 2.0 * radius / SHADOW_MAP_SIZE as f32;
    let sx = (c_ls.x / texel).round() * texel;
    let sy = (c_ls.y / texel).round() * texel;

    // Fixed square XY ortho bounds around the snapped centre.
    let l = sx - radius;
    let r = sx + radius;
    let b = sy - radius;
    let t = sy + radius;

    ortho_rh_wgpu(l, r, b, t, depth_range.0, depth_range.1) * *light_view
}

/// PSSM split scheme blending logarithmic and uniform distributions.
fn compute_cascade_splits(near: f32, far: f32) -> [f32; NUM_CASCADES] {
    let mut splits = [0f32; NUM_CASCADES];
    for (i, split) in splits.iter_mut().enumerate() {
        let t = (i + 1) as f32 / NUM_CASCADES as f32;
        let log = near * (far / near).powf(t);
        let uni = near + (far - near) * t;
        *split = CASCADE_LAMBDA * log + (1.0 - CASCADE_LAMBDA) * uni;
    }
    splits
}

impl PassDesc for ShadowPass {
    fn name(&self) -> &'static str {
        "shadow"
    }

    fn reads(&self) -> Vec<ResourceId> {
        vec![self.shadowed_buffer_id]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![
            self.shadow_map_id,
            self.shadow_params_id,
            self.cascade_vp_id,
            self.local_map_id,
            self.local_params_id,
            self.local_vp_id,
            self.glass_tint_id,
            self.local_glass_tint_id,
        ]
    }
}

impl Pass for ShadowPass {
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
        let Some(&params_buf) = ctx.buffers.get(&self.shadow_params_id) else {
            return;
        };
        let Some(&cascade_vp_buf) = ctx.buffers.get(&self.cascade_vp_id) else {
            return;
        };

        let cameras = world.get_entities::<Camera>();
        let Some(camera) = cameras.first() else {
            return;
        };

        let sky_lights = world.get_entities::<SkyLight>();
        let dir = sky_lights
            .first()
            .map(|l| l.direction.normalized())
            .unwrap_or(Vec3::new(0.0, -1.0, 0.0));

        let up = if dir.dot(Vec3::unit_y()).abs() > 0.99 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::unit_y()
        };
        let light_view = Mat4::look_at(camera.eye - dir, camera.eye, up);

        // Glass and other blended surfaces stay out of the depth maps; they tint the light that
        // passes them through the glass tint maps instead.
        let (glass, meshes): (Vec<_>, Vec<_>) =
            world.get_entity_handles::<Mesh>().into_iter().partition(|h| h.entity.is_blended());
        let mut scene = Aabb::empty();
        for handle in meshes.iter().chain(&glass) {
            if let Some(draw) = ctx.geometry.get(handle.id) {
                let world_bounds = draw.bounds.transformed(handle.entity.transform);
                if !world_bounds.is_empty() {
                    scene.extend(world_bounds.min);
                    scene.extend(world_bounds.max);
                }
            }
        }
        if scene.is_empty() {
            return;
        }

        // Cascades only need to reach the farthest scene point, and their depth
        // range only needs to span the scene, so precision and bias follow the
        // scene's scale rather than the camera's far plane.
        let view = camera.view_matrix();
        let (mut min_z, mut max_z) = (f32::MAX, f32::MIN);
        let mut shadow_far = 0f32;
        for corner in scene.corners() {
            shadow_far = shadow_far.max(-view.transform_point3(corner).z);
            let z = light_view.transform_point3(corner).z;
            min_z = min_z.min(z);
            max_z = max_z.max(z);
        }
        let shadow_far = shadow_far.clamp(camera.near * 2.0, camera.far);
        let pad = (max_z - min_z) * 0.01 + 1e-3;
        let depth_range = (-max_z - pad, -min_z + pad);

        let splits = compute_cascade_splits(camera.near, shadow_far);

        // Lazy init: pipeline + cascade bind group + per-layer views.
        if self.pipeline.is_none() {
            let module = self.shader.compile(device).expect("shadow shader compile failed");

            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(VP_SIZE),
                    },
                    count: None,
                }],
            });

            let instance_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow_instance_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ..GeometryArena::instance_layout_entry(0)
                }],
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow_pipeline_layout"),
                bind_group_layouts: &[Some(&bgl), Some(&instance_layout)],
                immediate_size: 0,
            });

            let depth_pipeline = |label: &str,
                                  layout: &wgpu::PipelineLayout,
                                  masked: bool,
                                  cull_mode: Option<wgpu::Face>,
                                  bias: wgpu::DepthBiasState| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some(if masked { "vs_masked" } else { "vs_main" }),
                        buffers: &[Some(crate::mesh::Vertex::buffer_descriptor())],
                        compilation_options: Default::default(),
                    },
                    fragment: masked.then(|| wgpu::FragmentState {
                        module: &module,
                        entry_point: Some("fs_masked"),
                        targets: &[],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState { cull_mode, ..Default::default() },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(wgpu::CompareFunction::LessEqual),
                        stencil: wgpu::StencilState::default(),
                        bias,
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                })
            };
            // Cube faces sit inside rooms and see walls from both sides, so they can't rely on
            // front-face culling against acne like the cascades do.
            let local_bias = wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 };

            let pipeline = depth_pipeline(
                "shadow_pipeline",
                &pipeline_layout,
                false,
                Some(wgpu::Face::Front),
                wgpu::DepthBiasState::default(),
            );
            self.local_pipeline = Some(depth_pipeline("shadow_local_pipeline", &pipeline_layout, false, None, local_bias));

            let masked_texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow_masked_texture_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

            let masked_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow_masked_pipeline_layout"),
                bind_group_layouts: &[Some(&bgl), Some(&instance_layout), Some(&masked_texture_layout)],
                immediate_size: 0,
            });

            let masked_pipeline = depth_pipeline(
                "shadow_masked_pipeline",
                &masked_pipeline_layout,
                true,
                None,
                wgpu::DepthBiasState::default(),
            );
            self.local_masked_pipeline =
                Some(depth_pipeline("shadow_local_masked_pipeline", &masked_pipeline_layout, true, None, local_bias));

            self.glass_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("shadow_glass_pipeline"),
                layout: Some(&masked_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_glass"),
                    buffers: &[Some(crate::mesh::Vertex::buffer_descriptor())],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_glass"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: GLASS_TINT_FORMAT,
                        blend: Some(GLASS_TINT_BLEND),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            }));

            let clear_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow_clear_layout"),
                bind_group_layouts: &[],
                immediate_size: 0,
            });
            let clear_pipeline = |label: &str, fragment: Option<wgpu::FragmentState>, depth: bool| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&clear_layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vs_clear"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment,
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: depth.then(|| wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(wgpu::CompareFunction::Always),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                })
            };
            self.clear_depth_pipeline = Some(clear_pipeline("shadow_clear_depth", None, true));
            self.clear_tint_pipeline = Some(clear_pipeline(
                "shadow_clear_tint",
                Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_clear_tint"),
                    targets: &[Some(GLASS_TINT_FORMAT.into())],
                    compilation_options: Default::default(),
                }),
                false,
            ));

            self.albedo = Some(AlbedoUpload {
                mips: MipGenerator::new(device),
                layout: masked_texture_layout,
                sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("shadow_albedo_sampler"),
                    address_mode_u: wgpu::AddressMode::Repeat,
                    address_mode_v: wgpu::AddressMode::Repeat,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::MipmapFilterMode::Linear,
                    ..Default::default()
                }),
            });
            self.masked_pipeline = Some(masked_pipeline);

            self.cascade_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shadow_cascade_bg"),
                layout: &bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        size: wgpu::BufferSize::new(VP_SIZE),
                        ..cascade_vp_buf.buffer_binding()
                    }),
                }],
            }));

            if let Some(&local_vp_buf) = ctx.buffers.get(&self.local_vp_id) {
                self.local_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("shadow_local_bg"),
                    layout: &bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            size: wgpu::BufferSize::new(VP_SIZE),
                            ..local_vp_buf.buffer_binding()
                        }),
                    }],
                }));
            }

            self.bgl = Some(bgl);
            self.instance_layout = Some(instance_layout);
            self.pipeline = Some(pipeline);
        }

        // Per-layer render attachment views, created once from the graph-allocated textures.
        let layer_views = |id: &ResourceId, layers: usize| -> Vec<wgpu::TextureView> {
            let Some(&tex) = ctx.textures.get(id) else { return Vec::new() };
            (0..layers as u32)
                .map(|layer| {
                    tex.create_view(&wgpu::TextureViewDescriptor {
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        base_array_layer: layer,
                        array_layer_count: Some(1),
                        ..Default::default()
                    })
                })
                .collect()
        };
        if self.cascade_views.is_empty() {
            self.cascade_views = layer_views(&self.shadow_map_id, NUM_CASCADES);
        }
        if self.glass_tint_views.is_empty() {
            self.glass_tint_views = layer_views(&self.glass_tint_id, NUM_CASCADES);
        }

        if self.cascade_views.is_empty() || self.glass_tint_views.is_empty() {
            return;
        }

        let generation = ctx.geometry.generation();
        if self.instance_bind_group.as_ref().is_none_or(|(g, _)| *g != generation)
            && let Some(layout) = self.instance_layout.as_ref()
        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shadow_instances"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ctx.geometry.instance_buffer().as_entire_binding(),
                }],
            });
            self.instance_bind_group = Some((generation, bind_group));
        }
        if let Some(albedo) = &self.albedo {
            let masked = meshes.iter().map(|h| h.entity).filter(|m| m.is_masked());
            albedo.sync(device, queue, encoder, &mut self.masked_textures, masked, |mesh| TextureKind::Linear {
                alpha_cutoff: albedo_alpha_cutoff(mesh),
            });
            let blended = glass.iter().map(|h| h.entity);
            albedo.sync(device, queue, encoder, &mut self.glass_textures, blended, |_| TextureKind::Color {
                alpha_cutoff: None,
            });
        }

        let Some((_, instance_bg)) = self.instance_bind_group.as_ref() else {
            return;
        };

        let (Some(pipeline), Some(masked_pipeline), Some(glass_pipeline)) =
            (&self.pipeline, &self.masked_pipeline, &self.glass_pipeline)
        else {
            return;
        };
        let mut masked: Vec<_> = meshes.iter().filter(|h| h.entity.is_masked()).collect();
        masked.sort_unstable_by_key(|h| albedo_key(h.entity));
        let inv_cam_view = camera.view_matrix().inversed();
        let (w, h) = ctx.surface_size;
        let aspect = camera.aspect_or(w as f32 / h as f32);

        let mut view_to_shadow_raw = [[[0f32; 4]; 4]; NUM_CASCADES];

        for i in 0..NUM_CASCADES {
            let near_z = if i == 0 { camera.near } else { splits[i - 1] };
            let far_z  = splits[i];
            let light_vp = cascade_vp(camera, aspect, &light_view, near_z, far_z, depth_range);

            let v2s = light_vp * inv_cam_view;
            let cols = v2s.cols;
            view_to_shadow_raw[i] = [cols[0].into(), cols[1].into(), cols[2].into(), cols[3].into()];

            let lp_cols = light_vp.cols;
            let vp_raw: [[f32; 4]; 4] = [
                lp_cols[0].into(), lp_cols[1].into(), lp_cols[2].into(), lp_cols[3].into(),
            ];
            let vp_offset = CASCADE_VP_STRIDE * i as u64;
            let view = ShadowView { view_proj: vp_raw, light: [dir.x, dir.y, dir.z, 0.0] };
            cascade_vp_buf.write_at(queue, vp_offset, bytemuck::bytes_of(&view));

            let layer_view = &self.cascade_views[i];
            let bg = self.cascade_bind_group.as_ref().expect("shadow cascade bind group");

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&format!("shadow_cascade{i}")),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: layer_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bg, &[vp_offset as u32]);
            pass.set_bind_group(1, instance_bg, &[]);
            pass.set_vertex_buffer(0, ctx.geometry.vertex_buffer().slice(..));
            pass.set_index_buffer(ctx.geometry.index_buffer().slice(..), wgpu::IndexFormat::Uint32);

            for (_, draw) in visible(meshes.iter().filter(|h| !h.entity.is_masked()), ctx.geometry, light_vp) {
                pass.draw_indexed(draw.indices.clone(), draw.base_vertex, draw.instances());
            }

            if !masked.is_empty() {
                pass.set_pipeline(masked_pipeline);
                pass.set_bind_group(0, bg, &[vp_offset as u32]);
                pass.set_bind_group(1, instance_bg, &[]);
                let mut current_texture = None;
                for (handle, draw) in visible(masked.iter().copied(), ctx.geometry, light_vp) {
                    let key = albedo_key(handle.entity);
                    if current_texture != Some(key) {
                        let Some((_, texture_bg)) = self.masked_textures.get(&key).or(self.masked_textures.get(&0))
                        else {
                            continue;
                        };
                        pass.set_bind_group(2, texture_bg, &[]);
                        current_texture = Some(key);
                    }
                    pass.draw_indexed(draw.indices.clone(), draw.base_vertex, draw.instances());
                }
            }
            drop(pass);

            draw_glass(
                encoder,
                &self.glass_tint_views[i],
                glass_pipeline,
                (bg, vp_offset as u32),
                instance_bg,
                &self.glass_textures,
                ctx.geometry,
                &glass,
                light_vp,
            );
        }

        self.render_local_shadows(encoder, queue, ctx, &meshes, &masked, &glass);

        let params = ShadowParams {
            view_to_shadow: view_to_shadow_raw,
            cascade_splits: splits,
            bias: 0.002,
            inv_shadow_map_size: 1.0 / SHADOW_MAP_SIZE as f32,
            _pad: [0.0; 2],
        };
        params_buf.write(queue, bytemuck::bytes_of(&params));
    }
}
