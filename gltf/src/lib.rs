pub mod lights;

use assets::{Asset, AssetError, AssetResolver, LoadContext};
use engine_core::{entity::EntityKey, world::World};
use gltf::{Gltf, buffer, image, image::Format, mesh::util::ReadIndices};
use rendering::{
    TextureAsset,
    camera::Camera,
    mesh::{Mat4, Mesh, Vertex},
};
use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    path::Path,
    sync::Arc,
};
use thiserror::Error;
use ultraviolet::Vec4;

#[derive(Error, Debug)]
pub enum GLTFAssetError {
    #[error(transparent)]
    Asset(#[from] AssetError),

    #[error(transparent)]
    DocumentError(#[from] gltf::Error),

    #[error("image {uri}: {source}")]
    Image { uri: String, source: ::image::ImageError },
}

impl From<GLTFAssetError> for AssetError {
    fn from(value: GLTFAssetError) -> Self {
        match value {
            GLTFAssetError::Asset(e) => e,
            other => AssetError::Other(format!("GLTF document error: {other}")),
        }
    }
}

pub struct GLTFAsset {
    pub document: gltf::Document,
    pub buffers: Vec<gltf::buffer::Data>,
    pub images: Vec<gltf::image::Data>,
}

impl Asset for GLTFAsset {}

/// Loads `.gltf`/`.glb`. External buffers and images are read through the `LoadContext`, so
/// scenes load the same from loose files and from packs.
pub struct GLTFAssetResolver;

/// A relative URI naming a file next to the document, percent-decoded. `None` for data URIs.
fn external_uri(uri: &str) -> Option<Cow<'_, str>> {
    if uri.starts_with("data:") {
        return None;
    }
    Some(urlencoding::decode(uri).unwrap_or(Cow::Borrowed(uri)))
}

fn decode_image(bytes: &[u8], uri: &str) -> Result<image::Data, GLTFAssetError> {
    let rgba = ::image::load_from_memory(bytes)
        .map_err(|source| GLTFAssetError::Image { uri: uri.to_string(), source })?
        .into_rgba8();
    Ok(image::Data {
        width: rgba.width(),
        height: rgba.height(),
        format: Format::R8G8B8A8,
        pixels: rgba.into_raw(),
    })
}

impl AssetResolver for GLTFAssetResolver {
    type Asset = GLTFAsset;
    type Error = GLTFAssetError;

    fn resolve(&self, ctx: &mut LoadContext<'_>) -> Result<Self::Asset, Self::Error> {
        let Gltf { document, mut blob } = Gltf::from_slice(ctx.bytes())?;

        let mut buffers = Vec::new();
        for buffer in document.buffers() {
            let data = match buffer.source() {
                buffer::Source::Uri(uri) if let Some(file) = external_uri(uri) => {
                    let mut data = ctx.read_relative(&file)?;
                    data.resize(data.len().next_multiple_of(4), 0);
                    buffer::Data(data)
                }
                source => buffer::Data::from_source_and_blob(source, None, &mut blob)?,
            };
            if data.len() < buffer.length() {
                return Err(gltf::Error::BufferLength {
                    buffer: buffer.index(),
                    expected: buffer.length(),
                    actual: data.len(),
                }
                .into());
            }
            buffers.push(data);
        }

        let mut images = Vec::new();
        for img in document.images() {
            let data = match img.source() {
                image::Source::Uri { uri, .. } if let Some(file) = external_uri(uri) => {
                    decode_image(&ctx.read_relative(&file)?, uri)?
                }
                source => image::Data::from_source(source, Some(Path::new("")), &buffers)?,
            };
            images.push(data);
        }

        Ok(GLTFAsset { document, buffers, images })
    }

    fn dependencies(&self, ctx: &LoadContext<'_>) -> Result<Vec<String>, Self::Error> {
        let document = Gltf::from_slice_without_validation(ctx.bytes())?.document;
        let buffers = document.buffers().filter_map(|b| match b.source() {
            buffer::Source::Uri(uri) => external_uri(uri),
            buffer::Source::Bin => None,
        });
        let images = document.images().filter_map(|i| match i.source() {
            image::Source::Uri { uri, .. } => external_uri(uri),
            image::Source::View { .. } => None,
        });
        Ok(buffers.chain(images).map(Cow::into_owned).collect())
    }
}

fn image_to_texture_asset(img: &gltf::image::Data) -> Option<Arc<TextureAsset>> {
    let rgba: Vec<u8> = match img.format {
        Format::R8G8B8A8 => img.pixels.clone(),
        Format::R8G8B8 => img
            .pixels
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|&[r, g, b]| [r, g, b, 255])
            .collect(),
        other => {
            log::warn!("Unsupported image format: {:?} — skipping texture", other);
            return None;
        }
    };

    Some(Arc::new(TextureAsset::new(img.width, img.height, rgba)))
}

/// Decodes each image at most once, so primitives sharing a texture share one
/// `Arc` (and one GPU texture).
#[derive(Default)]
struct TextureCache {
    textures: HashMap<usize, Option<Arc<TextureAsset>>>,
}

impl TextureCache {
    fn get(&mut self, images: &[gltf::image::Data], texture: gltf::Texture<'_>) -> Option<Arc<TextureAsset>> {
        let index = texture.source().index();
        self.textures
            .entry(index)
            .or_insert_with(|| images.get(index).and_then(image_to_texture_asset))
            .clone()
    }
}

fn build_mesh_primitives(
    mesh: &gltf::Mesh<'_>,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
    textures: &mut TextureCache,
) -> Vec<Mesh> {
    let mut out = Vec::new();

    for primitive in mesh.primitives() {
        let reader = primitive.reader(|e| buffers.get(e.index()).map(|e| e.0.as_slice()));

        let positions: Vec<[f32; 3]> = match reader.read_positions() {
            Some(it) => it.collect(),
            None => continue,
        };

        let mut vertices: Vec<Vertex> = positions
            .into_iter()
            .map(|position| Vertex {
                position,
                color: [1.0, 1.0, 1.0],
                normal: [0.0, 0.0, 0.0],
                uv: [0.0, 0.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            })
            .collect();

        if let Some(normals) = reader.read_normals() {
            for (i, normal) in normals.enumerate() {
                if let Some(v) = vertices.get_mut(i) {
                    v.normal = normal;
                }
            }
        }

        if let Some(colors) = reader.read_colors(0) {
            for (i, color) in colors.into_rgb_f32().enumerate() {
                if let Some(v) = vertices.get_mut(i) {
                    v.color = color;
                }
            }
        }

        if let Some(uvs) = reader.read_tex_coords(0) {
            for (i, uv) in uvs.into_f32().enumerate() {
                if let Some(v) = vertices.get_mut(i) {
                    v.uv = uv;
                }
            }
        }

        if let Some(tangents) = reader.read_tangents() {
            for (i, tangent) in tangents.enumerate() {
                if let Some(v) = vertices.get_mut(i) {
                    v.tangent = tangent;
                }
            }
        }

        let indices: Vec<u32> = match reader.read_indices() {
            Some(ReadIndices::U32(iter)) => iter.collect(),
            Some(ReadIndices::U16(iter)) => iter.map(|e| e as u32).collect(),
            Some(ReadIndices::U8(iter)) => iter.map(|e| e as u32).collect(),
            None => (0..vertices.len() as u32).collect(),
        };

        let material = primitive.material();
        let pbr = material.pbr_metallic_roughness();

        let albedo_texture = pbr
            .base_color_texture()
            .and_then(|info| textures.get(images, info.texture()));

        let normal_texture = material
            .normal_texture()
            .and_then(|info| textures.get(images, info.texture()));

        out.push(Mesh::new(vertices, indices, albedo_texture, normal_texture));
    }

    out
}

/// Matrix of a node's own transform, relative to its parent.
pub fn local_matrix(node: &gltf::Node<'_>) -> Mat4 {
    node.transform().matrix().into()
}

/// Calls `f` with every node of the default scene (or of all root nodes if there is none)
/// and its world matrix, in depth-first order.
fn visit_scene<'a>(document: &'a gltf::Document, mut f: impl FnMut(gltf::Node<'a>, Mat4)) {
    fn visit<'a>(node: gltf::Node<'a>, parent: Mat4, f: &mut impl FnMut(gltf::Node<'a>, Mat4)) {
        let world = parent * local_matrix(&node);
        f(node.clone(), world);
        for child in node.children() {
            visit(child, world, f);
        }
    }

    let roots: Vec<gltf::Node<'_>> = match document.default_scene().or_else(|| document.scenes().next()) {
        Some(scene) => scene.nodes().collect(),
        None => {
            let children: HashSet<usize> = document
                .nodes()
                .flat_map(|n| n.children().map(|c| c.index()))
                .collect();
            document.nodes().filter(|n| !children.contains(&n.index())).collect()
        }
    };

    for root in roots {
        visit(root, Mat4::identity(), &mut f);
    }
}

fn mesh_nodes(document: &gltf::Document) -> Vec<(usize, Mat4)> {
    let mut out = Vec::new();
    visit_scene(document, |node, world| {
        if node.mesh().is_some() {
            out.push((node.index(), world));
        }
    });
    out
}

pub struct GltfScene {
    asset: Arc<GLTFAsset>,
    pending: Vec<(usize, Mat4)>,
    next_node: usize,
    cameras_loaded: bool,
    textures: TextureCache,
    node_entities: HashMap<usize, Vec<EntityKey>>,
}

impl GltfScene {
    pub fn new(asset: Arc<GLTFAsset>) -> Self {
        Self {
            pending: mesh_nodes(&asset.document),
            asset,
            next_node: 0,
            cameras_loaded: false,
            textures: TextureCache::default(),
            node_entities: HashMap::new(),
        }
    }

    pub fn is_complete(&self) -> bool {
        self.next_node >= self.pending.len() && self.cameras_loaded
    }

    /// Mesh entities created so far, keyed by glTF node index. Each primitive of
    /// the node's mesh is one entity.
    pub fn node_entities(&self) -> &HashMap<usize, Vec<EntityKey>> {
        &self.node_entities
    }

    /// Inserts the meshes of up to `count` nodes into `world`, each placed at its
    /// node's world transform. Returns `true` if more remain.
    pub fn load_batch(&mut self, world: &mut World, count: usize) -> bool {
        if !self.cameras_loaded {
            load_cameras(&self.asset.document, world);
            self.cameras_loaded = true;
        }

        let end = (self.next_node + count).min(self.pending.len());
        for &(node_index, transform) in &self.pending[self.next_node..end] {
            let Some(mesh) = self.asset.document.nodes().nth(node_index).and_then(|n| n.mesh()) else {
                continue;
            };
            let primitives =
                build_mesh_primitives(&mesh, &self.asset.buffers, &self.asset.images, &mut self.textures);
            let keys = self.node_entities.entry(node_index).or_default();
            for entity in primitives {
                keys.push(world.add_entity(entity.with_transform(transform)));
            }
        }
        self.next_node = end;

        !self.is_complete()
    }
}

/// Each perspective camera node becomes a `Camera` looking down the node's −Z.
fn load_cameras(document: &gltf::Document, world: &mut World) {
    visit_scene(document, |node, transform| {
        let Some(camera) = node.camera() else { return };
        let gltf::camera::Projection::Perspective(perspective) = camera.projection() else { return };
        let eye = transform.cols[3].xyz();
        let forward = (transform * Vec4::new(0.0, 0.0, -1.0, 0.0)).xyz().normalized();
        let up = (transform * Vec4::new(0.0, 1.0, 0.0, 0.0)).xyz().normalized();
        world.add_entity(Camera {
            fovy: perspective.yfov(),
            near: perspective.znear(),
            far: perspective.zfar().unwrap_or(1000.0),
            exposure: 1.0,
            aspect: perspective.aspect_ratio().unwrap_or(4.0 / 3.0),
            up,
            eye,
            target: eye + forward,
        });
    });
}
