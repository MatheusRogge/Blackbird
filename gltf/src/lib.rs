use assets::{Asset, AssetError, AssetResolver};
use engine_core::world::World;
use gltf::{Gltf, image::Format, import_buffers, import_images, mesh::util::ReadIndices};
use rendering::{
    TextureAsset,
    camera::Camera,
    mesh::{Mesh, Vertex},
};
use std::{
    fs::File,
    io::{self},
    path::Path,
    sync::Arc,
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GLTFAssetError {
    #[error(transparent)]
    IOError(#[from] io::Error),

    #[error(transparent)]
    DocumentError(#[from] gltf::Error),
}

impl From<GLTFAssetError> for AssetError {
    fn from(value: GLTFAssetError) -> Self {
        AssetError::Other(format!("GLTF document error: {}", value))
    }
}

pub struct GLTFAsset {
    pub(crate) document: gltf::Document,
    pub(crate) buffers: Vec<gltf::buffer::Data>,
    pub(crate) images: Vec<gltf::image::Data>,
}

impl Asset for GLTFAsset {}

pub struct GLTFAssetResolver;

impl AssetResolver for GLTFAssetResolver {
    type Asset = GLTFAsset;
    type Error = GLTFAssetError;

    fn resolve(&self, base_path: &Path, file: File) -> Result<Self::Asset, Self::Error> {
        let base = Some(base_path);

        let reader = io::BufReader::new(file);
        let Gltf { document, blob } = Gltf::from_reader(reader)?;

        let buffer_data = import_buffers(&document, base, blob)?;
        let image_data = import_images(&document, base, &buffer_data)?;

        Ok(GLTFAsset {
            document,
            images: image_data,
            buffers: buffer_data,
        })
    }
}

fn image_to_texture_asset(img: &gltf::image::Data) -> Option<Arc<TextureAsset>> {
    let rgba: Vec<u8> = match img.format {
        Format::R8G8B8A8 => img.pixels.clone(),
        Format::R8G8B8 => img
            .pixels
            .chunks_exact(3)
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect(),
        other => {
            log::warn!("Unsupported image format: {:?} — skipping texture", other);
            return None;
        }
    };

    Some(Arc::new(TextureAsset::new(img.width, img.height, rgba)))
}

fn build_mesh_primitives(
    mesh: &gltf::Mesh<'_>,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
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

        let indices: Vec<u32> = match reader.read_indices() {
            Some(ReadIndices::U32(iter)) => iter.collect(),
            Some(ReadIndices::U16(iter)) => iter.map(|e| e as u32).collect(),
            Some(ReadIndices::U8(iter)) => iter.map(|e| e as u32).collect(),
            None => (0..vertices.len() as u32).collect(),
        };

        let albedo_texture = primitive
            .material()
            .pbr_metallic_roughness()
            .base_color_texture()
            .and_then(|tex_info| images.get(tex_info.texture().source().index()))
            .and_then(image_to_texture_asset);

        out.push(Mesh::new(vertices, indices, albedo_texture));
    }

    out
}

pub struct GltfScene {
    asset: Arc<GLTFAsset>,
    next_mesh: usize,
    cameras_loaded: bool,
}

impl GltfScene {
    pub fn new(asset: Arc<GLTFAsset>) -> Self {
        Self {
            asset,
            next_mesh: 0,
            cameras_loaded: false,
        }
    }

    pub fn is_complete(&self) -> bool {
        let total = self.asset.document.meshes().count();
        self.next_mesh >= total && self.cameras_loaded
    }

    /// Inserts up to `count` meshes into `world`. Returns `true` if more remain.
    pub fn load_batch(&mut self, world: &mut World, count: usize) -> bool {
        if !self.cameras_loaded {
            load_cameras(&self.asset.document, world);
            self.cameras_loaded = true;
        }

        let meshes = self
            .asset
            .document
            .meshes()
            .skip(self.next_mesh)
            .take(count);

        if meshes.len() == 0 {
            return false;
        }

        for mesh in meshes {
            self.next_mesh = mesh.index() + 1;
            for entity in build_mesh_primitives(&mesh, &self.asset.buffers, &self.asset.images) {
                world.add_entity(entity);
            }
        }

        !self.is_complete()
    }
}

fn load_cameras(document: &gltf::Document, world: &mut World) {
    for camera in document.cameras() {
        if let gltf::camera::Projection::Perspective(perspective) = camera.projection() {
            world.add_entity(Camera {
                fovy: perspective.yfov(),
                near: perspective.znear(),
                far: perspective.zfar().unwrap_or(1000.0),
                aspect: perspective.aspect_ratio().unwrap_or((4 / 3) as f32),
                up: (0.0, 1.0, 0.0).into(),
                eye: (0.0, 40.0, 150.0).into(),
                target: (0.0, 40.0, 0.0).into(),
            });
        }
    }
}
