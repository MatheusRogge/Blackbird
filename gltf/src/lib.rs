use engine::{
    Engine,
    asset::{Asset, AssetError, AssetResolver},
    plugin::{EnginePlugin, EnginePluginError},
    world::World,
};
use gltf::{Gltf, import_buffers, import_images, mesh::util::ReadIndices};
use rendering::{
    camera::Camera,
    mesh::{Mesh, Vertex},
};
use std::{
    fs::File,
    io::{self},
    path::Path,
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

#[derive(Error, Debug)]
pub enum GLTFError {
    #[error("Scene loading failed")]
    LoadSceneError,
}

impl From<GLTFError> for EnginePluginError {
    fn from(val: GLTFError) -> Self {
        match val {
            GLTFError::LoadSceneError => EnginePluginError {
                message: val.to_string(),
                source: Box::new(val),
            },
        }
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
        println!(
            "[GLTFAssetResolver] Reading: base_path={:?}, file={:?}",
            base_path, &file
        );

        let base = Some(base_path);
        let reader = io::BufReader::new(file);
        let Gltf { document, blob } = Gltf::from_reader(reader)?;
        println!("GLTF document loaded!");

        let buffer_data = import_buffers(&document, base, blob)?;
        println!("Buffers read: {:?}", buffer_data.len());

        let image_data = import_images(&document, base, &buffer_data)?;
        println!("Read images: {:?}", image_data.len());

        Ok(GLTFAsset {
            document,
            images: image_data,
            buffers: buffer_data,
        })
    }
}

pub struct GLTFEnginePlugin;

impl EnginePlugin for GLTFEnginePlugin {
    fn setup(&self, engine: &mut Engine) -> Result<Self, EnginePluginError>
    where
        Self: Sized,
    {
        engine
            .asset_manager()
            .add_resolver("gltf", GLTFAssetResolver);

        engine
            .asset_manager()
            .add_resolver("glb", GLTFAssetResolver);

        Ok(Self)
    }
}

impl GLTFEnginePlugin {
    pub fn load_scene(&self, asset: &GLTFAsset, world: &mut World) -> Result<(), GLTFError> {
        let buffers = &asset.buffers;

        for mesh in asset.document.meshes() {
            let mut indices = Vec::new();
            let mut vertices = Vec::new();

            for primitive in mesh.primitives() {
                let reader = primitive.reader(|e| buffers.get(e.index()).map(|e| e.0.as_slice()));

                if let Some(reader) = reader.read_indices() {
                    let values = match reader {
                        ReadIndices::U32(iter) => iter.map(|e| e as u16).collect(),
                        ReadIndices::U8(iter) => iter.map(|e| e as u16).collect(),
                        ReadIndices::U16(iter) => iter.collect(),
                    };

                    indices = values;
                }

                if let Some(positions) = reader.read_positions() {
                    for position in positions {
                        vertices.push(Vertex {
                            position,
                            color: [1.0, 1.0, 1.0],
                            normal: [0.0, 0.0, 0.0],
                        });
                    }
                }

                if let Some(normals) = reader.read_normals() {
                    for (i, normal) in normals.enumerate() {
                        vertices[i].normal = normal;
                    }
                }

                if let Some(colors) = reader.read_colors(1) {
                    for (i, color) in colors.into_rgb_f32().enumerate() {
                        vertices[i].color = color;
                    }
                }
            }

            if !vertices.is_empty() {
                let mesh = Mesh::new(vertices, indices);
                world.add_entity(mesh);
            }
        }

        for camera in asset.document.cameras() {
            match camera.projection() {
                gltf::camera::Projection::Orthographic(_orthographic) => {
                    // Not implemeted.
                }
                gltf::camera::Projection::Perspective(perspective) => {
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

        Ok(())
    }
}
