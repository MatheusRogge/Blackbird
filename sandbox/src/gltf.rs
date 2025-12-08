use engine::{
    Engine,
    application::ApplicationError,
    asset::{Asset, AssetError, AssetResolver},
    plugin::{EnginePlugin, EnginePluginError},
    world::World,
};
use gltf::{import_slice, mesh::util::ReadIndices};
use rendering::{
    camera::Camera,
    mesh::{Mesh, Vertex},
};
use std::{
    fs::File,
    io::{self, Read},
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

impl From<GLTFError> for ApplicationError {
    fn from(val: GLTFError) -> Self {
        match val {
            GLTFError::LoadSceneError => ApplicationError {
                message: val.to_string(),
                source: Box::new(val),
            },
        }
    }
}

pub struct GLTFAsset {
    pub(crate) document: gltf::Document,
    pub(crate) buffers: Vec<gltf::buffer::Data>,
    // pub(crate) images: Vec<gltf::image::Data>,
}

impl Asset for GLTFAsset {}

pub struct GLTFAssetResolver;

impl AssetResolver for GLTFAssetResolver {
    type Asset = GLTFAsset;
    type Error = GLTFAssetError;

    fn resolve(&self, file: File) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        let mut reader = io::BufReader::new(file);
        let _ = reader.read_to_end(&mut bytes)?;

        let (document, buffers, _images) = import_slice(bytes.as_slice())?;

        Ok(GLTFAsset {
            document,
            buffers,
            // images,
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
                        ReadIndices::U32(_) => Vec::default(),
                        ReadIndices::U8(iter) => iter.map(|e| e as u16).collect(),
                        ReadIndices::U16(iter) => iter.collect(),
                    };

                    indices = values;
                }

                if let Some(positions) = reader.read_positions() {
                    for position in positions {
                        vertices.push(Vertex {
                            position,
                            color: [0.0, 0.0, 0.0],
                        });
                    }
                }
            }

            if !vertices.is_empty() {
                let mut mesh = Mesh::new(vertices);
                if !indices.is_empty() {
                    mesh.push_indices(indices);
                }

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
                        eye: (0.0, 0.0, 0.0).into(),
                        target: (0.0, 0.0, 0.0).into(),
                        up: (0.0, 1.0, 0.0).into(),
                        aspect: perspective.aspect_ratio().unwrap_or((4 / 3) as f32),
                        field_of_view: perspective.yfov(),
                        znear: perspective.znear(),
                        zfar: perspective.zfar().unwrap_or(1000.0),
                    });
                }
            }
        }

        Ok(())
    }
}
