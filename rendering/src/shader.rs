use std::{
    borrow::Cow,
    fs::File,
    io::{self, BufReader, Read},
    path::Path,
};

use assets::{Asset, AssetError, AssetResolver};
use wgpu::{Device, ShaderModule, ShaderModuleDescriptor, ShaderSource};

pub use wgpu::{BindGroupLayoutEntry, BindingType, BufferBindingType, ShaderStages};

#[derive(Debug, Clone)]
pub struct ShaderAsset {
    pub content: String,
    pub bindings: Vec<BindGroupLayoutEntry>,
}

impl Asset for ShaderAsset {}

impl ShaderAsset {
    pub fn from_raw(content: impl Into<String>) -> Self {
        Self {
            bindings: Vec::new(),
            content: content.into(),
        }
    }

    pub fn compile(&self, device: &Device) -> Result<ShaderModule, io::Error> {
        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Borrowed(&self.content)),
        });

        Ok(shader_module)
    }

    pub fn bind(&mut self, entry: BindGroupLayoutEntry) {
        self.bindings.push(entry);
    }
}

pub struct ShaderAssetResolver;

impl AssetResolver for ShaderAssetResolver {
    type Asset = ShaderAsset;
    type Error = AssetError;

    fn resolve(&self, _base_path: &Path, file: File) -> Result<Self::Asset, Self::Error> {
        let mut reader = BufReader::new(file);

        let mut content = String::new();
        reader.read_to_string(&mut content)?;

        let asset = ShaderAsset {
            content,
            bindings: Vec::new(),
        };

        Ok(asset)
    }
}
