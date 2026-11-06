use std::sync::Arc;

use engine_core::entity::Entity;

pub use ultraviolet::{Mat4, Vec3};

use crate::texture::TextureAsset;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
    pub uv: [f32; 2],
    pub tangent: [f32; 4],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 5] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x3, 3 => Float32x2, 4 => Float32x4];

    pub fn buffer_descriptor() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            attributes: &Self::ATTRIBS,
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
        }
    }
}

/// glTF metallic-roughness factors. Each multiplies its texture, if any. The default is a
/// dielectric at roughness 0.5, not glTF's fully rough metal, so meshes built in code look plain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaterialFactors {
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    /// `KHR_materials_emissive_strength`, a multiplier on `emissive`.
    pub emissive_strength: f32,
    /// `KHR_materials_transmission`: the fraction of light passing through a `Blend` surface
    /// instead of being diffusely reflected.
    pub transmission: f32,
    /// `KHR_materials_ior`: the dielectric reflectance and, with `thickness`, the refraction of
    /// transmissive surfaces.
    pub ior: f32,
    /// `KHR_materials_volume` thickness in mesh-local units. 0 is thin-walled: no refraction offset.
    pub thickness: f32,
    /// `KHR_materials_volume`: transmitted light reaches `attenuation_color` after travelling
    /// `attenuation_distance` (world units) inside the volume. Infinite means no absorption.
    pub attenuation_color: [f32; 3],
    pub attenuation_distance: f32,
}

impl Default for MaterialFactors {
    fn default() -> Self {
        Self {
            base_color: [1.0; 4],
            metallic: 0.0,
            roughness: 0.5,
            emissive: [0.0; 3],
            emissive_strength: 1.0,
            transmission: 0.0,
            ior: 1.5,
            thickness: 0.0,
            attenuation_color: [1.0; 3],
            attenuation_distance: f32::INFINITY,
        }
    }
}

/// glTF `alphaMode`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AlphaMode {
    #[default]
    Opaque,
    /// Alpha-tested ("cutout"): fragments with albedo alpha below the cutoff are discarded, in the
    /// G-buffer and in shadow maps.
    Mask(f32),
    /// Drawn after the opaque scene by the `transparent` pass, blended back to front. Casts no
    /// shadows and doesn't block GI rays.
    Blend,
}

#[derive(Debug, Clone)]
pub struct Mesh {
    pub indices: Vec<u32>,
    pub vertices: Vec<Vertex>,
    pub albedo_texture: Option<Arc<TextureAsset>>,
    pub normal_texture: Option<Arc<TextureAsset>>,
    /// glTF layout: G = roughness, B = metallic, linear.
    pub metallic_roughness_texture: Option<Arc<TextureAsset>>,
    /// sRGB.
    pub emissive_texture: Option<Arc<TextureAsset>>,
    pub material: MaterialFactors,
    pub alpha_mode: AlphaMode,
    /// Rendered without back-face culling, with normals flipped on back faces.
    pub double_sided: bool,
    /// Local-to-world transform, read every frame. Vertices stay in local space.
    pub transform: Mat4,
}

impl Default for Mesh {
    fn default() -> Self {
        Self {
            indices: Vec::new(),
            vertices: Vec::new(),
            albedo_texture: None,
            normal_texture: None,
            metallic_roughness_texture: None,
            emissive_texture: None,
            material: MaterialFactors::default(),
            alpha_mode: AlphaMode::Opaque,
            double_sided: false,
            transform: Mat4::identity(),
        }
    }
}

impl Mesh {
    pub fn new(
        vertices: Vec<Vertex>,
        indices: Vec<u32>,
        albedo_texture: Option<Arc<TextureAsset>>,
        normal_texture: Option<Arc<TextureAsset>>,
    ) -> Self {
        Self {
            indices,
            vertices,
            albedo_texture,
            normal_texture,
            metallic_roughness_texture: None,
            emissive_texture: None,
            material: MaterialFactors::default(),
            alpha_mode: AlphaMode::Opaque,
            double_sided: false,
            transform: Mat4::identity(),
        }
    }

    pub fn with_material(
        mut self,
        material: MaterialFactors,
        metallic_roughness_texture: Option<Arc<TextureAsset>>,
        emissive_texture: Option<Arc<TextureAsset>>,
    ) -> Self {
        self.material = material;
        self.metallic_roughness_texture = metallic_roughness_texture;
        self.emissive_texture = emissive_texture;
        self
    }

    pub fn with_alpha_mode(mut self, alpha_mode: AlphaMode) -> Self {
        self.alpha_mode = alpha_mode;
        self
    }

    pub fn alpha_cutoff(&self) -> Option<f32> {
        match self.alpha_mode {
            AlphaMode::Mask(cutoff) => Some(cutoff),
            AlphaMode::Opaque | AlphaMode::Blend => None,
        }
    }

    pub fn is_blended(&self) -> bool {
        self.alpha_mode == AlphaMode::Blend
    }

    pub fn with_double_sided(mut self, double_sided: bool) -> Self {
        self.double_sided = double_sided;
        self
    }

    /// Needs the alpha-tested / double-sided path rather than the plain opaque one.
    pub fn is_masked(&self) -> bool {
        self.alpha_cutoff().is_some() || self.double_sided
    }

    pub fn with_transform(mut self, transform: Mat4) -> Self {
        self.transform = transform;
        self
    }

    pub fn get_indices_count(&self) -> usize {
        self.indices.len()
    }

    pub fn get_vertices_count(&self) -> usize {
        self.vertices.len()
    }

    pub fn get_vertex_buffer_content(&self) -> &[u8] {
        bytemuck::cast_slice(self.vertices.as_slice())
    }

    pub fn get_indices_buffer_content(&self) -> &[u8] {
        bytemuck::cast_slice(self.indices.as_slice())
    }
}

impl Entity for Mesh {}

#[derive(Clone, Default)]
pub struct MeshBatch {
    pub meshes: Vec<Mesh>,
    pub viewport_width: u32,
    pub viewport_height: u32,
}

impl MeshBatch {
    pub fn is_empty(&self) -> bool {
        self.meshes.is_empty()
    }

    pub fn packed_vertices(&self) -> Vec<Vertex> {
        self.meshes
            .iter()
            .flat_map(|m| m.vertices.iter().copied())
            .collect()
    }

    pub fn packed_indices(&self) -> Vec<u32> {
        let mut out = Vec::new();
        let mut base: u32 = 0;
        for mesh in &self.meshes {
            for &idx in &mesh.indices {
                out.push(idx + base);
            }
            base += mesh.vertices.len() as u32;
        }
        out
    }

    pub fn total_vertex_count(&self) -> usize {
        self.meshes.iter().map(|m| m.vertices.len()).sum()
    }

    pub fn total_index_count(&self) -> usize {
        self.meshes.iter().map(|m| m.indices.len()).sum()
    }
}
