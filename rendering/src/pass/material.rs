use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::{
    mesh::Mesh,
    pass::mipmap::{MipGenerator, TextureKind},
    texture::TextureAsset,
};

/// `TextureAsset::id`s of a mesh's material (albedo, normal, metallic-roughness, emissive), 0 for none.
pub type MaterialKey = (u64, u64, u64, u64);

fn texture_key(texture: &Option<Arc<TextureAsset>>) -> u64 {
    texture.as_ref().map_or(0, |t| t.id())
}

pub fn material_key(mesh: &Mesh) -> MaterialKey {
    (
        texture_key(&mesh.albedo_texture),
        texture_key(&mesh.normal_texture),
        texture_key(&mesh.metallic_roughness_texture),
        texture_key(&mesh.emissive_texture),
    )
}

/// The alpha test `mesh`'s albedo texture is sampled with: its cutoff divided by the base color
/// factor's alpha, which multiplies the texture's.
pub(crate) fn albedo_alpha_cutoff(mesh: &Mesh) -> Option<f32> {
    mesh.alpha_cutoff().map(|cutoff| (cutoff / mesh.material.base_color[3].max(1e-3)).min(1.0))
}

/// GPU copies of mesh material textures and one bind group per material: albedo, normal,
/// metallic-roughness and emissive textures at bindings 0–3 and a repeating trilinear, 16×
/// anisotropic sampler at 4. Textures get full mip chains (see `MipGenerator`); albedo of
/// alpha-tested meshes keeps its alpha coverage per mip.
/// Missing textures use white / flat / white / white fallbacks (glTF emissive is factor × texture,
/// so no texture means the factor alone).
pub struct MaterialTextures {
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// Keyed by `TextureAsset::id`. Entries no mesh uses any more are dropped by `prepare`.
    textures: HashMap<u64, (wgpu::Texture, wgpu::TextureView)>,
    bind_groups: HashMap<MaterialKey, wgpu::BindGroup>,
    fallback_white: (wgpu::Texture, wgpu::TextureView),
    fallback_normal: (wgpu::Texture, wgpu::TextureView),
    fallback_bind_group: wgpu::BindGroup,
    mips: MipGenerator,
}

impl MaterialTextures {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, label: &str) -> Self {
        let texture_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            count: None,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                view_dimension: wgpu::TextureViewDimension::D2,
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
            },
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &[
                texture_entry(0),
                texture_entry(1),
                texture_entry(2),
                texture_entry(3),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    count: None,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                },
            ],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(label),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 16,
            ..Default::default()
        });

        let mips = MipGenerator::new(device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        let white = TextureAsset::new(1, 1, vec![255, 255, 255, 255]);
        let fallback_white =
            mips.upload(device, queue, &mut encoder, &white, TextureKind::Color { alpha_cutoff: None });
        // [128, 128, 255] decodes to (0, 0, 1) in tangent space — a flat normal map
        let flat = TextureAsset::new(1, 1, vec![128, 128, 255, 255]);
        let fallback_normal = mips.upload(device, queue, &mut encoder, &flat, TextureKind::Normal);
        queue.submit([encoder.finish()]);

        let fallback_bind_group = Self::create_bind_group(
            device,
            &layout,
            &sampler,
            [&fallback_white.1, &fallback_normal.1, &fallback_white.1, &fallback_white.1],
        );

        Self {
            layout,
            sampler,
            textures: HashMap::new(),
            bind_groups: HashMap::new(),
            fallback_white,
            fallback_normal,
            fallback_bind_group,
            mips,
        }
    }

    pub fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        views: [&wgpu::TextureView; 4],
    ) -> wgpu::BindGroup {
        let [albedo, normal, metallic_roughness, emissive] = views;
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(albedo) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(normal) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(metallic_roughness) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(emissive) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(sampler) },
            ],
        })
    }

    /// Uploads the textures of `meshes` that aren't on the GPU yet (their mip chains are recorded
    /// into `encoder`) and builds their bind groups, then drops textures and bind groups none of
    /// them use. Call before opening the render pass. A texture keeps the kind it was first
    /// uploaded with, so an albedo shared by an alpha-tested and an opaque mesh keeps one cutoff.
    pub fn prepare<'m>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        meshes: impl IntoIterator<Item = &'m Mesh>,
    ) {
        let mut live: HashSet<u64> = HashSet::from([0]);
        let mut live_keys: HashSet<MaterialKey> = HashSet::new();
        for mesh in meshes {
            let textures = [
                (&mesh.albedo_texture, TextureKind::Color { alpha_cutoff: albedo_alpha_cutoff(mesh) }),
                (&mesh.normal_texture, TextureKind::Normal),
                (&mesh.metallic_roughness_texture, TextureKind::Linear { alpha_cutoff: None }),
                (&mesh.emissive_texture, TextureKind::Color { alpha_cutoff: None }),
            ];
            for (texture, kind) in textures {
                if let Some(tex) = texture {
                    live.insert(tex.id());
                    self.textures
                        .entry(tex.id())
                        .or_insert_with(|| self.mips.upload(device, queue, encoder, tex, kind));
                }
            }

            let key = material_key(mesh);
            live_keys.insert(key);
            if !self.bind_groups.contains_key(&key) {
                let (albedo, normal, mr, emissive) = key;
                let textures = &self.textures;
                let white = &self.fallback_white.1;
                let view = |id: u64, fallback| textures.get(&id).map_or(fallback, |(_, v)| v);
                let views = [
                    view(albedo, white),
                    view(normal, &self.fallback_normal.1),
                    view(mr, white),
                    view(emissive, white),
                ];
                let bind_group = Self::create_bind_group(device, &self.layout, &self.sampler, views);
                self.bind_groups.insert(key, bind_group);
            }
        }

        self.textures.retain(|id, _| live.contains(id));
        self.bind_groups.retain(|key, _| live_keys.contains(key));
    }

    pub fn bind_group(&self, key: MaterialKey) -> &wgpu::BindGroup {
        self.bind_groups.get(&key).unwrap_or(&self.fallback_bind_group)
    }
}
