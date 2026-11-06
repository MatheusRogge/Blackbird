use wgpu::util::DeviceExt;

use crate::texture::TextureAsset;

const SHADER: &str = include_str!("mipmap.wgsl");

/// What a texture holds, which picks its format and how its mips are filtered.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TextureKind {
    /// sRGB color (albedo, emissive). `alpha_cutoff` is the alpha test it is sampled with, if any:
    /// each mip's alpha is then scaled so the fraction of texels passing the test stays the same,
    /// instead of cut-outs thinning away with distance.
    Color { alpha_cutoff: Option<f32> },
    /// Linear data (metallic-roughness, alpha-only masks).
    Linear { alpha_cutoff: Option<f32> },
    /// Tangent-space normal map: renormalized at every level.
    Normal,
}

impl TextureKind {
    fn format(self) -> wgpu::TextureFormat {
        match self {
            TextureKind::Color { .. } => wgpu::TextureFormat::Rgba8UnormSrgb,
            TextureKind::Linear { .. } | TextureKind::Normal => wgpu::TextureFormat::Rgba8Unorm,
        }
    }

    fn alpha_cutoff(self) -> Option<f32> {
        match self {
            TextureKind::Color { alpha_cutoff } | TextureKind::Linear { alpha_cutoff } => alpha_cutoff,
            TextureKind::Normal => None,
        }
    }
}

pub fn mip_level_count(width: u32, height: u32) -> u32 {
    width.max(height).max(1).ilog2() + 1
}

struct Pipelines {
    downsample: wgpu::RenderPipeline,
    downsample_normal: wgpu::RenderPipeline,
    scale_alpha: wgpu::RenderPipeline,
}

/// Uploads `TextureAsset`s with a full mip chain built on the GPU: each level is a 2×2 box filter of
/// the one above, recorded into the caller's encoder.
pub struct MipGenerator {
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// `alpha_scale` of 1, for the plain downsample passes.
    unit_scale: wgpu::Buffer,
    srgb: Pipelines,
    linear: Pipelines,
}

impl MipGenerator {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mipmap"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mipmap_bgl"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mipmap_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry_point, format: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry_point),
                    targets: &[Some(format.into())],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipelines = |format| Pipelines {
            downsample: pipeline("fs_downsample", format),
            downsample_normal: pipeline("fs_downsample_normal", format),
            scale_alpha: pipeline("fs_scale_alpha", format),
        };

        Self {
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("mipmap_sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            unit_scale: scale_buffer(device, 1.0),
            srgb: pipelines(wgpu::TextureFormat::Rgba8UnormSrgb),
            linear: pipelines(wgpu::TextureFormat::Rgba8Unorm),
            layout,
        }
    }

    /// Creates the texture, writes `asset` into level 0 and records the rest of the chain into
    /// `encoder`, which must be submitted before the texture is sampled.
    pub fn upload(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        asset: &TextureAsset,
        kind: TextureKind,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let format = kind.format();
        let levels = mip_level_count(asset.width, asset.height);
        let texture = create_texture(device, asset, format, levels);
        write_level_0(queue, &texture, asset);

        let pipelines = if format.is_srgb() { &self.srgb } else { &self.linear };
        let downsample = if kind == TextureKind::Normal { &pipelines.downsample_normal } else { &pipelines.downsample };

        match kind.alpha_cutoff().filter(|_| levels > 1) {
            None => {
                for level in 1..levels {
                    self.blit(device, encoder, downsample, &texture, level - 1, &texture, level, &self.unit_scale);
                }
            }
            Some(cutoff) => {
                // The unscaled chain goes into a scratch texture, so each level is filtered from
                // the unscaled one above, then copied over with its scale applied.
                let scratch = create_texture(device, asset, format, levels);
                write_level_0(queue, &scratch, asset);
                let scales = alpha_coverage_scales(asset, cutoff, levels);
                for level in 1..levels {
                    self.blit(device, encoder, downsample, &scratch, level - 1, &scratch, level, &self.unit_scale);
                    let scale = scale_buffer(device, scales[level as usize]);
                    self.blit(device, encoder, &pipelines.scale_alpha, &scratch, level, &texture, level, &scale);
                }
            }
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    #[allow(clippy::too_many_arguments)]
    fn blit(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        src: &wgpu::Texture,
        src_level: u32,
        dst: &wgpu::Texture,
        dst_level: u32,
        alpha_scale: &wgpu::Buffer,
    ) {
        let level_view = |texture: &wgpu::Texture, level| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: level,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };
        let src_view = level_view(src, src_level);
        let dst_view = level_view(dst, dst_level);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mipmap_bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: alpha_scale.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mipmap"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &dst_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn scale_buffer(device: &wgpu::Device, scale: f32) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mipmap_alpha_scale"),
        contents: bytemuck::cast_slice(&[scale, 0.0, 0.0, 0.0]),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

fn create_texture(device: &wgpu::Device, asset: &TextureAsset, format: wgpu::TextureFormat, levels: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: asset.width, height: asset.height, depth_or_array_layers: 1 },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}

fn write_level_0(queue: &wgpu::Queue, texture: &wgpu::Texture, asset: &TextureAsset) {
    let unpadded = 4 * asset.width;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let bytes_per_row = unpadded.div_ceil(align) * align;

    let padded: Vec<u8>;
    let data = if bytes_per_row == unpadded {
        &asset.data
    } else {
        let padding = (bytes_per_row - unpadded) as usize;
        padded = asset
            .data
            .chunks_exact(unpadded as usize)
            .flat_map(|row| row.iter().copied().chain(std::iter::repeat_n(0, padding)))
            .collect();
        &padded
    };

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bytes_per_row), rows_per_image: None },
        wgpu::Extent3d { width: asset.width, height: asset.height, depth_or_array_layers: 1 },
    );
}

/// Per-level alpha multipliers that keep the fraction of texels with `alpha ≥ cutoff` equal to
/// level 0's (Castaño, "Computing Alpha Mipmaps"). Only ever scales up, so no level loses coverage. Mirrors the GPU chain on the alpha channel: each
/// level is the 2×2 box filter of the unscaled level above. Index 0 is level 0 (always 1).
pub fn alpha_coverage_scales(asset: &TextureAsset, cutoff: f32, levels: u32) -> Vec<f32> {
    let (mut width, mut height) = (asset.width as usize, asset.height as usize);
    let mut alpha: Vec<f32> = asset.data.iter().skip(3).step_by(4).map(|&a| a as f32 / 255.0).collect();
    let target = alpha.iter().filter(|&&a| a >= cutoff).count() as f32 / alpha.len().max(1) as f32;

    let mut scales = vec![1.0];
    for _ in 1..levels {
        let (w, h) = ((width / 2).max(1), (height / 2).max(1));
        alpha = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w * 2, i / w * 2);
                let at = |dx: usize, dy: usize| alpha[(y + dy).min(height - 1) * width + (x + dx).min(width - 1)];
                (at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1)) * 0.25
            })
            .collect();
        (width, height) = (w, h);

        // The alpha that `target` of the texels reach: scaling it up to `cutoff` makes exactly
        // those texels pass.
        let mut sorted = alpha.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let passing = ((target * sorted.len() as f32).round() as usize).clamp(1, sorted.len());
        let threshold = sorted[passing - 1];
        scales.push(if target == 0.0 || threshold <= 0.0 { 1.0 } else { (cutoff / threshold).clamp(1.0, 16.0) });
    }
    scales
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_counts() {
        assert_eq!(mip_level_count(1, 1), 1);
        assert_eq!(mip_level_count(512, 512), 10);
        assert_eq!(mip_level_count(2048, 256), 12);
        assert_eq!(mip_level_count(3, 5), 3);
    }

    #[test]
    fn coverage_scale_restores_passing_fraction() {
        // A 4×4 checker of opaque and clear texels: half pass at 0.5. Level 1 averages to 0.5 everywhere.
        let data = (0..16).flat_map(|i| [255, 255, 255, if (i % 4 + i / 4) % 2 == 0 { 255 } else { 0 }]).collect();
        let asset = TextureAsset::new(4, 4, data);
        let scales = alpha_coverage_scales(&asset, 0.6, 3);
        assert_eq!(scales[0], 1.0);
        assert!(scales[1] * 0.5 >= 0.6, "{scales:?}");
    }

    #[test]
    fn opaque_texture_is_unscaled() {
        let asset = TextureAsset::new(2, 2, vec![255; 16]);
        assert_eq!(alpha_coverage_scales(&asset, 0.5, 2), vec![1.0, 1.0]);
    }
}
