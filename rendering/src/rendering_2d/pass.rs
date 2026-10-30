use std::{collections::HashMap, sync::Arc};

use engine_core::world::World;
use crate::{
    graph::NodeId,
    pass::{Pass, PassContext, PassDesc},
    rendering_2d::{camera::Camera2D, sprite::{Sprite, SpriteTexture}},
    resource::ResourceId,
};
use wgpu::util::DeviceExt;

const SHADER: &str = include_str!("shader.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct QuadVertex {
    position: [f32; 2],
    uv: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SpriteInstance {
    model: [[f32; 4]; 4],
    color: [f32; 4],
    uv_rect: [f32; 4],
}

struct GpuTexture {
    _texture: wgpu::Texture,
    _view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

struct Pipeline {
    pipeline: wgpu::RenderPipeline,
    texture_bgl: wgpu::BindGroupLayout,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    quad_vb: wgpu::Buffer,
    quad_ib: wgpu::Buffer,
}

pub struct SpritePass {
    node_id: Option<NodeId>,
    surface_id: ResourceId,
    surface_format: wgpu::TextureFormat,
    clear_color: Option<wgpu::Color>,
    pipeline: Option<Pipeline>,
    // Keyed by Arc<SpriteTexture> pointer identity
    texture_cache: HashMap<usize, GpuTexture>,
}

impl SpritePass {
    pub fn new(
        surface_id: ResourceId,
        surface_format: wgpu::TextureFormat,
        clear_color: Option<wgpu::Color>,
    ) -> Self {
        Self {
            node_id: None,
            surface_id,
            surface_format,
            clear_color,
            pipeline: None,
            texture_cache: HashMap::new(),
        }
    }

    fn ensure_pipeline(&mut self, device: &wgpu::Device) {
        if self.pipeline.is_some() {
            return;
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sprite_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sprite_camera_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let texture_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sprite_texture_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sprite_pipeline_layout"),
            bind_group_layouts: &[Some(&camera_bgl), Some(&texture_bgl)],
            immediate_size: 0,
        });

        let vertex_attrs = [
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            },
            wgpu::VertexAttribute {
                offset: 8,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x2,
            },
        ];

        let instance_attrs = [
            wgpu::VertexAttribute { offset: 0,  shader_location: 2, format: wgpu::VertexFormat::Float32x4 },
            wgpu::VertexAttribute { offset: 16, shader_location: 3, format: wgpu::VertexFormat::Float32x4 },
            wgpu::VertexAttribute { offset: 32, shader_location: 4, format: wgpu::VertexFormat::Float32x4 },
            wgpu::VertexAttribute { offset: 48, shader_location: 5, format: wgpu::VertexFormat::Float32x4 },
            wgpu::VertexAttribute { offset: 64, shader_location: 6, format: wgpu::VertexFormat::Float32x4 },
            wgpu::VertexAttribute { offset: 80, shader_location: 7, format: wgpu::VertexFormat::Float32x4 },
        ];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sprite_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<QuadVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &vertex_attrs,
                    }),
                    Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<SpriteInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &instance_attrs,
                    }),
                ],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sprite_camera_buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sprite_camera_bg"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // Unit quad centered at origin, Y-up.
        // UVs: u increases right, v increases down (standard texture convention).
        let vertices = [
            QuadVertex { position: [-0.5,  0.5], uv: [0.0, 0.0] },
            QuadVertex { position: [ 0.5,  0.5], uv: [1.0, 0.0] },
            QuadVertex { position: [ 0.5, -0.5], uv: [1.0, 1.0] },
            QuadVertex { position: [-0.5, -0.5], uv: [0.0, 1.0] },
        ];
        let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];

        let quad_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sprite_quad_vb"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let quad_ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sprite_quad_ib"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        self.pipeline = Some(Pipeline {
            pipeline,
            texture_bgl,
            camera_buffer,
            camera_bind_group,
            quad_vb,
            quad_ib,
        });
    }

    fn upload_texture(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, tex: &Arc<SpriteTexture>) {
        let key = Arc::as_ptr(tex) as usize;
        if self.texture_cache.contains_key(&key) {
            return;
        }

        let gpu_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sprite_texture"),
            size: wgpu::Extent3d {
                width: tex.width,
                height: tex.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &gpu_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &tex.data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * tex.width),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: tex.width,
                height: tex.height,
                depth_or_array_layers: 1,
            },
        );

        let view = gpu_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sprite_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let texture_bgl = &self.pipeline.as_ref().unwrap().texture_bgl;
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sprite_texture_bg"),
            layout: texture_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        self.texture_cache.insert(key, GpuTexture { _texture: gpu_tex, _view: view, bind_group });
    }

    fn sprite_model(sprite: &Sprite) -> [[f32; 4]; 4] {
        let cos_r = sprite.rotation.cos();
        let sin_r = sprite.rotation.sin();
        let sx = sprite.scale.x;
        let sy = sprite.scale.y;
        let px = sprite.position.x;
        let py = sprite.position.y;

        // Column-major TRS matrix embedded in 4x4
        [
            [ sx * cos_r, sx * sin_r, 0.0, 0.0], // col 0: scaled X axis
            [-sy * sin_r, sy * cos_r, 0.0, 0.0], // col 1: scaled Y axis
            [0.0,         0.0,        1.0, 0.0], // col 2: Z passthrough
            [px,          py,         0.0, 1.0], // col 3: translation
        ]
    }
}

impl PassDesc for SpritePass {
    fn name(&self) -> &'static str {
        "sprite"
    }

    fn reads(&self) -> Vec<ResourceId> {
        vec![]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![self.surface_id]
    }
}

impl Pass for SpritePass {
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
        self.ensure_pipeline(device);

        let sprites = world.get_entities::<Sprite>();

        // Update camera uniform
        let (width, height) = ctx.surface_size;
        let view_proj = world
            .get_entities::<Camera2D>()
            .into_iter()
            .next()
            .map(|cam| cam.ortho_matrix(width, height))
            .unwrap_or_else(|| Camera2D::default().ortho_matrix(width, height));

        let p = self.pipeline.as_ref().unwrap();
        queue.write_buffer(&p.camera_buffer, 0, bytemuck::bytes_of(&CameraUniform { view_proj }));

        if sprites.is_empty() {
            if let Some(color) = self.clear_color {
                let surface_view = ctx.views[&self.surface_id];
                let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("sprite_clear"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: surface_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(color),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                drop(pass);
            }
            return;
        }

        // Upload any new textures before borrowing encoder
        for sprite in &sprites {
            self.upload_texture(device, queue, &sprite.texture);
        }

        // Sort back-to-front (ascending z_order = painter's algorithm)
        let mut sorted: Vec<&Sprite> = sprites;
        sorted.sort_by(|a, b| {
            a.z_order
                .partial_cmp(&b.z_order)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Group consecutive sprites sharing the same texture into instance ranges
        struct Batch {
            tex_key: usize,
            instances: std::ops::Range<u32>,
        }

        let mut instances: Vec<SpriteInstance> = Vec::with_capacity(sorted.len());
        let mut batches: Vec<Batch> = Vec::new();
        for sprite in &sorted {
            let tex_key = Arc::as_ptr(&sprite.texture) as usize;
            let index = instances.len() as u32;
            instances.push(SpriteInstance {
                model: Self::sprite_model(sprite),
                color: sprite.color,
                uv_rect: sprite.uv_rect,
            });
            match batches.last_mut() {
                Some(b) if b.tex_key == tex_key => b.instances.end = index + 1,
                _ => batches.push(Batch { tex_key, instances: index..index + 1 }),
            }
        }

        let instance_slice = ctx.frame.alloc(
            device,
            queue,
            bytemuck::cast_slice(&instances),
            size_of::<SpriteInstance>() as u64,
        );

        let load = match self.clear_color {
            Some(color) => wgpu::LoadOp::Clear(color),
            None => wgpu::LoadOp::Load,
        };

        let surface_view = ctx.views[&self.surface_id];
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("sprite"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: surface_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });

        let p = self.pipeline.as_ref().unwrap();
        pass.set_pipeline(&p.pipeline);
        pass.set_bind_group(0, &p.camera_bind_group, &[]);
        pass.set_vertex_buffer(0, p.quad_vb.slice(..));
        pass.set_index_buffer(p.quad_ib.slice(..), wgpu::IndexFormat::Uint16);
        pass.set_vertex_buffer(1, instance_slice.range().slice());

        for batch in &batches {
            let gpu_tex = &self.texture_cache[&batch.tex_key];
            pass.set_bind_group(1, &gpu_tex.bind_group, &[]);
            pass.draw_indexed(0..6, 0, batch.instances.clone());
        }
    }
}
