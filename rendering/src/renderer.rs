use crate::{
    camera::{Camera, CameraUniform},
    mesh::Mesh,
    pipeline::{RenderingPipelineDescriptor, RenderingPipelineError},
};
use bytemuck::PodCastError;
use engine::world::World;
use thiserror::Error;
pub use wgpu::SurfaceError;
use wgpu::{
    Backends, BindGroup, Buffer, BufferDescriptor, CreateSurfaceError, Device, DeviceDescriptor,
    Features, Instance, InstanceDescriptor, Limits, MemoryHints, PowerPreference, Queue,
    RenderPipeline, RequestAdapterError, RequestAdapterOptionsBase, RequestDeviceError, Surface,
    SurfaceConfiguration, SurfaceTarget,
};

pub struct Renderer<'a> {
    device: Device,
    surface: Surface<'a>,
    surface_config: SurfaceConfiguration,
    pipeline: RenderPipeline,
    queue: Queue,
    is_surface_configured: bool,
    camera_buffer: Buffer,
    camera_bind_group: BindGroup,
}

#[derive(Error, Debug)]
pub enum RendererError {
    #[error("Surface Initialization error")]
    SurfaceInitializationError,

    #[error(transparent)]
    AdapterError(#[from] RequestAdapterError),

    #[error(transparent)]
    DeviceError(#[from] RequestDeviceError),

    #[error(transparent)]
    CreateSurfaceError(#[from] CreateSurfaceError),

    #[error(transparent)]
    PipelineError(#[from] RenderingPipelineError),

    #[error(transparent)]
    SurfaceError(#[from] SurfaceError),

    #[error("Failed to cast buffer")]
    PodCastError(PodCastError),

    #[error("No camera error")]
    NoCamera,
}

impl<'a> Renderer<'a> {
    pub async fn new<W>(
        window: W,
        initial_size: (u32, u32),
        pipeline_descriptor: &RenderingPipelineDescriptor<'a>,
    ) -> Result<Self, RendererError>
    where
        W: Into<SurfaceTarget<'a>>,
    {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::VULKAN,
            ..Default::default()
        });

        let surface = instance.create_surface(window)?;

        let adapter = instance
            .request_adapter(&RequestAdapterOptionsBase {
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                power_preference: PowerPreference::None,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: None,
                trace: wgpu::Trace::Off,
                memory_hints: MemoryHints::MemoryUsage,
                required_features: Features::empty(),
                required_limits: Limits::downlevel_webgl2_defaults(),
                ..Default::default()
            })
            .await?;

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                label: Some("camera_bind_group_layout"),
            });

        let camera_buffer = {
            device.create_buffer(&BufferDescriptor {
                mapped_at_creation: false,
                label: Some("Camera Buffer"),
                size: std::mem::size_of::<CameraUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            label: Some("camera_bind_group"),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let surface_config = surface
            .get_default_config(&adapter, initial_size.0, initial_size.1)
            .ok_or(RendererError::SurfaceInitializationError)?;

        let pipeline = pipeline_descriptor.create_pipeline(
            &device,
            &surface_config,
            &[camera_bind_group_layout],
        )?;

        Ok(Self {
            device,
            surface,
            surface_config,
            queue,
            pipeline,
            is_surface_configured: false,
            camera_buffer,
            camera_bind_group,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        self.surface_config.width = width;
        self.surface_config.height = height;

        self.surface.configure(&self.device, &self.surface_config);
        self.is_surface_configured = true;
    }

    pub fn render(&self, world: &World) -> Result<(), RendererError> {
        if !self.is_surface_configured {
            return Ok(());
        }

        let output = self.surface.get_current_texture()?;

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let cameras = world.get_entities::<Camera>();
        let main_camera = cameras.first().ok_or(RendererError::NoCamera)?;

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                timestamp_writes: None,
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        store: wgpu::StoreOp::Store,
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                    },
                })],
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);

            let meshes = world.get_entities::<Mesh>();
            for mesh in meshes {
                let indices_count = mesh.get_indices_count() as u32;
                let index_buffer = mesh.get_index_buffer(&self.device);
                let vertex_buffer = mesh.get_vertex_buffer(&self.device);

                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..indices_count, 0, 0..1);
            }
        };

        let camera_uniform = main_camera.get_camera_uniform();

        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[camera_uniform]),
        );

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
