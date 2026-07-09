use std::sync::{Arc, RwLock};

use engine_core::world::World;
use thiserror::Error;
use wgpu::{
    Backends, CreateSurfaceError, Device, DeviceDescriptor, Features, Instance, InstanceDescriptor,
    InstanceFlags, MemoryBudgetThresholds, MemoryHints, PowerPreference, Queue,
    RequestAdapterError, RequestAdapterOptions, RequestDeviceError, Surface, SurfaceConfiguration,
    SurfaceTarget,
};

use crate::{debug_flags::RenderDebugFlags, graph::RenderGraph, profiler::FrameStats, resource::ResourceId};

pub trait OverlayPass: Send + 'static {
    fn execute(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        size: (u32, u32),
        world: &World,
    );
}

pub trait RenderGraphBuilder: Send + 'static {
    /// Build the render graph.
    ///
    /// Returns:
    /// - The compiled `RenderGraph`.
    /// - The `surface_id` that `execute` traces backward from.
    fn build(
        self,
        device: &wgpu::Device,
        surface_config: &SurfaceConfiguration,
    ) -> (RenderGraph, ResourceId);
}

#[derive(Error, Debug)]
pub enum RendererError {
    #[error("Surface not supported")]
    SurfaceNotSupportedError,

    #[error(transparent)]
    RequestAdapterError(#[from] RequestAdapterError),

    #[error(transparent)]
    RequestDeviceError(#[from] RequestDeviceError),

    #[error(transparent)]
    CreateSurfaceError(#[from] CreateSurfaceError),
}

pub struct Renderer {
    surface: Surface<'static>,
    device: Device,
    queue: Queue,

    surface_config: SurfaceConfiguration,
    max_texture_dimension: u32,

    graph: RenderGraph,

    surface_id: ResourceId,
    needs_reconfigure: bool,
    is_surface_configured: bool,
}

impl Renderer {
    pub async fn new<W, B>(window: W, builder: B) -> Result<Self, RendererError>
    where
        W: Into<SurfaceTarget<'static>>,
        B: RenderGraphBuilder,
    {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::VULKAN,
            flags: InstanceFlags::from_env_or_default(),
            memory_budget_thresholds: MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::from_env_or_default(),
            display: None,
        });

        let surface = instance.create_surface(window)?;

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                power_preference: PowerPreference::None,
            })
            .await?;

        let timing_features = Features::TIMESTAMP_QUERY | Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: None,
                trace: wgpu::Trace::Off,
                memory_hints: MemoryHints::MemoryUsage,
                required_features: adapter.features() & timing_features,
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await?;

        let max_texture_dimension = device.limits().max_texture_dimension_2d;

        let mut surface_config = surface
            .get_default_config(&adapter, 800, 600)
            .ok_or(RendererError::SurfaceNotSupportedError)?;

        // Prefer an sRGB surface so the hardware gamma-encodes our linear output.
        // Without this, sRGB-decoded texture values render ~4× too dark on Vulkan.
        let caps = surface.get_capabilities(&adapter);
        if let Some(&srgb_fmt) = caps.formats.iter().find(|f| f.is_srgb()) {
            surface_config.format = srgb_fmt;
            let linear_fmt = srgb_fmt.remove_srgb_suffix();
            if linear_fmt != srgb_fmt {
                surface_config.view_formats = vec![linear_fmt];
            }
        }

        let (graph, surface_id) = builder.build(&device, &surface_config);

        surface.configure(&device, &surface_config);

        Ok(Self {
            device,
            queue,
            surface,
            surface_config,
            max_texture_dimension,
            surface_id,
            is_surface_configured: true,
            needs_reconfigure: false,
            graph,
        })
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.surface_config.format
    }

    pub fn frame_stats(&self) -> &FrameStats {
        self.graph.frame_stats()
    }

    pub fn debug_flags_arc(&self) -> Arc<RwLock<RenderDebugFlags>> {
        self.graph.debug_flags_arc()
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        let max = self.max_texture_dimension;
        self.surface_config.width = width.min(max);
        self.surface_config.height = height.min(max);

        self.needs_reconfigure = true;
        self.is_surface_configured = true;

        self.graph.on_resize(width, height);
    }

    pub fn render(&mut self, world: &World, overlay: Option<&mut dyn OverlayPass>) {
        if !self.is_surface_configured {
            return;
        }

        if self.needs_reconfigure {
            self.surface.configure(&self.device, &self.surface_config);
            self.needs_reconfigure = false;
        }

        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => {
                self.needs_reconfigure = true;
                surface_texture
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.needs_reconfigure = true;
                return;
            }
        };

        let surface_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let cmd = self.graph.execute(
            &self.device,
            &self.queue,
            self.surface_id,
            &surface_view,
            (self.surface_config.width, self.surface_config.height),
            world,
        );

        if let Some(overlay) = overlay {
            let linear_fmt = self.surface_config.format.remove_srgb_suffix();
            let overlay_view = if linear_fmt != self.surface_config.format {
                surface_texture.texture.create_view(&wgpu::TextureViewDescriptor {
                    format: Some(linear_fmt),
                    ..Default::default()
                })
            } else {
                surface_texture.texture.create_view(&wgpu::TextureViewDescriptor::default())
            };

            let mut ui_encoder = self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("ui") },
            );
            overlay.execute(
                &self.device,
                &self.queue,
                &mut ui_encoder,
                &overlay_view,
                (self.surface_config.width, self.surface_config.height),
                world,
            );
            self.queue.submit([cmd, ui_encoder.finish()]);
        } else {
            self.queue.submit(std::iter::once(cmd));
        }

        self.graph.schedule_gpu_readback(&self.device);
        surface_texture.present();
    }
}
