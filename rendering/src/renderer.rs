use engine::world::World;
use thiserror::Error;
use wgpu::{
    Backends, CreateSurfaceError, Device, DeviceDescriptor, Features, Instance, InstanceDescriptor,
    InstanceFlags, Limits, MemoryBudgetThresholds, MemoryHints, PowerPreference, Queue,
    RequestAdapterError, RequestAdapterOptions, RequestDeviceError, Surface, SurfaceConfiguration,
    SurfaceTarget,
};

use crate::{graph::RenderGraph, resource::ResourceId};

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

        let surface_config = surface
            .get_default_config(&adapter, 800, 600)
            .ok_or(RendererError::SurfaceNotSupportedError)?;

        let (graph, surface_id) = builder.build(&device, &surface_config);

        surface.configure(&device, &surface_config);

        Ok(Self {
            device,
            queue,
            surface,
            surface_config,
            surface_id,
            is_surface_configured: true,
            needs_reconfigure: false,
            graph,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        self.surface_config.width = width;
        self.surface_config.height = height;

        self.needs_reconfigure = true;
        self.is_surface_configured = true;

        self.graph.on_resize(width, height);
    }

    pub fn render(&mut self, world: &World) {
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

        let cmd = self.graph
            .execute(&self.device, &self.queue, self.surface_id, &surface_view, world);

        self.queue.submit(std::iter::once(cmd));
        surface_texture.present();
    }
}
