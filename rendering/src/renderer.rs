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

    #[error("only a headless renderer can read its frame back")]
    NotHeadless,

    #[error(transparent)]
    Poll(#[from] wgpu::PollError),

    #[error(transparent)]
    Readback(#[from] wgpu::BufferAsyncError),

    #[error(transparent)]
    MapRange(#[from] wgpu::MapRangeError),
}

/// A rendered frame read back from a headless renderer: `width × height` RGBA8 texels, sRGB
/// encoded, rows top to bottom.
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Format of the headless target, matching the sRGB surface a window gets.
const HEADLESS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

enum Target {
    Surface {
        surface: Surface<'static>,
        needs_reconfigure: bool,
    },
    Offscreen {
        texture: wgpu::Texture,
    },
}

/// Draws a `RenderGraph` into a window surface ([`Renderer::new`]) or an offscreen texture
/// ([`Renderer::new_headless`]). Both share device setup, the graph and the overlay path.
pub struct Renderer {
    device: Device,
    queue: Queue,

    target: Target,
    /// The window surface's configuration, or the offscreen texture's size and format.
    config: SurfaceConfiguration,
    max_texture_dimension: u32,

    graph: RenderGraph,

    surface_id: ResourceId,
}

fn vulkan_instance() -> Instance {
    Instance::new(InstanceDescriptor {
        backends: Backends::VULKAN,
        flags: InstanceFlags::from_env_or_default(),
        memory_budget_thresholds: MemoryBudgetThresholds::default(),
        backend_options: wgpu::BackendOptions::from_env_or_default(),
        display: None,
    })
}

async fn request_device(
    instance: &Instance,
    compatible_surface: Option<&Surface<'_>>,
) -> Result<(wgpu::Adapter, Device, Queue), RendererError> {
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            compatible_surface,
            force_fallback_adapter: false,
            power_preference: PowerPreference::None,
            apply_limit_buckets: false,
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
    Ok((adapter, device, queue))
}

fn create_offscreen(device: &Device, config: &SurfaceConfiguration) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("headless_target"),
        size: wgpu::Extent3d { width: config.width, height: config.height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: config.format,
        usage: config.usage,
        view_formats: &config.view_formats,
    })
}

impl Renderer {
    pub async fn new<W, B>(window: W, builder: B) -> Result<Self, RendererError>
    where
        W: Into<SurfaceTarget<'static>>,
        B: RenderGraphBuilder,
    {
        let instance = vulkan_instance();
        let surface = instance.create_surface(window)?;
        let (adapter, device, queue) = request_device(&instance, Some(&surface)).await?;

        let mut config = surface
            .get_default_config(&adapter, 800, 600)
            .ok_or(RendererError::SurfaceNotSupportedError)?;

        // Prefer an sRGB surface so the hardware gamma-encodes our linear output.
        // Without this, sRGB-decoded texture values render ~4× too dark on Vulkan.
        let caps = surface.get_capabilities(&adapter);
        if let Some(&srgb_fmt) = caps.formats.iter().find(|f| f.is_srgb()) {
            config.format = srgb_fmt;
            let linear_fmt = srgb_fmt.remove_srgb_suffix();
            if linear_fmt != srgb_fmt {
                config.view_formats = vec![linear_fmt];
            }
        }

        let (graph, surface_id) = builder.build(&device, &config);
        surface.configure(&device, &config);

        Ok(Self::assemble(device, queue, Target::Surface { surface, needs_reconfigure: false }, config, graph, surface_id))
    }

    /// Renders into an offscreen `width × height` sRGB texture instead of a window. Each
    /// `render` waits for the GPU, and `read_frame` returns the result.
    pub async fn new_headless<B: RenderGraphBuilder>(builder: B, width: u32, height: u32) -> Result<Self, RendererError> {
        let instance = vulkan_instance();
        let (_, device, queue) = request_device(&instance, None).await?;

        let max = device.limits().max_texture_dimension_2d;
        let config = SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format: HEADLESS_FORMAT,
            width: width.clamp(1, max),
            height: height.clamp(1, max),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 1,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![HEADLESS_FORMAT.remove_srgb_suffix()],
            color_space: Default::default(),
        };

        let (graph, surface_id) = builder.build(&device, &config);
        let texture = create_offscreen(&device, &config);
        Ok(Self::assemble(device, queue, Target::Offscreen { texture }, config, graph, surface_id))
    }

    fn assemble(
        device: Device,
        queue: Queue,
        target: Target,
        config: SurfaceConfiguration,
        graph: RenderGraph,
        surface_id: ResourceId,
    ) -> Self {
        Self {
            max_texture_dimension: device.limits().max_texture_dimension_2d,
            device,
            queue,
            target,
            config,
            graph,
            surface_id,
        }
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
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
        self.config.width = width.min(max);
        self.config.height = height.min(max);

        match &mut self.target {
            Target::Surface { needs_reconfigure, .. } => *needs_reconfigure = true,
            Target::Offscreen { texture } => *texture = create_offscreen(&self.device, &self.config),
        }

        self.graph.on_resize(width, height);
    }

    pub fn render(&mut self, world: &World, overlay: Option<&mut dyn OverlayPass>) {
        let (texture, surface_texture) = match &mut self.target {
            Target::Surface { surface, needs_reconfigure } => {
                if *needs_reconfigure {
                    surface.configure(&self.device, &self.config);
                    *needs_reconfigure = false;
                }
                let surface_texture = match surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
                    wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => {
                        *needs_reconfigure = true;
                        surface_texture
                    }
                    wgpu::CurrentSurfaceTexture::Timeout
                    | wgpu::CurrentSurfaceTexture::Occluded
                    | wgpu::CurrentSurfaceTexture::Validation => return,
                    wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                        *needs_reconfigure = true;
                        return;
                    }
                };
                (surface_texture.texture.clone(), Some(surface_texture))
            }
            Target::Offscreen { texture } => (texture.clone(), None),
        };

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let size = (self.config.width, self.config.height);

        let cmd = self.graph.execute(&self.device, &self.queue, self.surface_id, &view, size, world);

        if let Some(overlay) = overlay {
            let linear_fmt = self.config.format.remove_srgb_suffix();
            let overlay_view = if linear_fmt != self.config.format {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    format: Some(linear_fmt),
                    ..Default::default()
                })
            } else {
                texture.create_view(&wgpu::TextureViewDescriptor::default())
            };

            let mut ui_encoder = self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("ui") },
            );
            overlay.execute(&self.device, &self.queue, &mut ui_encoder, &overlay_view, size, world);
            self.queue.submit([cmd, ui_encoder.finish()]);
        } else {
            self.queue.submit(std::iter::once(cmd));
        }

        self.graph.schedule_gpu_readback(&self.device);
        match surface_texture {
            Some(surface_texture) => self.queue.present(surface_texture),
            // Nothing paces offscreen frames, so keep at most one in flight.
            None => {
                if let Err(e) = self.device.poll(wgpu::PollType::wait_indefinitely()) {
                    log::error!("waiting for the headless frame failed: {e}");
                }
            }
        }
    }

    /// Copies the last rendered frame back from a headless renderer.
    pub fn read_frame(&self) -> Result<Frame, RendererError> {
        let Target::Offscreen { texture } = &self.target else {
            return Err(RendererError::NotHeadless);
        };
        let (width, height) = (self.config.width, self.config.height);
        let row_bytes = width * 4;
        let padded_row = row_bytes.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("headless_readback"),
            size: padded_row as u64 * height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        self.queue.submit([encoder.finish()]);

        let (tx, rx) = std::sync::mpsc::channel();
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv().expect("map_async callback dropped")?;

        let mapped = buffer.slice(..).get_mapped_range()?;
        let rgba = mapped
            .chunks_exact(padded_row as usize)
            .flat_map(|row| &row[..row_bytes as usize])
            .copied()
            .collect();
        Ok(Frame { width, height, rgba })
    }
}
