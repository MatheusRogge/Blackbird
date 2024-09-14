use core::fmt;
use std::{fmt::Debug, sync::Arc};

use wgpu::{Adapter, Instance, RequestAdapterOptions, Surface, SurfaceTargetUnsafe};

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

pub struct WGPURenderer<'a> {
    instance: Instance,
    adapter: Option<Arc<Adapter>>,
    surface: Option<Arc<Surface<'a>>>,
}

#[derive(Debug, Clone)]
pub enum RendererError {
    SurfaceInitialization,
}

impl fmt::Display for RendererError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub trait Renderer<'a, T> {
    fn attach_window<Window>(&mut self, window: &Window) -> Result<(), RendererError>
    where
        Window: HasDisplayHandle + HasWindowHandle;

    fn initialize(&mut self) -> impl std::future::Future<Output = Result<(), RendererError>>;
    fn render(&mut self) -> Result<(), RendererError>;
}

impl<'a> Default for WGPURenderer<'a> {
    fn default() -> Self {
        let instance = wgpu::Instance::default();

        Self {
            instance,
            adapter: None,
            surface: None,
        }
    }
}

impl<'a> Renderer<'a, WGPURenderer<'a>> for WGPURenderer<'a> {
    fn attach_window<Window>(&mut self, window: &Window) -> Result<(), RendererError>
    where
        Window: HasDisplayHandle + HasWindowHandle,
    {
        let target = unsafe {
            SurfaceTargetUnsafe::from_window(window).expect("Failed to retrive target surface")
        };

        let surface = unsafe {
            self.instance
                .create_surface_unsafe(target)
                .expect("Failed to create window surface")
        };

        self.surface = Some(Arc::new(surface));
        Ok(())
    }

    async fn initialize(&mut self) -> Result<(), RendererError> {
        if self.surface.is_none() {
            return Err(RendererError::SurfaceInitialization);
        }

        let adapter_options = RequestAdapterOptions {
            compatible_surface: Some(self.surface.as_ref().unwrap()),
            ..Default::default()
        };

        let adapter = self
            .instance
            .request_adapter(&adapter_options)
            .await
            .expect("Failed to find an appropriate adapter");

        self.adapter = Some(Arc::new(adapter));
        Ok(())
    }

    fn render(&mut self) -> Result<(), RendererError> {
        Ok(())
    }
}
