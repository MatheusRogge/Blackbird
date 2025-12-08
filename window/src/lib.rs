use application::{Application, ApplicationError};
use engine::{Engine, input::InputEvent};
use rendering::{
    pipeline::RenderingPipelineDescriptor,
    renderer::{Renderer, RendererError, SurfaceError},
};
use std::sync::Arc;
use thiserror::Error;
use winit::{
    application::ApplicationHandler,
    error::EventLoopError,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    platform::scancode::PhysicalKeyExtScancode,
    window::{Window, WindowAttributes, WindowId},
};

#[derive(Debug, Error)]
#[error(transparent)]
pub struct ApplicationEventLoopError(#[from] EventLoopError);

impl From<ApplicationEventLoopError> for ApplicationError {
    fn from(value: ApplicationEventLoopError) -> Self {
        Self {
            message: value.0.to_string(),
            source: Box::new(value),
        }
    }
}

pub struct WindowedApplication<'a, A> {
    engine: Engine,
    pipeline_descriptor: RenderingPipelineDescriptor<'a>,

    window: Option<Arc<Window>>,
    renderer: Option<Renderer<'a>>,

    inner: A,
}

impl<'a, A> WindowedApplication<'a, A>
where
    A: Application,
{
    pub fn create(
        mut engine: Engine,
        pipeline_descriptor: RenderingPipelineDescriptor<'a>,
    ) -> Result<Self, ApplicationError> {
        let inner = A::setup(&mut engine)?;

        Ok(Self {
            engine,
            pipeline_descriptor,
            renderer: None,
            window: None,
            inner,
        })
    }

    pub async fn execute(&mut self) -> Result<(), ApplicationError> {
        let event_loop = EventLoop::builder()
            .build()
            .map_err(ApplicationEventLoopError)?;

        event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

        event_loop
            .run_app(self)
            .map_err(ApplicationEventLoopError)?;

        Ok(())
    }
}

impl<'a, A> ApplicationHandler for WindowedApplication<'a, A>
where
    A: Application,
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(WindowAttributes::default())
                .unwrap(),
        );

        let renderer_fut = Renderer::new(
            window.clone(),
            window.inner_size().into(),
            &self.pipeline_descriptor,
        );

        self.window = Some(window);
        self.renderer = Some(pollster::block_on(renderer_fut).unwrap());
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let renderer = match self.renderer {
            Some(ref mut renderer) => renderer,
            None => return,
        };

        let window = match self.window {
            Some(ref window) => window,
            None => return,
        };

        match event {
            WindowEvent::KeyboardInput {
                device_id: _,
                event,
                is_synthetic: _,
            } => {
                let key_code = event.physical_key.to_scancode().unwrap().into();

                let event = {
                    match event.state {
                        ElementState::Released => InputEvent::KeyReleased { key_code },
                        ElementState::Pressed => InputEvent::KeyPressed { key_code },
                    }
                };

                self.engine.input().push(event);
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => renderer.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                // Process game logic
                pollster::block_on(self.inner.run(&mut self.engine)).unwrap();

                // Request next frame
                window.request_redraw();

                if let Err(error) = renderer.render(self.engine.world()) {
                    if let RendererError::SurfaceError(
                        SurfaceError::Lost | SurfaceError::Outdated,
                    ) = error
                    {
                        let size = window.inner_size();
                        renderer.resize(size.width, size.height);
                    }

                    println!("Failed to render: {}", error);
                }
            }
            _ => (),
        }
    }
}
