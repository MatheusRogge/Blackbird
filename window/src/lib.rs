use std::sync::Arc;

use application::{Application, ApplicationError};
use engine::{Engine, input::InputEvent};
use rendering::renderer::{RenderGraphBuilder, Renderer};
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

pub enum WindowApplicationState<B> {
    Uninitialized(Option<B>),
    Initialized {
        window: Arc<Window>,
        renderer: Box<Renderer>,
    },
}

pub struct WindowedApplication<A: Application, B: RenderGraphBuilder> {
    pub inner: A,
    engine: Engine,
    state: WindowApplicationState<B>,
}

impl<A, B> WindowedApplication<A, B>
where
    A: Application,
    B: RenderGraphBuilder,
{
    pub fn create(mut engine: Engine, builder: B) -> Result<Self, ApplicationError> {
        let inner = A::setup(&mut engine)?;

        Ok(Self {
            inner,
            engine,
            state: WindowApplicationState::Uninitialized(Some(builder)),
        })
    }

    pub async fn run(&mut self) -> Result<(), ApplicationError> {
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

impl<A, B> ApplicationHandler for WindowedApplication<A, B>
where
    A: Application,
    B: RenderGraphBuilder,
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let WindowApplicationState::Uninitialized(ref mut graph_builder) = self.state else {
            return;
        };

        let Some(builder) = graph_builder.take() else {
            return;
        };

        let window = Arc::new(
            event_loop
                .create_window(WindowAttributes::default())
                .unwrap(),
        );

        let renderer = pollster::block_on(Renderer::new(window.clone(), builder)).unwrap();

        self.state = WindowApplicationState::Initialized {
            window,
            renderer: Box::new(renderer),
        };
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let WindowApplicationState::Initialized {
            ref window,
            ref mut renderer,
        } = self.state
        else {
            return;
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
                        ElementState::Pressed => InputEvent::KeyPressed { key_code },
                        ElementState::Released => InputEvent::KeyReleased { key_code },
                    }
                };

                self.engine.input().push(event);
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => renderer.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                pollster::block_on(self.inner.run(&mut self.engine)).unwrap();
                window.request_redraw();
                renderer.render(self.engine.world());
            }
            _ => (),
        }
    }
}
