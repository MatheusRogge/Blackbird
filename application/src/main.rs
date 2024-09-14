mod event_loop;
mod platform;

use rendering::renderer::{Renderer, WGPURenderer};

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

#[derive(Debug)]
struct ApplicationError(String);

#[derive(Clone, Copy, Default)]
struct State {}

struct App<'a> {
    window: Option<Window>,
    renderer: WGPURenderer<'a>,
}

impl<'a> App<'a> {
    fn new() -> Result<Self, ApplicationError> {
        Ok(Self {
            renderer: Default::default(),
            window: Default::default(),
        })
    }

    async fn run(&mut self, event_loop: EventLoop<()>) -> Result<(), ApplicationError> {
        event_loop
            .run_app(self)
            .map_err(|error| ApplicationError(error.to_string()))?;

        self.renderer
            .initialize()
            .await
            .map_err(|error| ApplicationError(error.to_string()))?;

        Ok(())
    }
}

impl<'a> ApplicationHandler for App<'a> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(Window::default_attributes())
            .unwrap();

        self.renderer.attach_window(&window).unwrap();
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                // Redraw the application.
                //
                // It's preferable for applications that do not render continuously to render in
                // this event rather than in AboutToWait, since rendering in here allows
                // the program to gracefully handle redraws requested by the OS.

                // Draw.

                // Queue a RedrawRequested event.
                //
                // You only need to call this if you've determined that you need to redraw in
                // applications which do not always need to. Applications that redraw continuously
                // can render here instead.
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => (),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new().unwrap();
    app.run(event_loop)
        .await
        .expect("Failed to run application");

    Ok(())
}
