use winit::window::Window;
use winit::window::WindowBuilder;
use winit::event_loop::EventLoop;
use winit::event_loop::{ControlFlow};
use winit::event::{Event,WindowEvent};

#[derive(Debug)]
pub struct Application {
    pub event_loop: EventLoop<()>,
    pub window: Window
}

impl Application {
    pub fn initialize() -> Self {
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new().build(&event_loop).unwrap();

        Self {
            event_loop,
            window
        }
    }

    pub fn main_loop(self) {
        let event_loop = self.event_loop;
        let current_window_id = self.window.id();

        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Wait;
    
            match event {
                Event::WindowEvent { event: WindowEvent::CloseRequested, window_id } => {
                    if window_id == current_window_id {
                        *control_flow = ControlFlow::Exit
                    }
                },
                _ => (),
            }
        });
    }
}