mod application;

use crate::application::Application;
use winit::event_loop::{ControlFlow};
use winit::event::{Event,WindowEvent};

fn main() {
    let application = Application::initialize();    

    loop {
        application.event_loop.run(|event, _, control_flow| {
            match event {
                Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                    *control_flow = ControlFlow::Exit;
                },
                _ => ()
            }
        });
    }
}