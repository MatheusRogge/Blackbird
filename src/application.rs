use std::sync::Arc;

use winit::{
    event::{Event,WindowEvent},
    event_loop::{ControlFlow,EventLoop},
    window::{Window,WindowBuilder},
};

use vulkano::{
    instance::{ApplicationInfo,Instance,Version}
};


#[derive(Debug)]
pub struct Application {
    event_loop: EventLoop<()>,
    instance: Arc<Instance>,
    window: Window,
}

impl Application {
    pub fn initialize() -> Self {
        let event_loop = EventLoop::new();
        let instance = Self::create_instance();

        let window = WindowBuilder::new()
            .with_title("My Application")
            .build(&event_loop)
            .unwrap();

        Self {
            event_loop,
            instance,
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

    fn create_instance() -> Arc<Instance> {
        let required_extensions = vulkano_win::required_extensions();

        let app_info = ApplicationInfo {
            application_name: Some("My Application".into()),
            application_version: Some(Version { major: 1, minor: 0, patch: 0 }),
            engine_name: Some("Blackbird".into()),
            engine_version: Some(Version { major: 1, minor: 0, patch: 0 }),
        };

        Instance::new(Some(&app_info), &required_extensions, None).unwrap()
    }
}