use std::sync::Arc;

use winit::{
    event::{Event,WindowEvent},
    event_loop::{ControlFlow,EventLoop},
    window::{Window,WindowBuilder},
};

use vulkano::{
    instance::{ApplicationInfo,Instance,InstanceExtensions,Version,layers_list},
    instance::debug::{DebugCallback,MessageSeverity,MessageType}
};

#[cfg(all(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = true;

#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;

const VALIDATION_LAYERS: &'static [&'static str] = &["VK_LAYER_LUNARG_standard_validation"];

pub struct Application {
    debug_callback: Option<DebugCallback>,
    event_loop: EventLoop<()>,
    instance: Arc<Instance>,
    window: Window,
}

impl Application {
    pub fn initialize() -> Self {
        let event_loop = EventLoop::new();
        let instance = Self::create_instance();
        let debug_callback = Self::setup_debug_callback(&instance);

        let window = WindowBuilder::new()
            .with_title("My Application")
            .build(&event_loop)
            .unwrap();

        Self {
            debug_callback,
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

    fn setup_debug_callback(instance: &Arc<Instance>) -> Option<DebugCallback> {
        if !ENABLE_VALIDATION_LAYERS  {
            return None;
        }

        let msg_types = MessageType::all();
        let message_severity = MessageSeverity::errors_and_warnings();

        let debug_callback = DebugCallback::new(&instance, message_severity, msg_types, move |msg| {
            println!("Debug callback: {:?}", msg.description);
        });

        debug_callback.ok()
    }

    fn check_validation_layer_support() -> bool {
        let available_validation_layers: Vec<_> = layers_list().unwrap().map(|l| l.name().to_owned()).collect();

        VALIDATION_LAYERS
            .iter()
            .all(|layer_name| available_validation_layers.contains(&layer_name.to_string()))
    }

    fn get_required_extensions() -> InstanceExtensions {
        let mut extensions = vulkano_win::required_extensions();

        if ENABLE_VALIDATION_LAYERS {
            extensions.ext_debug_utils = true;
        }

        extensions
    }

    fn create_instance() -> Arc<Instance> {
        let required_extensions = Self::get_required_extensions();

        let app_info = ApplicationInfo {
            application_name: Some("My Application".into()),
            application_version: Some(Version { major: 1, minor: 0, patch: 0 }),
            engine_name: Some("Blackbird".into()),
            engine_version: Some(Version { major: 1, minor: 0, patch: 0 }),
        };

        if ENABLE_VALIDATION_LAYERS && Self::check_validation_layer_support() {
            Instance::new(Some(&app_info), &required_extensions, VALIDATION_LAYERS.iter().cloned()).unwrap()
        } 
        else {
            Instance::new(Some(&app_info), &required_extensions, None).unwrap()
        }
    }
}