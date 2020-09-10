use winit::dpi::Size;
use winit::dpi::LogicalSize;
use winit::window::WindowBuilder;
use winit::event_loop::EventLoop;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;    

pub struct Application {
    pub event_loop: EventLoop<()>
}

impl Application {
    pub fn initialize() -> Self {
        let event_loop = Self::init_window();

        Self {
            event_loop: event_loop
        }
    }

    fn init_window() -> EventLoop<()> {
        let logical_size = LogicalSize {
            width: f64::from(WIDTH),
            height: f64::from(HEIGHT)
        };

        let event_loop = EventLoop::new();

        let _window = WindowBuilder::new()
            .with_title("Vulkan")
            
            .with_inner_size(Size::from(logical_size))
            .build(&event_loop);

        event_loop
    }
}