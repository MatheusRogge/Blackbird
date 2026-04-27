use std::mem;
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Instant;

use engine_core::input::{InputEvent, MouseButton};
use rendering::renderer::{RenderGraphBuilder, Renderer};
use winit::{
    application::ApplicationHandler,
    error::EventLoopError,
    event::{ElementState, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    platform::scancode::PhysicalKeyExtScancode,
    window::{Window, WindowAttributes, WindowId},
};

use crate::application::{ApplicationError, HeadlessApplication};
use crate::plugin::Plugin;

enum GameMsg {
    Tick { inputs: Vec<InputEvent> },
    Shutdown,
}

enum ResizeMsg {
    Resized(u32, u32),
}

pub struct WindowedApplication<B: RenderGraphBuilder> {
    inner: HeadlessApplication,
    graph_builder: Option<B>,

    pending_inputs: Vec<InputEvent>,
    game_tx: Option<mpsc::SyncSender<GameMsg>>,
    resize_tx: Option<mpsc::SyncSender<ResizeMsg>>,
    window: Option<Arc<Window>>,
    game_thread: Option<thread::JoinHandle<()>>,
    render_thread: Option<thread::JoinHandle<()>>,
}

impl<B: RenderGraphBuilder> WindowedApplication<B> {
    /// Creates a windowed application pre-loaded with AssetsPlugin, GltfPlugin.
    pub fn new(render_graph: B) -> Self {
        use crate::plugins::assets::AssetsPlugin;
        #[cfg(feature = "gltf")]
        use crate::plugins::gltf::GltfPlugin;

        let mut inner = HeadlessApplication::new().add_plugin(AssetsPlugin);

        #[cfg(feature = "gltf")]
        {
            inner = inner.add_plugin(GltfPlugin);
        }

        Self {
            inner,
            graph_builder: Some(render_graph),
            pending_inputs: Vec::new(),
            game_tx: None,
            resize_tx: None,
            window: None,
            game_thread: None,
            render_thread: None,
        }
    }

    pub fn add_plugin(mut self, plugin: impl Plugin) -> Self {
        self.inner = self.inner.add_plugin(plugin);
        self
    }

    pub fn run(mut self) -> Result<(), ApplicationError> {
        let event_loop =
            EventLoop::builder()
                .build()
                .map_err(|e: EventLoopError| ApplicationError {
                    message: e.to_string(),
                    source: Some(Box::new(e)),
                })?;

        event_loop.set_control_flow(ControlFlow::Poll);

        event_loop
            .run_app(&mut self)
            .map_err(|e| ApplicationError {
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        Ok(())
    }
}

impl<B: RenderGraphBuilder> ApplicationHandler for WindowedApplication<B> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.game_tx.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(WindowAttributes::default())
                .unwrap(),
        );

        let builder = self.graph_builder.take().unwrap();
        let renderer = pollster::block_on(Renderer::new(window.clone(), builder)).unwrap();

        let world_arc = self.inner.engine.world_arc();
        let frame_stats_arc = self.inner.engine.frame_stats_arc();
        let mut plugins = std::mem::take(&mut self.inner.plugins);
        let mut engine = std::mem::take(&mut self.inner.engine);

        let (game_tx, game_rx) = mpsc::sync_channel::<GameMsg>(1);
        let (render_tx, render_rx) = mpsc::sync_channel::<()>(1);
        let (resize_tx, resize_rx) = mpsc::sync_channel::<ResizeMsg>(4);

        let window_clone = Arc::clone(&window);
        let render_thread = thread::Builder::new()
            .name("render".into())
            .spawn(move || {
                let mut renderer = renderer;

                loop {
                    while let Ok(ResizeMsg::Resized(w, h)) = resize_rx.try_recv() {
                        renderer.resize(w, h);
                    }

                    match render_rx.recv() {
                        Ok(()) => {
                            let world = world_arc.read().expect("world RwLock poisoned");
                            renderer.render(&world);
                            if let Ok(mut s) = frame_stats_arc.write() {
                                *s = renderer.frame_stats().clone();
                            }
                            window_clone.request_redraw();
                        }
                        Err(_) => break,
                    }
                }
            })
            .unwrap();

        let game_thread = thread::Builder::new()
            .name("game".into())
            .spawn(move || {
                let mut last_tick = Instant::now();

                while let Ok(GameMsg::Tick { inputs }) = game_rx.recv() {
                    let delta = last_tick.elapsed().as_secs_f32();
                    last_tick = Instant::now();

                    for event in inputs {
                        engine.input().push(event);
                    }

                    {
                        let mut world = engine.world();
                        world.tick_all(delta);
                    }

                    for plugin in &mut plugins {
                        plugin.tick(&mut engine, delta);
                    }

                    let _ = render_tx.send(());
                }
            })
            .unwrap();

        window.request_redraw();

        self.game_tx = Some(game_tx);
        self.resize_tx = Some(resize_tx);
        self.window = Some(window);
        self.game_thread = Some(game_thread);
        self.render_thread = Some(render_thread);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                let key_code = event.physical_key.to_scancode().unwrap().into();
                let input_event = match event.state {
                    ElementState::Pressed => InputEvent::KeyPressed { key_code },
                    ElementState::Released => InputEvent::KeyReleased { key_code },
                };
                self.pending_inputs.push(input_event);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let btn = match button {
                    winit::event::MouseButton::Left => MouseButton::Left,
                    winit::event::MouseButton::Right => MouseButton::Right,
                    winit::event::MouseButton::Middle => MouseButton::Middle,
                    _ => MouseButton::Other,
                };
                let input_event = match state {
                    ElementState::Pressed => InputEvent::MouseButtonPressed { button: btn },
                    ElementState::Released => InputEvent::MouseButtonReleased { button: btn },
                };
                self.pending_inputs.push(input_event);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_x, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => (pos.y as f32) / 20.0,
                };
                self.pending_inputs
                    .push(InputEvent::MouseScrolled { delta: scroll });
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.pending_inputs.push(InputEvent::MouseMoved {
                    x: position.x as f32,
                    y: position.y as f32,
                });
            }
            WindowEvent::Resized(size) => {
                if let Some(tx) = &self.resize_tx {
                    let _ = tx.try_send(ResizeMsg::Resized(size.width, size.height));
                }
            }
            WindowEvent::CloseRequested => {
                if let Some(tx) = self.game_tx.take() {
                    let _ = tx.send(GameMsg::Shutdown);
                }
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let Some(tx) = &self.game_tx {
                    let inputs = mem::take(&mut self.pending_inputs);
                    let _ = tx.try_send(GameMsg::Tick { inputs });
                }
            }
            _ => {}
        }
    }
}
