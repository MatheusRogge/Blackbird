use std::mem;
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Instant;

use engine_core::input::{InputEvent, Key, Modifiers, MouseButton};
use rendering::renderer::{RenderGraphBuilder, Renderer};
use winit::{
    application::ApplicationHandler,
    error::EventLoopError,
    event::{ElementState, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::NamedKey,
    platform::scancode::PhysicalKeyExtScancode,
    window::{Window, WindowAttributes, WindowId},
};

use crate::application::{ApplicationError, HeadlessApplication};
use crate::plugin::Plugin;

#[cfg(feature = "ui")]
use ui::{EguiRenderer, UiLayer};

#[cfg(feature = "ui")]
struct EguiOverlay(EguiRenderer);

#[cfg(feature = "ui")]
impl rendering::renderer::OverlayPass for EguiOverlay {
    fn execute(
        &mut self,
        device: &rendering::wgpu::Device,
        queue: &rendering::wgpu::Queue,
        encoder: &mut rendering::wgpu::CommandEncoder,
        view: &rendering::wgpu::TextureView,
        size: (u32, u32),
        world: &engine_core::world::World,
    ) {
        self.0.execute(device, queue, encoder, view, size, world);
    }
}

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
    current_modifiers: Modifiers,
    game_tx: Option<mpsc::SyncSender<GameMsg>>,
    resize_tx: Option<mpsc::SyncSender<ResizeMsg>>,
    window: Option<Arc<Window>>,
    window_title: Option<String>,
    game_thread: Option<thread::JoinHandle<()>>,
    render_thread: Option<thread::JoinHandle<()>>,

    #[cfg(feature = "ui")]
    pending_ui_layers: Vec<Box<dyn UiLayer>>,
}

impl<B: RenderGraphBuilder> WindowedApplication<B> {
    pub fn new(render_graph: B) -> Self {
        Self {
            inner: HeadlessApplication::new(),
            graph_builder: Some(render_graph),
            pending_inputs: Vec::new(),
            current_modifiers: Modifiers::default(),
            game_tx: None,
            resize_tx: None,
            window: None,
            window_title: None,
            game_thread: None,
            render_thread: None,
            #[cfg(feature = "ui")]
            pending_ui_layers: Vec::new(),
        }
    }

    pub fn add_plugin(mut self, plugin: impl Plugin) -> Self {
        self.inner = self.inner.add_plugin(plugin);
        self
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.window_title = Some(title.into());
        self
    }

    #[cfg(feature = "ui")]
    pub fn add_ui_layer(mut self, layer: impl UiLayer) -> Self {
        self.pending_ui_layers.push(Box::new(layer));
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

fn translate_named_key(named: NamedKey, scancode: u32) -> Key {
    match named {
        NamedKey::ArrowUp => Key::ArrowUp,
        NamedKey::ArrowDown => Key::ArrowDown,
        NamedKey::ArrowLeft => Key::ArrowLeft,
        NamedKey::ArrowRight => Key::ArrowRight,
        NamedKey::Enter => Key::Enter,
        NamedKey::Escape => Key::Escape,
        NamedKey::Tab => Key::Tab,
        NamedKey::Space => Key::Space,
        NamedKey::Backspace => Key::Backspace,
        NamedKey::Delete => Key::Delete,
        NamedKey::Insert => Key::Insert,
        NamedKey::Home => Key::Home,
        NamedKey::End => Key::End,
        NamedKey::PageUp => Key::PageUp,
        NamedKey::PageDown => Key::PageDown,
        NamedKey::F1 => Key::F1,
        NamedKey::F2 => Key::F2,
        NamedKey::F3 => Key::F3,
        NamedKey::F4 => Key::F4,
        NamedKey::F5 => Key::F5,
        NamedKey::F6 => Key::F6,
        NamedKey::F7 => Key::F7,
        NamedKey::F8 => Key::F8,
        NamedKey::F9 => Key::F9,
        NamedKey::F10 => Key::F10,
        NamedKey::F11 => Key::F11,
        NamedKey::F12 => Key::F12,
        _ => Key::Scancode(scancode),
    }
}

impl<B: RenderGraphBuilder> ApplicationHandler for WindowedApplication<B> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.game_tx.is_some() {
            return;
        }

        let mut attributes = WindowAttributes::default();
        if let Some(title) = self.window_title.take() {
            attributes = attributes.with_title(title);
        }
        let window = Arc::new(event_loop.create_window(attributes).unwrap());

        let builder = self.graph_builder.take().unwrap();
        let renderer = pollster::block_on(Renderer::new(window.clone(), builder)).unwrap();

        let world_arc = self.inner.engine.world_arc();
        let frame_stats_arc = self.inner.engine.frame_stats_arc();
        let mut plugins = std::mem::take(&mut self.inner.plugins);
        let mut engine = std::mem::take(&mut self.inner.engine);

        #[cfg(feature = "ui")]
        let mut egui_overlay: Option<EguiOverlay> = {
            let mut layers = mem::take(&mut self.pending_ui_layers);

            #[cfg(feature = "debug")]
            layers.insert(0, Box::new(rendering::DebugLayer::new(
                Arc::clone(&frame_stats_arc),
                renderer.debug_flags_arc(),
            )));

            if layers.is_empty() {
                None
            } else {
                let (wants_ptr, wants_kbd) = engine.ui_flag_arcs();
                Some(EguiOverlay(EguiRenderer::new(
                    renderer.device(),
                    renderer.surface_format().remove_srgb_suffix(),
                    layers,
                    wants_ptr,
                    wants_kbd,
                )))
            }
        };

        let (game_tx, game_rx) = mpsc::sync_channel::<GameMsg>(1);
        // Render channel carries the frame's input events so the render thread
        // can forward them to UI systems (e.g. egui) without a separate channel.
        // The game thread sends on this after finishing its tick, preserving the
        // original game→render ordering guarantee.
        let (render_tx, render_rx) = mpsc::sync_channel::<Vec<InputEvent>>(1);
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
                        Ok(events) => {
                            #[cfg(feature = "ui")]
                            if let Some(overlay) = &mut egui_overlay {
                                overlay.0.on_events(&events);
                            }
                            #[cfg(not(feature = "ui"))]
                            drop(events);

                            let world = world_arc.read().expect("world RwLock poisoned");

                            #[cfg(feature = "ui")]
                            let overlay = egui_overlay
                                .as_mut()
                                .map(|e| e as &mut dyn rendering::renderer::OverlayPass);
                            #[cfg(not(feature = "ui"))]
                            let overlay: Option<&mut dyn rendering::renderer::OverlayPass> = None;

                            renderer.render(&world, overlay);

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

                    let render_events = inputs.clone();
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

                    let _ = render_tx.send(render_events);
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
                let scancode = event.physical_key.to_scancode().unwrap_or(0);
                let key = match &event.logical_key {
                    winit::keyboard::Key::Named(named) => {
                        translate_named_key(*named, scancode)
                    }
                    _ => Key::Scancode(scancode),
                };
                let modifiers = self.current_modifiers;
                let input_event = match event.state {
                    ElementState::Pressed => InputEvent::KeyPressed { key, modifiers },
                    ElementState::Released => InputEvent::KeyReleased { key, modifiers },
                };
                self.pending_inputs.push(input_event);
                if event.state == ElementState::Pressed
                    && let Some(text) = event.text
                {
                    let s = text.to_string();
                    if !s.is_empty() && !s.chars().all(|c| c.is_control()) {
                        self.pending_inputs.push(InputEvent::TextInput { text: s });
                    }
                }
            }
            WindowEvent::ModifiersChanged(state) => {
                let m = state.state();
                self.current_modifiers = Modifiers {
                    shift: m.shift_key(),
                    ctrl: m.control_key(),
                    alt: m.alt_key(),
                    super_: m.super_key(),
                };
                self.pending_inputs
                    .push(InputEvent::ModifiersChanged(self.current_modifiers));
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let btn = match button {
                    winit::event::MouseButton::Left => MouseButton::Left,
                    winit::event::MouseButton::Right => MouseButton::Right,
                    winit::event::MouseButton::Middle => MouseButton::Middle,
                    _ => MouseButton::Other,
                };
                let modifiers = self.current_modifiers;
                let input_event = match state {
                    ElementState::Pressed => {
                        InputEvent::MouseButtonPressed { button: btn, modifiers }
                    }
                    ElementState::Released => {
                        InputEvent::MouseButtonReleased { button: btn, modifiers }
                    }
                };
                self.pending_inputs.push(input_event);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let (x, y) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => (x, y),
                    MouseScrollDelta::PixelDelta(pos) => {
                        (pos.x as f32 / 20.0, pos.y as f32 / 20.0)
                    }
                };
                self.pending_inputs.push(InputEvent::MouseScrolled { x, y });
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.pending_inputs.push(InputEvent::MouseMoved {
                    x: position.x as f32,
                    y: position.y as f32,
                });
            }
            WindowEvent::CursorLeft { .. } => {
                self.pending_inputs.push(InputEvent::CursorLeft);
            }
            WindowEvent::Focused(focused) => {
                self.pending_inputs.push(InputEvent::WindowFocused(focused));
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.pending_inputs
                    .push(InputEvent::ScaleFactorChanged(scale_factor as f32));
            }
            WindowEvent::DroppedFile(path) => {
                self.pending_inputs.push(InputEvent::FileDrop(path));
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
