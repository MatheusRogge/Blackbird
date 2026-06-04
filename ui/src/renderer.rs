use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use egui::{Event, Modifiers, Pos2, Vec2};
use engine_core::input::{InputEvent, Key, MouseButton};
use engine_core::world::World;

use crate::layer::UiLayer;

pub struct EguiRenderer {
    ctx: egui::Context,
    renderer: egui_wgpu::Renderer,
    layers: Vec<Box<dyn UiLayer>>,
    raw_input: egui::RawInput,
    pixels_per_point: f32,
    current_modifiers: Modifiers,
    cursor_pos: Option<Pos2>,
    ui_wants_pointer: Arc<AtomicBool>,
    ui_wants_keyboard: Arc<AtomicBool>,
}

impl EguiRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        layers: Vec<Box<dyn UiLayer>>,
        ui_wants_pointer: Arc<AtomicBool>,
        ui_wants_keyboard: Arc<AtomicBool>,
    ) -> Self {
        let renderer = egui_wgpu::Renderer::new(
            device,
            surface_format,
            egui_wgpu::RendererOptions::default(),
        );

        Self {
            ctx: egui::Context::default(),
            renderer,
            layers,
            raw_input: egui::RawInput::default(),
            pixels_per_point: 1.0,
            current_modifiers: Modifiers::default(),
            cursor_pos: None,
            ui_wants_pointer,
            ui_wants_keyboard,
        }
    }

    pub fn on_events(&mut self, events: &[InputEvent]) {
        for event in events {
            match event {
                InputEvent::KeyPressed { key, modifiers } => {
                    self.current_modifiers = to_egui_mods(*modifiers);
                    if let Some(egui_key) = to_egui_key(key) {
                        self.raw_input.events.push(Event::Key {
                            key: egui_key,
                            physical_key: None,
                            pressed: true,
                            repeat: false,
                            modifiers: self.current_modifiers,
                        });
                    }
                }
                InputEvent::KeyReleased { key, modifiers } => {
                    self.current_modifiers = to_egui_mods(*modifiers);
                    if let Some(egui_key) = to_egui_key(key) {
                        self.raw_input.events.push(Event::Key {
                            key: egui_key,
                            physical_key: None,
                            pressed: false,
                            repeat: false,
                            modifiers: self.current_modifiers,
                        });
                    }
                }
                InputEvent::TextInput { text } => {
                    self.raw_input.events.push(Event::Text(text.clone()));
                }
                InputEvent::ModifiersChanged(m) => {
                    self.current_modifiers = to_egui_mods(*m);
                    self.raw_input.modifiers = self.current_modifiers;
                }
                InputEvent::MouseButtonPressed { button, modifiers } => {
                    self.current_modifiers = to_egui_mods(*modifiers);
                    if let Some(pos) = self.cursor_pos {
                        self.raw_input.events.push(Event::PointerButton {
                            pos,
                            button: to_egui_button(*button),
                            pressed: true,
                            modifiers: self.current_modifiers,
                        });
                    }
                }
                InputEvent::MouseButtonReleased { button, modifiers } => {
                    self.current_modifiers = to_egui_mods(*modifiers);
                    if let Some(pos) = self.cursor_pos {
                        self.raw_input.events.push(Event::PointerButton {
                            pos,
                            button: to_egui_button(*button),
                            pressed: false,
                            modifiers: self.current_modifiers,
                        });
                    }
                }
                InputEvent::MouseMoved { x, y } => {
                    let ppp = self.pixels_per_point;
                    let pos = Pos2::new(x / ppp, y / ppp);
                    self.cursor_pos = Some(pos);
                    self.raw_input.events.push(Event::PointerMoved(pos));
                }
                InputEvent::MouseScrolled { x, y } => {
                    self.raw_input.events.push(Event::MouseWheel {
                        unit: egui::MouseWheelUnit::Point,
                        delta: Vec2::new(*x, *y),
                        modifiers: self.current_modifiers,
                        phase: egui::TouchPhase::Move,
                    });
                }
                InputEvent::CursorLeft => {
                    self.cursor_pos = None;
                    self.raw_input.events.push(Event::PointerGone);
                }
                InputEvent::WindowFocused(focused) => {
                    self.raw_input.events.push(Event::WindowFocused(*focused));
                }
                InputEvent::ScaleFactorChanged(scale) => {
                    self.pixels_per_point = *scale;
                }
                InputEvent::FileDrop(_) => {}
            }
        }
    }
}

impl EguiRenderer {
    pub fn execute(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        size: (u32, u32),
        world: &World,
    ) {
        let ppp = self.pixels_per_point;
        let screen_desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [size.0, size.1],
            pixels_per_point: ppp,
        };

        self.raw_input.screen_rect = Some(egui::Rect::from_min_size(
            Pos2::ZERO,
            Vec2::new(size.0 as f32 / ppp, size.1 as f32 / ppp),
        ));
        self.raw_input
            .viewports
            .entry(egui::ViewportId::ROOT)
            .or_default()
            .native_pixels_per_point = Some(ppp);

        let raw_input = std::mem::take(&mut self.raw_input);
        self.ctx.begin_pass(raw_input);
        for layer in self.layers.iter_mut() {
            layer.ui(&self.ctx, world);
        }
        let full_output = self.ctx.end_pass();

        self.ui_wants_pointer
            .store(self.ctx.egui_wants_pointer_input(), Ordering::Relaxed);
        self.ui_wants_keyboard
            .store(self.ctx.egui_wants_keyboard_input(), Ordering::Relaxed);

        for (id, delta) in &full_output.textures_delta.set {
            self.renderer.update_texture(device, queue, *id, delta);
        }
        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }

        let primitives = self.ctx.tessellate(full_output.shapes, ppp);

        let extra_cmds =
            self.renderer
                .update_buffers(device, queue, encoder, &primitives, &screen_desc);

        {
            let mut rpass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                })
                .forget_lifetime();
            self.renderer.render(&mut rpass, &primitives, &screen_desc);
        }

        if !extra_cmds.is_empty() {
            queue.submit(extra_cmds);
        }
    }
}

fn to_egui_mods(m: engine_core::input::Modifiers) -> Modifiers {
    Modifiers {
        alt: m.alt,
        ctrl: m.ctrl,
        shift: m.shift,
        mac_cmd: false,
        command: m.ctrl,
    }
}

fn to_egui_key(key: &Key) -> Option<egui::Key> {
    match key {
        Key::ArrowUp => Some(egui::Key::ArrowUp),
        Key::ArrowDown => Some(egui::Key::ArrowDown),
        Key::ArrowLeft => Some(egui::Key::ArrowLeft),
        Key::ArrowRight => Some(egui::Key::ArrowRight),
        Key::Enter => Some(egui::Key::Enter),
        Key::Escape => Some(egui::Key::Escape),
        Key::Tab => Some(egui::Key::Tab),
        Key::Space => Some(egui::Key::Space),
        Key::Backspace => Some(egui::Key::Backspace),
        Key::Delete => Some(egui::Key::Delete),
        Key::Insert => Some(egui::Key::Insert),
        Key::Home => Some(egui::Key::Home),
        Key::End => Some(egui::Key::End),
        Key::PageUp => Some(egui::Key::PageUp),
        Key::PageDown => Some(egui::Key::PageDown),
        Key::F1 => Some(egui::Key::F1),
        Key::F2 => Some(egui::Key::F2),
        Key::F3 => Some(egui::Key::F3),
        Key::F4 => Some(egui::Key::F4),
        Key::F5 => Some(egui::Key::F5),
        Key::F6 => Some(egui::Key::F6),
        Key::F7 => Some(egui::Key::F7),
        Key::F8 => Some(egui::Key::F8),
        Key::F9 => Some(egui::Key::F9),
        Key::F10 => Some(egui::Key::F10),
        Key::F11 => Some(egui::Key::F11),
        Key::F12 => Some(egui::Key::F12),
        Key::Scancode(_) => None,
    }
}

fn to_egui_button(button: MouseButton) -> egui::PointerButton {
    match button {
        MouseButton::Left => egui::PointerButton::Primary,
        MouseButton::Right => egui::PointerButton::Secondary,
        MouseButton::Middle => egui::PointerButton::Middle,
        MouseButton::Other => egui::PointerButton::Extra1,
    }
}
