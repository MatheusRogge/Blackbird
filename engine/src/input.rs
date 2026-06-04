use std::collections::VecDeque;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, Default)]
pub struct Modifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    pub super_: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Key {
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    Enter,
    Escape,
    Tab,
    Space,
    Backspace,
    Delete,
    Insert,
    Home,
    End,
    PageUp,
    PageDown,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    Scancode(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    Other,
}

#[derive(Debug, Clone)]
pub enum InputEvent {
    KeyPressed { key: Key, modifiers: Modifiers },
    KeyReleased { key: Key, modifiers: Modifiers },
    TextInput { text: String },
    ModifiersChanged(Modifiers),
    MouseButtonPressed { button: MouseButton, modifiers: Modifiers },
    MouseButtonReleased { button: MouseButton, modifiers: Modifiers },
    MouseMoved { x: f32, y: f32 },
    MouseScrolled { x: f32, y: f32 },
    CursorLeft,
    WindowFocused(bool),
    ScaleFactorChanged(f32),
    FileDrop(PathBuf),
}

pub struct InputBuffer {
    events: VecDeque<InputEvent>,
}

impl Default for InputBuffer {
    fn default() -> Self {
        Self {
            events: VecDeque::with_capacity(64),
        }
    }
}

impl InputBuffer {
    pub fn push(&mut self, event: InputEvent) {
        self.events.push_front(event);
        self.events.truncate(64);
    }

    pub fn pop(&mut self) -> Option<InputEvent> {
        self.events.pop_back()
    }
}
