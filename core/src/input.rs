use std::collections::VecDeque;

#[derive(Debug)]
pub enum KeyCodes {
    W,
    A,
    S,
    D,
    Unknown,
}

impl From<u32> for KeyCodes {
    fn from(value: u32) -> Self {
        match value {
            17 => Self::W,
            30 => Self::A,
            31 => Self::S,
            32 => Self::D,
            _ => Self::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    Other,
}

#[derive(Debug)]
pub enum InputEvent {
    KeyReleased { key_code: KeyCodes },
    KeyPressed { key_code: KeyCodes },
    MouseButtonPressed { button: MouseButton },
    MouseButtonReleased { button: MouseButton },
    /// Absolute cursor position in physical pixels.
    MouseMoved { x: f32, y: f32 },
    /// Scroll delta — positive is up / zoom-in.
    MouseScrolled { delta: f32 },
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
