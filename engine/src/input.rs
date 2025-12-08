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
        if value == 17 {
            return Self::W;
        }

        if value == 30 {
            return Self::A;
        }

        if value == 31 {
            return Self::S;
        }

        if value == 32 {
            return Self::D;
        }

        Self::Unknown
    }
}

#[derive(Debug)]
pub enum InputEvent {
    KeyReleased { key_code: KeyCodes },
    KeyPressed { key_code: KeyCodes },
}

pub struct InputBuffer {
    events: VecDeque<InputEvent>,
}

impl Default for InputBuffer {
    fn default() -> Self {
        Self {
            events: VecDeque::with_capacity(10),
        }
    }
}

impl InputBuffer {
    pub fn push(&mut self, event: InputEvent) {
        self.events.push_front(event);
        self.events.truncate(10);
    }

    pub fn pop(&mut self) -> Option<InputEvent> {
        self.events.pop_back()
    }
}
