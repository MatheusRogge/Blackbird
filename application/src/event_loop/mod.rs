#[derive(Debug, Clone, Copy)]
pub struct EventLoopError {}

pub enum Event {
    Exit,
}

pub(super) trait EventLoopEventProvider {
    fn run(&self) -> Result<(), EventLoopError>;
}
