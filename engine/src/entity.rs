use downcast_rs::{Downcast, impl_downcast};
use slotmap::new_key_type;
use std::any::Any;

new_key_type! {
    pub struct EntityKey;
}

pub trait Entity: Any + Downcast + Send + Sync + 'static {}

impl_downcast!(Entity);
