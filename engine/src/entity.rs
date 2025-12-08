use std::any::Any;

use downcast_rs::{Downcast, impl_downcast};

pub trait Entity: Any + Downcast + Send + Sync + 'static {}
impl_downcast!(Entity);
