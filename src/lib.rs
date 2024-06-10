//! Tiny graphics crate made for nothing but fun.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(warnings)]

pub mod generic;
mod traits;

#[cfg_attr(
    any(windows, all(unix, not(any(target_os = "macos", target_os = "ios")))),
    path = "vulkan/mod.rs"
)]
#[cfg_attr(any(target_os = "macos", target_os = "ios"), path = "metal/mod.rs")]
pub mod backend;

mod private {
    pub trait Sealed {}
}

pub use self::{backend::*, generic::*};
pub use mev_proc::{Arguments, DeviceRepr};

#[doc(hidden)]
pub mod for_macro {
    pub use crate::backend::for_macro::*;
    pub use crate::generic::{
        Automatic, DeviceRepr, LibraryInput, Sampled, ShaderSource, Storage, Uniform,
    };
}
