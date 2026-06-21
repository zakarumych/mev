//! Tiny graphics crate made for nothing but fun.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(unused)]

mod generic;
mod traits;

/// Macro that passes-through any tokens inside if chosen backend is Metal.
/// Otherwise, it unwraps to nothing.
///
/// # Example
///
/// ```
/// with_metal! {
///    // Metal-specific code
/// }
/// ```
#[macro_export]
#[cfg(mev_backend = "metal")]
macro_rules! with_metal {
    ($($tokens:tt)*) => {
        $($tokens)*
    };
}

/// Macro that passes-through any tokens inside if chosen backend is Metal.
/// Otherwise, it unwraps to nothing.
///
/// # Example
///
/// ```
/// with_metal! {
///    // Metal-specific code
/// }
/// ```
#[macro_export]
#[cfg(not(mev_backend = "metal"))]
macro_rules! with_metal {
    ($($tokens:tt)*) => {
        // Nothing
    };
}

/// Macro that passes-through any tokens inside if chosen backend is Vulkan.
/// Otherwise, it unwraps to nothing.
///
/// # Example
///
/// ```
/// with_vulkan! {
///    // Vulkan-specific code
/// }
/// ```
#[macro_export]
#[cfg(mev_backend = "vulkan")]
macro_rules! with_vulkan {
    ($($tokens:tt)*) => {
        $($tokens)*
    };
}

/// Macro that passes-through any tokens inside if chosen backend is Vulkan.
/// Otherwise, it unwraps to nothing.
///
/// # Example
///
/// ```
/// with_vulkan! {
///    // Vulkan-specific code
/// }
/// ```
#[macro_export]
#[cfg(not(mev_backend = "vulkan"))]
macro_rules! with_vulkan {
    ($($tokens:tt)*) => {
        // Nothing
    };
}

/// Macro that passes-through any tokens inside if chosen backend is WebGL.
/// Otherwise, it unwraps to nothing.
///
/// # Example
///
/// ```
/// with_webgl! {
///    // WebGL-specific code
/// }
/// ```
#[macro_export]
#[cfg(mev_backend = "webgl")]
macro_rules! with_webgl {
    ($($tokens:tt)*) => {
        $($tokens)*
    };
}

/// Macro that passes-through any tokens inside if chosen backend is WebGL.
/// Otherwise, it unwraps to nothing.
///
/// # Example
///
/// ```
/// with_webgl! {
///    // WebGL-specific code
/// }
/// ```
#[macro_export]
#[cfg(not(mev_backend = "webgl"))]
macro_rules! with_webgl {
    ($($tokens:tt)*) => {
        // Nothing
    };
}

with_metal! {
    #[path = "metal/mod.rs"]
    mod backend;
}

with_vulkan! {
    #[path = "vulkan/mod.rs"]
    mod backend;
}

with_webgl! {
    #[path = "webgl/mod.rs"]
    mod backend;
}

/// Backend that is used for rendering.
pub enum Backend {
    Metal,
    Vulkan,
    WebGL,
}

impl Backend {
    with_metal! {
        /// Current backend constant.
        const CURRENT: Self = Self::Metal;
    }

    with_vulkan! {
        /// Current backend constant.
        const CURRENT: Self = Self::Vulkan;
    }

    with_webgl! {
        /// Current backend constant.
        const CURRENT: Self = Self::WebGL;
    }
}

mod private {
    pub trait Sealed {}
}

pub use self::{backend::*, generic::*};
pub use mev_proc::{match_backend, Arguments, AutoDeviceRepr, DeviceRepr};

#[doc(hidden)]
pub mod for_macro {
    pub use crate::backend::for_macro::*;

    pub use crate::generic::{
        AutoDeviceRepr, Automatic, DeviceRepr, LibraryInput, Sampled, ShaderSource, Storage,
        Uniform,
    };

    pub use bytemuck::{Pod, Zeroable};
    pub use std::{
        mem::{align_of, size_of, MaybeUninit},
        ptr::addr_of,
    };

    pub const fn align_end(end: usize, align: usize) -> usize {
        ((end + (align - 1)) & !(align - 1))
    }

    pub const fn repr_pad_for<T: DeviceRepr>(end: usize) -> usize {
        let align = T::ALIGN;
        pad_align(end, align)
    }

    pub const fn pad_align(end: usize, align: usize) -> usize {
        align_end(end, align) - end
    }

    pub const fn repr_append_field<T: DeviceRepr>(end: usize) -> usize {
        align_end(end, T::ALIGN) + T::SIZE
    }

    pub const fn repr_align_of<T: DeviceRepr>() -> usize {
        T::ALIGN
    }

    pub const fn is_repr<T: DeviceRepr>() {}

    pub const fn is_auto_repr<T: AutoDeviceRepr>() {}
}
