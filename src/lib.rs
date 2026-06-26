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
pub use mev_proc::{match_backend, Arguments, AutoDeviceRepr, DeviceRepr, VertexBinding};

#[doc(hidden)]
pub mod for_macro {
    pub use crate::backend::for_macro::*;

    pub use crate::generic::{
        AutoDeviceRepr, Automatic, DataType, DeviceRepr, LibraryInput, Sampled, ScalarType,
        ShaderSource, Storage, Uniform, VectorSize, VertexAttributeDesc, VertexAttributes,
        VertexBinding, VertexFormat, VertexLayoutDesc, VertexStepMode,
    };

    pub use bytemuck::{Pod, Zeroable};
    use std::any::type_name;
    pub use std::{
        fmt,
        mem::{align_of, offset_of, size_of, MaybeUninit},
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

    #[doc(hidden)]
    pub struct VertexAttributeDescs<T: ?Sized> {
        pub buffer_index: u32,
        pub offset: usize,
        marker: std::marker::PhantomData<T>,
    }

    impl<T: ?Sized> Clone for VertexAttributeDescs<T> {
        #[inline(always)]
        fn clone(&self) -> Self {
            *self
        }
    }

    impl<T: ?Sized> Copy for VertexAttributeDescs<T> {}

    impl<T: ?Sized> fmt::Debug for VertexAttributeDescs<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("VertexAttributes")
                .field("type", &type_name::<T>())
                .field("buffer_index", &self.buffer_index)
                .field("offset", &self.offset)
                .finish()
        }
    }

    impl<T: ?Sized> VertexAttributeDescs<T>
    where
        T: VertexAttributes,
    {
        #[inline(always)]
        pub fn new(buffer_index: u32, offset: usize) -> VertexAttributeDescs<T> {
            VertexAttributeDescs {
                buffer_index,
                offset,
                marker: std::marker::PhantomData,
            }
        }
    }

    impl<T: ?Sized> IntoIterator for VertexAttributeDescs<T>
    where
        T: VertexAttributes,
    {
        type Item = VertexAttributeDesc;
        type IntoIter = VertexAttributeDescIter;

        fn into_iter(self) -> VertexAttributeDescIter {
            VertexAttributeDescIter {
                format: T::FORMAT,
                buffer_index: self.buffer_index,
                offset: self.offset,
                count: T::COUNT as u32,
            }
        }
    }

    pub struct VertexAttributeDescIter {
        format: VertexFormat,
        buffer_index: u32,
        offset: usize,
        count: u32,
    }

    impl Iterator for VertexAttributeDescIter {
        type Item = VertexAttributeDesc;

        #[inline]
        fn next(&mut self) -> Option<VertexAttributeDesc> {
            if self.count == 0 {
                return None;
            }

            let desc = VertexAttributeDesc {
                format: self.format,
                buffer_index: self.buffer_index,
                offset: u32::try_from(self.offset)
                    .expect("Vertex attribute offset exceeds u32::MAX"),
            };

            self.offset += self.format.size();
            self.count -= 1;

            Some(desc)
        }
    }
}
