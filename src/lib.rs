//! Tiny graphics crate made for nothing but fun.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(warnings)]

mod generic;
mod traits;

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
///
#[macro_export]
#[cfg(any(windows, all(unix, not(any(target_os = "macos", target_os = "ios")))))]
macro_rules! with_vulkan {
    ($($tokens:tt)*) => {
        $($tokens)*
    };
}

/// Macro that passes-through any tokens inside if chosen backend is Vulkan.
/// Otherwise, it unwraps to nothing.
#[macro_export]
#[cfg(any(target_os = "macos", target_os = "ios"))]
macro_rules! with_vulkan {
    ($($tokens:tt)*) => {
        // Nothing
    };
}

/// Macro that passes-through any tokens inside if chosen backend is Metal.
/// Otherwise, it unwraps to nothing.
#[macro_export]
#[cfg(any(target_os = "macos", target_os = "ios"))]
macro_rules! with_metal {
    ($($tokens:tt)*) => {
        $($tokens)*
    };
}

/// Macro that passes-through any tokens inside if chosen backend is Metal.
/// Otherwise, it unwraps to nothing.
#[macro_export]
#[cfg(any(windows, all(unix, not(any(target_os = "macos", target_os = "ios")))))]
macro_rules! with_metal {
    ($($tokens:tt)*) => {
        // Nothing
    };
}

// #[doc(hidden)]
// #[macro_export]
// macro_rules! match_backend_impl {
//     // Wildcard match check.
//     (# [- $metal:tt] { $($tokens:tt)* }) => {
//         // Vulkan was not matched
//         $crate::match_backend_impl!{# [+ $metal] { $($tokens)* }}
//         $crate::with_vulkan!{$($tokens)*}
//     };
//     (# [+ -] { $($tokens:tt)* }) => {
//         // Metal was not matched
//         $crate::with_metal!{$($tokens)*}
//     };
//     (# [+ +] { $($tokens:tt)* }) => {};
    
//     // Second match
//     (^ $backend:tt) => {
//         ::core::compile_error!(::core::concat!("`", stringify!($backend), "` backend is matched more than once"));
//     };

//     // Missing match
//     (- $backend:tt) => {
//         ::core::compile_error!(::core::concat!("`", stringify!($backend), "` backend is matched more than once"));
//     };

//     // Wildcard is last
//     (_ $backend:tt) => {};
//     // Wildcard is not last
//     (_ $backend:tt $($tail:tt)+) => {
//         ::core::compile_error!(::core::concat!("Wildcard pattern `", stringify!($backend), "` must be the last"));
//     };

//     // Start
//     (! $backend:tt => { $($tokens:tt)* } $($tail_backend:tt => {$($tail_tokens:tt)*})*) => {
//         $crate::match_backend_impl!{@ [- -] $backend => { $($tokens)* } $($tail_backend => {$($tail_tokens)*})*}
//     };

//     // End of the pattern list with wildcard.
//     (* [$vulkan:tt $metal:tt]) => {};

//     // End of pattern list without wildcard.
//     (@ [- $metal:tt]) => {
//         $crate::match_backend_impl!{- vulkan}
//         $crate::match_backend_impl!{@ [+ $metal]}
//     };
//     (@ [+ -]) => {
//         $crate::match_backend_impl!{- metal}
//     };
//     (@ [+ +]) => {};

//     // Recursion
//     ($w:tt [$vulkan:tt $metal:tt] $backend:tt => { $($tokens:tt)* } $($tail_backend:tt => {$($tail_tokens:tt)*})*) => {
//         $crate::match_backend_impl!{~ @ [$vulkan $metal] $backend $backend => { $($tokens)* } $($tail_backend => {$($tail_tokens)*})*}
//     };

//     // Match Vulkan
//     (~ @ [- $metal:tt] $backend:tt vulkan => { $($tokens:tt)* } $($tail_backend:tt => {$($tail_tokens:tt)*})*) => {
//         // Recursion
//         $crate::match_backend_impl!{@ [+ $metal] $($tail_backend => {$($tail_tokens)*})*}

//         // Emit tokens
//         $crate::with_vulkan! { $($tokens)* }
//     };
//     (~ * [- $metal:tt] $backend:tt vulkan => { $($tokens:tt)* } $($tail_backend:tt => {$($tail_tokens:tt)*})*) => {
//         // Recursion
//         $crate::match_backend_impl!{* [+ $metal] $($tail_backend => {$($tail_tokens)*})*}
//     };
//     (~ $w:tt [+ $metal:tt] $backend:tt vulkan => { $($tokens:tt)* } $($tail_backend:tt => {$($tail_tokens:tt)*})*) => {
//         $crate::match_backend_impl!{^ $backend}
//         // Recursion
//         $crate::match_backend_impl!{$w [+ $metal] $($tail_backend => {$($tail_tokens)*})*}
//     };

//     // Match metal
//     (~ @ [$vulkan:tt -] $backend:tt metal => { $($tokens:tt)* } $($tail_backend:tt => {$($tail_tokens:tt)*})*) => {
//         // Recursion
//         $crate::match_backend_impl!{@ [$vulkan +] $($tail_backend => {$($tail_tokens)*})*}

//         // Emit tokens
//         $crate::with_metal! { $($tokens)* }
//     };
//     (~ * [$vulkan:tt -] $backend:tt metal => { $($tokens:tt)* } $($tail_backend:tt => {$($tail_tokens:tt)*})*) => {
//         // Recursion
//         $crate::match_backend_impl!{* [+ $metal] $($tail_backend => {$($tail_tokens)*})*}
//     };
//     (~ $w:tt [$vulkan:tt +] $backend:tt metal => { $($tokens:tt)* } $($tail_backend:tt => {$($tail_tokens:tt)*})*) => {
//         $crate::match_backend_impl!{^ $backend}
//         // Recursion
//         $crate::match_backend_impl!{$w [+ $metal] $($tail_backend => {$($tail_tokens)*})*}
//     };

//     // Match wildcard
//     (~ @ [$vulkan:tt $metal:tt] $backend:tt _ => { $($tokens:tt)* } $($tail_backend:tt => {$($tail_tokens:tt)*})*) => {
//         $crate::match_backend_impl!{_ $backend $($tail_backend => {$($tail_tokens)*})*}

//         // Recursion
//         $crate::match_backend_impl!{* [$vulkan +] $($tail_backend => {$($tail_tokens)*})*}

//         // Emit tokens
//         $crate::match_backend_impl!{ # [$vulkan $metal] { $($tokens)* } }
//     };
//     (~ * [$vulkan:tt $metal:tt] $backend:tt _ => { $($tokens:tt)* } $($tail_backend:tt => {$($tail_tokens:tt)*})*) => {
//         $crate::match_backend_impl!{_ $backend $($tail_backend => {$($tail_tokens)*})*}

//         // Recursion
//         $crate::match_backend_impl!{* [+ $metal] $($tail_backend => {$($tail_tokens)*})*}
//     };

//     // Match unknown
//     (~ $w:tt [$vulkan:tt $metal:tt] $backend:tt $unknown:ident => { $($tokens:tt)* } $($tail_backend:tt => {$($tail_tokens:tt)*})*) => {
//         ::core::compile_error!(::core::concat!("Unknown backend `", stringify!($unknown), "`"));

//         // Recursion
//         $crate::match_backend_impl!{$w [$vulkan $metal:tt] $($tail_backend => {$($tail_tokens)*})*}
//     };
// }

// /// Macro that matches the backend and emits tokens assigned to the chosen backend.
// #[macro_export]
// macro_rules! match_backend {
//     ($( $backend:tt => { $($tokens:tt)* } )*) => {
//         $crate::match_backend_impl!{! $($backend => { $($tokens)* })*}
//     };
// }

with_vulkan! {
    #[path = "vulkan/mod.rs"]
    mod backend;
}

with_metal! {
    #[path = "metal/mod.rs"]
    mod backend;
}

/// Backend that is used for rendering.
pub enum Backend {
    Vulkan,
    Metal,
}

impl Backend {
    with_vulkan! {
        /// Current backend constant.
        const CURRENT: Self = Self::Vulkan;
    }

    with_metal! {
        /// Current backend constant.
        const CURRENT: Self = Self::Metal;
    }
}

mod private {
    pub trait Sealed {}
}

pub use self::{backend::*, generic::*};
pub use mev_proc::{Arguments, DeviceRepr, match_backend};

#[doc(hidden)]
pub mod for_macro {
    pub use crate::backend::for_macro::*;

    pub use crate::generic::{
        Automatic, DeviceRepr, LibraryInput, Sampled, ShaderSource, Storage, Uniform,
    };
}
