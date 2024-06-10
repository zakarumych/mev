use proc_macro::TokenStream;

mod args;
mod repr;

#[cfg_attr(
    any(windows, all(unix, not(any(target_os = "macos", target_os = "ios")))),
    path = "vulkan/mod.rs"
)]
#[cfg_attr(any(target_os = "macos", target_os = "ios"), path = "metal/mod.rs")]
mod backend;

#[proc_macro_derive(Arguments, attributes(mev))]
pub fn arguments_derive(input: TokenStream) -> TokenStream {
    backend::arguments::derive(input)
}

#[proc_macro_derive(DeviceRepr, attributes(mev))]
pub fn repr_derive(input: TokenStream) -> TokenStream {
    repr::derive(input)
}
