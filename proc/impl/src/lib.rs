use proc_macro2::TokenStream;

macro_rules! parse_macro_input {
    ($token_stream:ident as $ty:ty) => {
        match syn::parse2::<$ty>($token_stream) {
            Ok(input) => input,
            Err(err) => return err.to_compile_error().into(),
        }
    };
}

mod args;
mod auto_repr;
mod r#match;
mod metal;
mod repr;
mod vertex;
mod vulkan;
// mod webgl;
mod hidden_trait;
mod webgpu;

pub fn arguments_derive(input: TokenStream, mev: &TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);

    let metal_tokens = match metal::arguments::derive(&input, mev) {
        Ok(tokens) => tokens,
        Err(err) => err.to_compile_error(),
    };

    let vulkan_tokens = match vulkan::arguments::derive(&input, mev) {
        Ok(tokens) => tokens,
        Err(err) => err.to_compile_error(),
    };

    let webgpu_tokens = match webgpu::arguments::derive(&input, mev) {
        Ok(tokens) => tokens,
        Err(err) => err.to_compile_error(),
    };
    quote::quote! {
        #mev::with_metal! { #metal_tokens }
        #mev::with_vulkan! { #vulkan_tokens }
        #mev::with_webgpu! { #webgpu_tokens }
    }
}

pub fn derive_repr(input: TokenStream, mev: &TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);

    match repr::derive(&input, mev) {
        Ok(tokens) => tokens,
        Err(err) => err.to_compile_error(),
    }
}

pub fn derive_auto_repr(input: TokenStream, mev: &TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);

    match auto_repr::derive(&input, mev) {
        Ok(tokens) => tokens,
        Err(err) => err.to_compile_error(),
    }
}

pub fn match_backend(input: TokenStream, mev: &TokenStream) -> TokenStream {
    r#match::match_backend(input, mev)
}

pub fn derive_vertex(input: TokenStream, mev: &TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);

    match vertex::derive(&input, mev) {
        Ok(tokens) => tokens,
        Err(err) => err.to_compile_error(),
    }
}

/// Implements the `#[hidden_trait]` attribute macro.
///
/// Applied on top of `impl Trait for Type { .. }` it splits the impl in two:
/// an inherent `impl Type { .. }` with all items made `pub`, and a companion
/// `impl Trait for Type { .. }` whose items simply reexport the inherent ones.
///
/// This lets callers use trait methods on `Type` without having the trait
/// itself in scope.
pub fn hidden_trait(attr: TokenStream, input: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        return syn::Error::new_spanned(attr, "`#[hidden_trait]` does not take any arguments")
            .to_compile_error();
    }

    let item_impl = parse_macro_input!(input as syn::ItemImpl);

    match hidden_trait::expand(item_impl) {
        Ok(tokens) => tokens,
        Err(err) => err.to_compile_error(),
    }
}
