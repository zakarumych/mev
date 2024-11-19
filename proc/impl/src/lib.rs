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
mod repr;
mod r#match;

mod metal;
mod vulkan;

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

    quote::quote! {
        #mev::with_metal!{#metal_tokens}
        #mev::with_vulkan!{#vulkan_tokens}
    }
}

pub fn repr_derive(input: TokenStream, mev: &TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);

    match repr::derive(&input, mev) {
        Ok(tokens) => tokens,
        Err(err) => err.to_compile_error(),
    }
}

pub fn match_backend(input: TokenStream, mev: &TokenStream) -> TokenStream {
    r#match::match_backend(input, mev)
}
