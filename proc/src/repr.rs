use proc_easy::private::Spanned;
use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn;

pub fn derive(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);

    match derive_impl(input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn derive_impl(input: syn::DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let name = &input.ident;
    let vis = &input.vis;

    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            input.generics,
            "generic arguments are not supported by `#[derive(DeviceRepr)]`",
        ));
    }

    let data = match input.data {
        syn::Data::Struct(data) => data,
        _ => {
            return Err(syn::Error::new_spanned(
                input,
                "only structs are supported by `#[derive(DeviceRepr)]`",
            ))
        }
    };

    let name_repr = quote::format_ident!("MevGenerated{}Pod", name);

    let field_types = data
        .fields
        .iter()
        .map(|field| &field.ty)
        .collect::<Vec<_>>();

    let field_pad_sizes = data
        .fields
        .iter()
        .enumerate()
        .map(|(idx, field)| {
            let end = data
                .fields
                .iter()
                .take(idx)
                .fold(quote! { 0 }, |acc, field| {
                    let ty = &field.ty;
                    quote_spanned! { ty.span() => mev::for_macro::repr_append_field::<#ty>(#acc) }
                });

            let ty = &field.ty;
            quote::quote_spanned! {
                ty.span() => mev::for_macro::repr_pad_for::<#ty>(#end)
            }
        })
        .collect::<Vec<_>>();

    let tail = data.fields.iter().fold(quote! { 0 }, |acc, field| {
        let ty = &field.ty;
        quote_spanned! { ty.span() => mev::for_macro::repr_append_field::<#ty>(#acc) }
    });

    let total_align = data.fields.iter().fold(quote! { 0 }, |acc, field| {
        let ty = &field.ty;
        quote_spanned! { ty.span() => #acc | (mev::for_macro::repr_align_of::<#ty>() - 1) }
    });

    let tail_pad = quote::quote!(mev::for_macro::pad_align(#tail, #total_align + 1));

    match data.fields {
        syn::Fields::Named(_) => {
            let field_names = data
                .fields
                .iter()
                .map(|field| &field.ident)
                .collect::<Vec<_>>();

            let pad_names = data
                .fields
                .iter()
                .map(|field| quote::format_ident!("_pad_for_{}", field.ident.as_ref().unwrap()))
                .collect::<Vec<_>>();

            let tokens = quote::quote! {
                #[repr(C)]
                #[doc(hidden)]
                #[derive(Clone, Copy, Debug)]
                #vis struct #name_repr {
                    #(
                        #pad_names: [u8; #field_pad_sizes],
                        #field_names: <#field_types as mev::for_macro::DeviceRepr>::Repr,
                    )*
                    _mev_tail_pad: [u8; #tail_pad],
                }

                unsafe impl mev::for_macro::Zeroable for #name_repr {}
                unsafe impl mev::for_macro::Pod for #name_repr {}

                impl mev::for_macro::DeviceRepr for #name {
                    type Repr = #name_repr;
                    type ArrayRepr = #name_repr;

                    #[inline(always)]
                    fn as_repr(&self) -> #name_repr {
                        #name_repr {
                            #(
                                #pad_names: [0xDAu8; #field_pad_sizes],
                                #field_names: mev::for_macro::DeviceRepr::as_repr(&self.#field_names),
                            )*
                            _mev_tail_pad: [0xDAu8; #tail_pad],
                        }
                    }

                    #[inline(always)]
                    fn as_array_repr(&self) -> #name_repr {
                        self.as_repr()
                    }

                    const ALIGN: usize = 1 + (#total_align);
                }
            };

            Ok(tokens)
        }
        _ => todo!(),
    }
}
