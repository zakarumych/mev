use proc_easy::private::Spanned;
use proc_macro2::TokenStream;
use quote::{quote, quote_spanned};
use syn;

pub fn derive(input: &syn::DeriveInput, mev: &TokenStream) -> syn::Result<TokenStream> {
    let name = &input.ident;

    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.generics,
            "generic arguments are not supported by `#[derive(DeviceRepr)]`",
        ));
    }

    let data = match &input.data {
        syn::Data::Struct(data) => data,
        _ => {
            return Err(syn::Error::new_spanned(
                input,
                "only structs are supported by `#[derive(DeviceRepr)]`",
            ))
        }
    };

    let mut pad_sizes_are_zero = quote! { let mut __mev_device_repr_end = 0; };

    for (idx, field) in data.fields.iter().enumerate() {
        let ty = &field.ty;
        let memeber = field
            .ident
            .as_ref()
            .map_or_else(|| format!("{idx}"), |ident| ident.to_string());

        pad_sizes_are_zero.extend(quote::quote_spanned! {
            ty.span() => {
                let padding = #mev::for_macro::repr_pad_for::<#ty>(__mev_device_repr_end);
                if 0 != padding {
                    match padding {
                        1 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Field `", #memeber, "` requires padding of 1 byte")),
                        2 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Field `", #memeber, "` requires padding of 2 bytes")),
                        3 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Field `", #memeber, "` requires padding of 3 bytes")),
                        4 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Field `", #memeber, "` requires padding of 4 bytes")),
                        5 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Field `", #memeber, "` requires padding of 5 bytes")),
                        6 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Field `", #memeber, "` requires padding of 6 bytes")),
                        7 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Field `", #memeber, "` requires padding of 7 bytes")),
                        8 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Field `", #memeber, "` requires padding of 8 bytes")),
                        9 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Field `", #memeber, "` requires padding of 9 bytes")),
                        10 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Field `", #memeber, "` requires padding of 10 bytes")),
                        11 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Field `", #memeber, "` requires padding of 11 bytes")),
                        12 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Field `", #memeber, "` requires padding of 12 bytes")),
                        13 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Field `", #memeber, "` requires padding of 13 bytes")),
                        14 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Field `", #memeber, "` requires padding of 14 bytes")),
                        15 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Field `", #memeber, "` requires padding of 15 bytes")),
                        16 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Field `", #memeber, "` requires padding of 16 bytes")),
                        _ => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Field `", #memeber, "` requires padding of many bytes")),
                    }
                }
                __mev_device_repr_end += <#ty as #mev::for_macro::DeviceRepr>::SIZE;
            }
        });
    }

    let total_align = data.fields.iter().fold(quote! { 0 }, |acc, field| {
        let ty = &field.ty;
        quote_spanned! { ty.span() => #acc | (#mev::for_macro::repr_align_of::<#ty>() - 1) }
    });

    pad_sizes_are_zero.extend(quote! {
        let padding = #mev::for_macro::pad_align(__mev_device_repr_end, (#total_align) + 1);
        if 0 != padding {
            match padding {
                1 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Tail padding is required of 1 byte")),
                2 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Tail padding is required of 2 bytes")),
                3 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Tail padding is required of 3 bytes")),
                4 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Tail padding is required of 4 bytes")),
                5 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Tail padding is required of 5 bytes")),
                6 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Tail padding is required of 6 bytes")),
                7 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Tail padding is required of 7 bytes")),
                8 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Tail padding is required of 8 bytes")),
                9 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Tail padding is required of 9 bytes")),
                10 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Tail padding is required of 10 bytes")),
                11 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Tail padding is required of 11 bytes")),
                12 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Tail padding is required of 12 bytes")),
                13 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Tail padding is required of 13 bytes")),
                14 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Tail padding is required of 14 bytes")),
                15 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Tail padding is required of 15 bytes")),
                16 => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Tail padding is required of 16 bytes")),
                _ => panic!(concat!("struct `", stringify!(#name), "` is not a valid device representation. Tail padding is required of many bytes")),
            }
        }
    });

    let mut all_fields_are_auto_repr = quote! {};

    for field in data.fields.iter() {
        let ty = &field.ty;

        all_fields_are_auto_repr.extend(quote::quote_spanned! {
            ty.span() => {
                #mev::for_macro::is_auto_repr::<#ty>();
            }
        });
    }

    match data.fields {
        syn::Fields::Named(_) => {
            let tokens = quote::quote! {
                impl #mev::for_macro::DeviceRepr for #name {
                    type Repr = Self;
                    type ArrayRepr = Self;

                    #[inline(always)]
                    fn as_repr(&self) -> Self {
                        *self
                    }

                    #[inline(always)]
                    fn as_array_repr(&self) -> Self {
                        *self
                    }

                    const ALIGN: usize = 1 + (#total_align);
                }

                const _: () = {
                    #pad_sizes_are_zero
                    #all_fields_are_auto_repr
                };
            };

            Ok(tokens)
        }
        _ => todo!(),
    }
}
