use proc_easy::{private::Spanned, EasyAttributes};
use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn;

use crate::args::*;

pub fn derive(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);

    match derive_impl(input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn derive_impl(input: syn::DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let name = &input.ident;

    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            input.generics,
            "generic arguments are not supported by `#[derive(Arguments)]`",
        ));
    }

    let data = match input.data {
        syn::Data::Struct(data) => data,
        _ => {
            return Err(syn::Error::new_spanned(
                input,
                "only structs are supported by `#[derive(Arguments)]`",
            ))
        }
    };

    let field_attrs = data
        .fields
        .iter()
        .map(|field| FieldAttributes::parse(&field.attrs, field.span()))
        .collect::<Result<Vec<_>, _>>()?;

    let field_argument_impls = data
        .fields
        .iter()
        .zip(&field_attrs)
        .map(|(field, attrs)| {
            let ty = &field.ty;
            match attrs.kind {
                None => quote::quote!(<#ty as mev::for_macro::ArgumentsField<mev::for_macro::Automatic>>),
                // Some(Kind::Constant(_)) => {
                //     quote::quote!(<#ty as mev::for_macro::ArgumentsField<mev::for_macro::Constant>>)
                // }
                Some(Kind::Uniform(_)) => {
                    quote::quote!(<#ty as mev::for_macro::ArgumentsField<mev::for_macro::Uniform>>)
                }
                Some(Kind::Sampled(_)) => {
                    quote::quote!(<#ty as mev::for_macro::ArgumentsField<mev::for_macro::Sampled>>)
                }
                Some(Kind::Storage(_)) => {
                    quote::quote!(<#ty as mev::for_macro::ArgumentsField<mev::for_macro::Storage>>)
                }
            }
        })
        .collect::<Vec<_>>();

    let field_stages = data
        .fields
        .iter()
        .zip(&field_attrs)
        .map(|(field, attrs)| {
            if attrs.shaders.flags.is_empty() {
                quote_spanned!(field.span() => mev::ShaderStage::empty())
            } else {
                let mut tokens = quote!(0);

                for stage in attrs.shaders.flags.iter() {
                    match stage {
                        Shader::Vertex(vertex) => {
                            if !tokens.is_empty() {
                                tokens.extend(quote_spanned!(vertex.span() => | ));
                            }
                            tokens
                                .extend(quote_spanned!(vertex.span() => mev::ShaderStages::VERTEX.bits()))
                        }
                        Shader::Fragment(fragment) => {
                            if !tokens.is_empty() {
                                tokens.extend(quote_spanned!(fragment.span() => | ));
                            }
                            tokens.extend(
                                quote_spanned!(fragment.span() => mev::ShaderStages::FRAGMENT.bits()),
                            )
                        }
                        Shader::Compute(compute) => {
                            if !tokens.is_empty() {
                                tokens.extend(quote_spanned!(compute.span() => | ));
                            }
                            tokens.extend(
                                quote_spanned!(compute.span() => mev::ShaderStages::COMPUTE.bits()),
                            )
                        }
                    }
                }

                quote!(mev::ShaderStages::from_bits_truncate(#tokens))
            }
        })
        .collect::<Vec<_>>();

    match &data.fields {
        syn::Fields::Unit => {
            return Err(syn::Error::new_spanned(
                data.fields,
                "unit structs are not supported by `#[derive(Arguments)]`",
            ));
        }
        syn::Fields::Unnamed(_) => todo!(),
        syn::Fields::Named(fields) => {
            let field_names = fields
                .named
                .iter()
                .map(|field| field.ident.as_ref().unwrap())
                .collect::<Vec<_>>();

            Ok(quote! {
                impl mev::for_macro::Arguments for #name {
                    const LAYOUT: mev::ArgumentGroupLayout<'static> = mev::ArgumentGroupLayout {
                        arguments: &[#(mev::ArgumentLayout {
                            kind: #field_argument_impls::KIND,
                            size: #field_argument_impls::SIZE,
                            stages: #field_stages,
                        },)*],
                    };

                    #[inline(always)]
                    fn bind_render(&self, group: u32, encoder: &mut mev::RenderCommandEncoder) {
                        let metal = encoder.metal();
                        let vertex_bindings = encoder.vertex_bindings();
                        let fragment_bindings = encoder.fragment_bindings();

                        let mut idx = 0;
                        #(
                            if #field_stages.contains(mev::ShaderStages::VERTEX) {
                                #field_argument_impls::bind_vertex_argument(&self.#field_names, group, idx, vertex_bindings, metal);
                            }

                            if #field_stages.contains(mev::ShaderStages::FRAGMENT) {
                                #field_argument_impls::bind_fragment_argument(&self.#field_names, group, idx, fragment_bindings, metal);
                            }

                            idx += 1;
                        )*
                    }

                    #[inline(always)]
                    fn bind_compute(&self, group: u32, encoder: &mut mev::ComputeCommandEncoder) {
                        let metal = encoder.metal();
                        let bindings = encoder.bindings();

                        let mut idx = 0;
                        #(
                            if #field_stages.contains(mev::ShaderStages::COMPUTE) {
                                #field_argument_impls::bind_compute_argument(&self.#field_names, group, idx, bindings, metal);
                            }

                            idx += 1;
                        )*
                    }
                }
            })
        }
    }
}
