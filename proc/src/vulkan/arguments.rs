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
    let vis = &input.vis;

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

    let field_bindings = (0..data.fields.len() as u32).collect::<Vec<_>>();
    let fields_count = data.fields.len();

    let update_name = quote::format_ident!("MevGenerated{}Update", name);

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
                #[doc(hidden)]
                #[derive(Clone, Copy)]
                #vis struct #update_name {
                    #(#field_names: #field_argument_impls::Update,)*
                }

                impl #name {
                    const fn mev_generated_template_entries() -> [mev::for_macro::DescriptorUpdateTemplateEntry; #fields_count] {
                        let update = mev::for_macro::MaybeUninit::<#update_name>::uninit();
                        let ptr = update.as_ptr();
                        [
                            #(
                                mev::for_macro::DescriptorUpdateTemplateEntry {
                                    dst_binding: #field_bindings,
                                    dst_array_element: 0,
                                    descriptor_count: {
                                        if #field_argument_impls::SIZE > u32::MAX as usize {
                                            panic!("Too many elements in the descriptor array");
                                        }
                                        #field_argument_impls::SIZE as u32
                                    },
                                    descriptor_type: mev::for_macro::descriptor_type(#field_argument_impls::KIND),
                                    offset: unsafe { mev::for_macro::addr_of!((*ptr).#field_names).cast::<u8>().offset_from(ptr.cast::<u8>()) as usize },
                                    stride: #field_argument_impls::STRIDE,
                                },
                            )*
                        ]
                    }
                }

                impl mev::for_macro::Arguments for #name {
                    const LAYOUT: mev::ArgumentGroupLayout<'static> = mev::ArgumentGroupLayout {
                        arguments: &[#(mev::ArgumentLayout {
                            kind: #field_argument_impls::KIND,
                            size: #field_argument_impls::SIZE,
                            stages: #field_stages,
                        },)*],
                    };

                    type Update = #update_name;

                    #[inline(always)]
                    fn template_entries() -> &'static [mev::for_macro::DescriptorUpdateTemplateEntry] {
                        static ENTRIES: [mev::for_macro::DescriptorUpdateTemplateEntry; #fields_count] = #name::mev_generated_template_entries();
                        &ENTRIES
                    }

                    #[inline(always)]
                    fn update(&self) -> #update_name {
                        #update_name {
                            #(#field_names: #field_argument_impls::update(&self.#field_names),)*
                        }
                    }

                    #[inline(always)]
                    fn add_refs(&self, refs: &mut mev::for_macro::Refs) {
                        #(#field_argument_impls::add_refs(&self.#field_names, refs);)*
                    }
                }
            })
        }
    }
}
