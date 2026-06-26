use proc_easy::EasyAttributes;
use proc_macro2::TokenStream;
use quote::quote;
use syn::{self, spanned::Spanned};

proc_easy::easy_token!(vertex);
proc_easy::easy_token!(constant);
proc_easy::easy_token!(instance);
proc_easy::easy_token!(skip);

proc_easy::easy_argument_value!(
    struct InstanceRate {
        instance: instance,
        rate: syn::LitInt,
    }
);

proc_easy::easy_argument_group! {
    enum StepMode {
        Vertex(vertex),
        Constant(constant),
        Instance(InstanceRate),
    }
}

proc_easy::easy_attributes! {
    @(mev)
    struct TypeAttributes {
        pub step_mode: Option<StepMode>,
    }
}

proc_easy::easy_attributes! {
    @(mev)
    pub struct FieldAttributes {
        pub skip: Option<skip>,
    }
}

pub fn derive(input: &syn::DeriveInput, mev: &TokenStream) -> syn::Result<TokenStream> {
    let name = &input.ident;
    let vis = &input.vis;

    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.generics,
            "generic arguments are not supported by `#[derive(VertexBinding)]`",
        ));
    }

    let data = match &input.data {
        syn::Data::Struct(data) => data,
        _ => {
            return Err(syn::Error::new_spanned(
                input,
                "only structs are supported by `#[derive(VertexBinding)]`",
            ))
        }
    };

    let type_attrs = TypeAttributes::parse(&input.attrs, input.span())?;

    let step_mode = match type_attrs.step_mode {
        Some(StepMode::Vertex(_)) | None => quote! { #mev::for_macro::VertexStepMode::Vertex },
        Some(StepMode::Constant(_)) => quote! { #mev::for_macro::VertexStepMode::Constant },
        Some(StepMode::Instance(rate)) => {
            let rate = rate.rate;
            quote! { #mev::for_macro::VertexStepMode::Instance { rate: #rate } }
        }
    };

    let name_descs = quote::format_ident!("__MevGenerated{}AttributeDescs", name);

    let field_attrs = data
        .fields
        .iter()
        .map(|field| FieldAttributes::parse(&field.attrs, field.span()))
        .collect::<Result<Vec<_>, _>>()?;

    let field_types = data
        .fields
        .iter()
        .zip(field_attrs.iter())
        .filter_map(|(field, attr)| {
            if attr.skip.is_none() {
                Some(&field.ty)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    match data.fields {
        syn::Fields::Named(_) => {
            let field_names = data
                .fields
                .iter()
                .zip(field_attrs.iter())
                .filter_map(|(field, attr)| {
                    if attr.skip.is_none() {
                        Some(&field.ident)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();

            let tokens = quote::quote! {
                #[allow(unused, non_camel_case_types)]
                #[doc(hidden)]
                #[derive(Clone, Copy, Debug)]
                #vis struct #name_descs {
                    #(
                        #[allow(non_snake_case)]
                        #field_names: #mev::for_macro::VertexAttributeDescs<#field_types>,
                    )*
                }

                impl #mev::for_macro::VertexBinding for #name {
                    const LAYOUT: #mev::for_macro::VertexLayoutDesc = #mev::for_macro::VertexLayoutDesc {
                        stride: std::mem::size_of::<Self>() as u32,
                        step_mode: #step_mode,
                    };

                    type AttributeDescs = #name_descs;

                    fn descs(buffer_index: u32) -> #name_descs {
                        #name_descs {
                            #(
                                #field_names: #mev::for_macro::VertexAttributeDescs::new(buffer_index, #mev::for_macro::offset_of!(Self, #field_names)),
                            )*
                        }
                    }
                }
            };

            Ok(tokens)
        }
        _ => unimplemented!("only named fields are supported by `#[derive(VertexBinding)]`"),
    }
}
