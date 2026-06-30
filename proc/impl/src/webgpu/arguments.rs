use proc_easy::{private::Spanned, EasyAttributes};
use proc_macro2::TokenStream;
use quote::quote_spanned;
use syn;

use crate::args::*;

pub fn derive(
    input: &syn::DeriveInput,
    mev: &TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    let name = &input.ident;

    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.generics,
            "generic arguments are not supported by `#[derive(Arguments)]`",
        ));
    }

    let data = match &input.data {
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
                None => quote::quote!(<#ty as #mev::for_macro::ArgumentsField<#mev::for_macro::Automatic>>),
                Some(Kind::Uniform(_)) => {
                    quote::quote!(<#ty as #mev::for_macro::ArgumentsField<#mev::for_macro::Uniform>>)
                }
                Some(Kind::Sampled(_)) => {
                    quote::quote!(<#ty as #mev::for_macro::ArgumentsField<#mev::for_macro::Sampled>>)
                }
                Some(Kind::Storage(_)) => {
                    quote::quote!(<#ty as #mev::for_macro::ArgumentsField<#mev::for_macro::Storage>>)
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
                quote_spanned!(field.span() => #mev::ShaderStages::empty())
            } else {
                let mut tokens = quote::quote!(0);
                for stage in attrs.shaders.flags.iter() {
                    match stage {
                        Shader::Vertex(v) => tokens.extend(
                            quote_spanned!(v.span() => | #mev::ShaderStages::VERTEX.bits()),
                        ),
                        Shader::Fragment(f) => tokens.extend(
                            quote_spanned!(f.span() => | #mev::ShaderStages::FRAGMENT.bits()),
                        ),
                        Shader::Compute(c) => tokens.extend(
                            quote_spanned!(c.span() => | #mev::ShaderStages::COMPUTE.bits()),
                        ),
                    }
                }
                quote::quote!(#mev::ShaderStages::from_bits_truncate(#tokens))
            }
        })
        .collect::<Vec<_>>();

    match &data.fields {
        syn::Fields::Unit => {
            return Err(syn::Error::new_spanned(
                &data.fields,
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

            let field_indices: Vec<u32> = (0..field_names.len() as u32).collect();

            Ok(quote::quote! {
                impl #mev::for_macro::Arguments for #name {
                    const LAYOUT: #mev::ArgumentGroupLayout<'static> = #mev::ArgumentGroupLayout {
                        arguments: &[#(#mev::ArgumentLayout {
                            kind: #field_argument_impls::KIND,
                            size: #field_argument_impls::SIZE,
                            stages: #field_stages,
                        },)*],
                    };

                    #[inline(always)]
                    fn bind_render(&self, group: u32, encoder: &mut #mev::RenderCommandEncoder) {
                        let device = encoder.wgpu_device();
                        let layout = encoder.bind_group_layout(group);
                        let entries = [#(
                            #mev::for_macro::wgpu::BindGroupEntry {
                                binding: #field_indices,
                                resource: #field_argument_impls::as_binding_resource(&self.#field_names),
                            },
                        )*];
                        let bind_group = device.create_bind_group(&#mev::for_macro::wgpu::BindGroupDescriptor {
                            label: None,
                            layout,
                            entries: &entries,
                        });
                        encoder.set_bind_group(group, &bind_group);
                    }

                    #[inline(always)]
                    fn bind_compute(&self, group: u32, encoder: &mut #mev::ComputeCommandEncoder) {
                        let device = encoder.wgpu_device();
                        let layout = encoder.bind_group_layout(group);
                        let entries = [#(
                            #mev::for_macro::wgpu::BindGroupEntry {
                                binding: #field_indices,
                                resource: #field_argument_impls::as_binding_resource(&self.#field_names),
                            },
                        )*];
                        let bind_group = device.create_bind_group(&#mev::for_macro::wgpu::BindGroupDescriptor {
                            label: None,
                            layout,
                            entries: &entries,
                        });
                        encoder.set_bind_group(group, &bind_group);
                    }
                }
            })
        }
    }
}
