use proc_macro2::TokenStream;
use syn::{
    parse::{ParseStream, Parser},
    spanned::Spanned,
};

proc_easy::easy_token!(metal);
proc_easy::easy_token!(vulkan);
// proc_easy::easy_token!(webgl);
proc_easy::easy_token!(webgpu);

proc_easy::easy_parse! {
    enum Pattern {
        Metal(metal),
        Vulkan(vulkan),
        // WebGL(webgl),
        WebGPU(webgpu),
        Wildcard(syn::Token![_]),
    }
}

proc_easy::easy_parse! {
    struct Arm {
        pipe: Option<syn::Token![|]>,
        patterns: proc_easy::EasySeparated<Pattern, syn::Token![|]>,
        fat_arrow: syn::Token![=>],
        body: proc_easy::EasyBraced<proc_macro2::TokenStream>,
    }
}

fn parse_arms(input: ParseStream) -> syn::Result<Vec<Arm>> {
    let mut arms = Vec::new();
    while !input.is_empty() {
        let arm = input.parse::<Arm>()?;
        arms.push(arm);
    }
    Ok(arms)
}

pub fn match_backend(tokens: TokenStream, mev: &TokenStream) -> TokenStream {
    match parse_arms.parse2(tokens) {
        Ok(arms) => {
            let mut metal_matched = false;
            let mut vulkan_matched = false;
            // let mut webgl_matched = false;
            let mut webgpu_matched = false;
            let mut wildcard_matched = false;

            let mut result = proc_macro2::TokenStream::new();
            for (idx, arm) in arms.iter().enumerate() {
                for pattern in &*arm.patterns {
                    match pattern {
                        Pattern::Metal(metal) => {
                            if !metal_matched {
                                metal_matched = true;

                                if !wildcard_matched {
                                    let body = &arm.body.0;
                                    result.extend(quote::quote_spanned! { body.span() => #mev::with_metal! { #body } });
                                } else {
                                    result.extend(quote::quote_spanned! { metal.span() => ::core::compile_error!("`metal` matched after wildcard");  });
                                }
                            } else {
                                result.extend(quote::quote_spanned! { metal.span() => ::core::compile_error!("`metal` matched more than once");  });
                            }
                        }
                        Pattern::Vulkan(vulkan) => {
                            if !vulkan_matched {
                                vulkan_matched = true;

                                if !wildcard_matched {
                                    let body = &arm.body.0;
                                    result.extend(quote::quote_spanned! { body.span() => #mev::with_vulkan! { #body } });
                                } else {
                                    result.extend(quote::quote_spanned! { vulkan.span() => ::core::compile_error!("`vulkan` matched after wildcard");  });
                                }
                            } else {
                                result.extend(quote::quote_spanned! { vulkan.span() => ::core::compile_error!("`vulkan` matched more than once");  });
                            }
                        }
                        // Pattern::WebGL(webgl) => {
                        //     if !webgl_matched {
                        //         webgl_matched = true;

                        //         if !wildcard_matched {
                        //             let body = &arm.body.0;
                        //             result.extend(quote::quote_spanned! { body.span() => #mev::with_webgl! { #body } });
                        //         } else {
                        //             result.extend(quote::quote_spanned! { webgl.span() => ::core::compile_error!("`webgl` matched after wildcard");  });
                        //         }
                        //     } else {
                        //         result.extend(quote::quote_spanned! { webgl.span() => ::core::compile_error!("`webgl` matched more than once");  });
                        //     }
                        // }
                        Pattern::WebGPU(webgpu) => {
                            if !webgpu_matched {
                                webgpu_matched = true;

                                if !wildcard_matched {
                                    let body = &arm.body.0;
                                    result.extend(quote::quote_spanned! { body.span() => #mev::with_webgpu! { #body } });
                                } else {
                                    result.extend(quote::quote_spanned! { webgpu.span() => ::core::compile_error!("`webgpu` matched after wildcard");  });
                                }
                            } else {
                                result.extend(quote::quote_spanned! { webgpu.span() => ::core::compile_error!("`webgpu` matched more than once");  });
                            }
                        }
                        Pattern::Wildcard(wildcard) => {
                            if idx != arms.len() - 1 {
                                result.extend(quote::quote_spanned! { wildcard.span() => ::core::compile_error!("Wildcard pattern must appear last"); });
                            }

                            if wildcard_matched
                                || (vulkan_matched && metal_matched  /*&& webgl_matched*/ && webgpu_matched)
                            {
                                result.extend(quote::quote_spanned! { wildcard.span() => ::core::compile_error!("Wildcard pattern is redundant"); });
                            }

                            if !wildcard_matched {
                                wildcard_matched = true;

                                let body = &arm.body.0;

                                if !metal_matched {
                                    result
                                            .extend(quote::quote_spanned! { body.span() => #mev::with_metal! { #body } });
                                }

                                if !vulkan_matched {
                                    result
                                            .extend(quote::quote_spanned! { body.span() => #mev::with_vulkan! { #body } });
                                }

                                //     if !webgl_matched {
                                //         result.extend(quote::quote_spanned! { body.span() => #mev::with_webgl! { #body } });
                                //     }
                                //

                                if !webgpu_matched {
                                    result
                                            .extend(quote::quote_spanned! { body.span() => #mev::with_webgpu! { #body } });
                                }
                            }
                        }
                    }
                }
            }

            result.into()
        }
        Err(err) => err.to_compile_error().into(),
    }
}
