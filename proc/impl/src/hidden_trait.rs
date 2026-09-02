use proc_macro2::{Span, TokenStream};
use quote::{ToTokens, format_ident, quote_spanned};
use syn::spanned::Spanned;

/// Turns `impl Trait for Type { ... }` into
///
/// ```ignore
/// impl Type {
///     // all items, made `pub`
/// }
///
/// impl Trait for Type {
///     // all items, reexported from the inherent impl above
/// }
/// ```
pub fn expand(item_impl: syn::ItemImpl) -> syn::Result<TokenStream> {
    let syn::ItemImpl {
        attrs,
        defaultness,
        unsafety,
        impl_token,
        generics,
        trait_,
        self_ty,
        brace_token,
        items,
    } = item_impl;

    let Some((not_token, trait_path, for_token)) = trait_ else {
        return Err(syn::Error::new_spanned(
            impl_token,
            "`#[hidden_trait]` can only be applied to trait impls: `impl Trait for Type { .. }`",
        ));
    };

    if let Some(not_token) = not_token {
        return Err(syn::Error::new_spanned(
            not_token,
            "`#[hidden_trait]` does not support negative impls",
        ));
    }

    let mut inherent_items = Vec::with_capacity(items.len());
    let mut trait_items = Vec::with_capacity(items.len());

    for item in items {
        match item {
            syn::ImplItem::Const(item_const) => {
                expand_const(item_const, &self_ty, &mut inherent_items, &mut trait_items);
            }
            syn::ImplItem::Type(item_type) => {
                expand_type(item_type, &self_ty, &mut inherent_items, &mut trait_items);
            }
            syn::ImplItem::Fn(item_fn) => {
                expand_fn(item_fn, &self_ty, &mut inherent_items, &mut trait_items);
            }
            other => {
                return Err(syn::Error::new_spanned(
                    other,
                    "`#[hidden_trait]` does not know how to reexport this item kind",
                ));
            }
        }
    }

    let inherent_impl = syn::ItemImpl {
        attrs: attrs.clone(),
        defaultness,
        // Inherent impls can never be `unsafe`, only trait impls can.
        unsafety: None,
        impl_token,
        generics: generics.clone(),
        trait_: None,
        self_ty: self_ty.clone(),
        brace_token,
        items: inherent_items,
    };

    let trait_impl = syn::ItemImpl {
        attrs,
        defaultness,
        unsafety,
        impl_token,
        generics,
        trait_: Some((None, trait_path, for_token)),
        self_ty,
        brace_token,
        items: trait_items,
    };

    let mut tokens = TokenStream::new();
    inherent_impl.to_tokens(&mut tokens);
    trait_impl.to_tokens(&mut tokens);
    Ok(tokens)
}

/// Builds a `pub` visibility that points at `span`, preserving diagnostics
/// locality instead of collapsing everything to the macro call site.
fn pub_vis(span: Span) -> syn::Visibility {
    syn::Visibility::Public(syn::token::Pub(span))
}

fn expand_const(
    item_const: syn::ImplItemConst,
    self_ty: &syn::Type,
    inherent_items: &mut Vec<syn::ImplItem>,
    trait_items: &mut Vec<syn::ImplItem>,
) {
    let span = item_const.span();
    let ident = item_const.ident.clone();

    let reexport_expr: syn::Expr = syn::parse2(quote_spanned! {span=>
        <#self_ty>::#ident
    })
    .expect("generated const reexport expression must parse");

    let trait_item = syn::ImplItemConst {
        attrs: Vec::new(),
        vis: syn::Visibility::Inherited,
        defaultness: None,
        const_token: item_const.const_token,
        ident: ident.clone(),
        generics: item_const.generics.clone(),
        colon_token: item_const.colon_token,
        ty: item_const.ty.clone(),
        eq_token: item_const.eq_token,
        expr: reexport_expr,
        semi_token: item_const.semi_token,
    };
    trait_items.push(syn::ImplItem::Const(trait_item));

    let mut inherent_const = item_const;
    inherent_const.vis = pub_vis(inherent_const.const_token.span);
    inherent_items.push(syn::ImplItem::Const(inherent_const));
}

fn expand_type(
    item_type: syn::ImplItemType,
    self_ty: &syn::Type,
    inherent_items: &mut Vec<syn::ImplItem>,
    trait_items: &mut Vec<syn::ImplItem>,
) {
    let span = item_type.span();
    let ident = item_type.ident.clone();
    let (_, ty_generics, _) = item_type.generics.split_for_impl();

    // Associated types may carry their own generic parameters (GATs).
    // Reference them positionally so `type Bar<T> = <Type>::Bar<T>;`.
    let reexport_ty: syn::Type = syn::parse2(quote_spanned! {span=>
        <#self_ty>::#ident #ty_generics
    })
    .expect("generated type reexport must parse");

    let trait_item = syn::ImplItemType {
        attrs: Vec::new(),
        vis: syn::Visibility::Inherited,
        defaultness: None,
        type_token: item_type.type_token,
        ident: ident.clone(),
        generics: item_type.generics.clone(),
        eq_token: item_type.eq_token,
        ty: reexport_ty,
        semi_token: item_type.semi_token,
    };
    trait_items.push(syn::ImplItem::Type(trait_item));

    let mut inherent_type = item_type;
    inherent_type.vis = pub_vis(inherent_type.type_token.span);
    inherent_items.push(syn::ImplItem::Type(inherent_type));
}

fn expand_fn(
    item_fn: syn::ImplItemFn,
    self_ty: &syn::Type,
    inherent_items: &mut Vec<syn::ImplItem>,
    trait_items: &mut Vec<syn::ImplItem>,
) {
    let span = item_fn.span();
    let mut sig = item_fn.sig.clone();
    let ident = sig.ident.clone();

    // Rename every non-receiver parameter to a fresh identifier so we can
    // forward the call regardless of how intricate the original pattern is
    // (tuples, `_`, references, etc.), then call with those fresh names in
    // the same order.
    let mut call_args: Vec<TokenStream> = Vec::new();
    for input in sig.inputs.iter_mut() {
        match input {
            syn::FnArg::Receiver(_) => {
                call_args.push(quote_spanned!(span=> self));
            }
            syn::FnArg::Typed(pat_type) => {
                let arg_span = pat_type.pat.span();
                let fresh = format_ident!("__arg{}", call_args.len(), span = arg_span);
                call_args.push(quote_spanned!(arg_span=> #fresh));
                *pat_type.pat = syn::Pat::Ident(syn::PatIdent {
                    attrs: Vec::new(),
                    by_ref: None,
                    mutability: None,
                    ident: fresh,
                    subpat: None,
                });
            }
        }
    }

    // Forward the method's own generic parameters explicitly so type
    // inference never has to guess them back from the call, e.g. when a
    // generic parameter only appears in the return type.
    let turbofish_args: Vec<TokenStream> = sig
        .generics
        .params
        .iter()
        .filter_map(|param| match param {
            syn::GenericParam::Type(t) => {
                let ident = &t.ident;
                Some(quote_spanned!(span=> #ident))
            }
            syn::GenericParam::Const(c) => {
                let ident = &c.ident;
                Some(quote_spanned!(span=> #ident))
            }
            syn::GenericParam::Lifetime(_) => None,
        })
        .collect();

    let turbofish = if turbofish_args.is_empty() {
        TokenStream::new()
    } else {
        quote_spanned!(span=> ::<#(#turbofish_args),*>)
    };

    let mut call_expr: syn::Expr = syn::parse2(quote_spanned! {span=>
        <#self_ty>::#ident #turbofish (#(#call_args),*)
    })
    .expect("generated call expression must parse");

    if sig.asyncness.is_some() {
        call_expr = syn::parse2(quote_spanned! {span=> #call_expr.await })
            .expect("generated await expression must parse");
    }

    if sig.unsafety.is_some() {
        call_expr = syn::parse2(quote_spanned! {span=> unsafe { #call_expr } })
            .expect("generated unsafe block must parse");
    }

    let block: syn::Block =
        syn::parse2(quote_spanned! {span=> { #call_expr } }).expect("generated fn body must parse");

    let trait_item = syn::ImplItemFn {
        attrs: item_fn.attrs.clone(),
        vis: syn::Visibility::Inherited,
        defaultness: item_fn.defaultness,
        sig,
        block,
    };
    trait_items.push(syn::ImplItem::Fn(trait_item));

    let mut inherent_fn = item_fn;
    inherent_fn.vis = pub_vis(inherent_fn.sig.fn_token.span);
    inherent_items.push(syn::ImplItem::Fn(inherent_fn));
}
