use proc_macro::TokenStream;

#[proc_macro_derive(Arguments, attributes(mev))]
pub fn arguments_derive(input: TokenStream) -> TokenStream {
    mev_proc_impl::arguments_derive(input.into(), &quote::quote!(mev)).into()
}

#[proc_macro_derive(DeviceRepr, attributes(mev))]
pub fn repr_derive(input: TokenStream) -> TokenStream {
    mev_proc_impl::repr_derive(input.into(), &quote::quote!(mev)).into()
}

#[proc_macro]
pub fn match_backend(input: TokenStream) -> TokenStream {
    mev_proc_impl::match_backend(input.into(), &quote::quote!(mev)).into()
}
