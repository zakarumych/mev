proc_easy::easy_token!(uniform);
proc_easy::easy_token!(sampled);
proc_easy::easy_token!(storage);

proc_easy::easy_argument_group! {
    pub enum Kind {
        Uniform(uniform),
        Sampled(sampled),
        Storage(storage),
    }
}

proc_easy::easy_flags! {
    pub Shader(shader) | pub Shaders(shaders) {
        Vertex(vertex),
        Fragment(fragment),
        Compute(compute),
    }
}

proc_easy::easy_attributes! {
    @(mev)
    pub struct FieldAttributes {
        pub kind: Option<Kind>,
        pub shaders: Shaders,
    }
}

/// Checks that every field is visible to at least one shader stage.
pub fn check_field_shaders(
    fields: &syn::Fields,
    field_attrs: &[FieldAttributes],
) -> syn::Result<()> {
    for (field, attrs) in fields.iter().zip(field_attrs) {
        if attrs.shaders.flags.is_empty() {
            return Err(syn::Error::new_spanned(
                field,
                "argument must be visible to at least one shader stage: add #[mev(vertex)], #[mev(fragment)] and/or #[mev(compute)]",
            ));
        }
    }
    Ok(())
}
