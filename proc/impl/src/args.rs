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
