proc_easy::easy_flags! {
    pub Kind(kind) {
        // Constant(constant),
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
