use crate::{generic::Shader, ArgumentGroupLayout};

/// Compute pipeline descriptor.
/// Used to create new compute pipelines.
pub struct ComputePipelineDesc<'a> {
    /// Name of the compute pipeline.
    /// It can be used for debugging purposes.
    pub name: &'a str,

    /// Compute shader.
    pub shader: Shader<'a>,

    /// Size of the work group.
    pub work_group_size: [u32; 3],

    /// Size in bytes of constants in the pipeline.
    pub constants: usize,

    /// Arguments in the pipeline.
    pub arguments: &'a [ArgumentGroupLayout<'a>],
}
