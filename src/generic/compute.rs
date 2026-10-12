use crate::{ArgumentGroupLayout, generic::Shader};

#[derive(Clone, Copy, Debug)]
pub(crate) struct ComputeKernelLimits {
    pub max_workgroup_size: u32,
    pub max_workgroup_memory: usize,
    pub max_storage_buffers: u32,
    pub max_uniform_buffers: u32,
    pub max_buffer_bindings: u32,
    pub max_storage_buffer_range: usize,
    pub max_uniform_buffer_range: usize,
    pub storage_offset_alignment: usize,
    pub uniform_offset_alignment: usize,
    pub max_dispatch_count: u32,
    pub max_subgroups_per_workgroup: Option<u32>,
    pub subgroup_arithmetic: bool,
    pub shader_f16: bool,
}

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
