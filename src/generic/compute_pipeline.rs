use crate::{generic::Shader, ArgumentGroupLayout};

pub struct ComputePipelineDesc<'a> {
    pub name: &'a str,
    pub shader: Shader<'a>,
    pub work_group_size: [u32; 3],
    pub constants: usize,
    pub arguments: &'a [ArgumentGroupLayout<'a>],
}
