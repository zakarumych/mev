use std::{fmt, rc::Rc};

#[derive(Clone)]
pub struct ComputePipeline {
    pipeline: wgpu::ComputePipeline,
    layouts: Rc<[wgpu::BindGroupLayout]>,
}

impl ComputePipeline {
    pub(super) fn new(
        pipeline: wgpu::ComputePipeline,
        layouts: Rc<[wgpu::BindGroupLayout]>,
    ) -> Self {
        ComputePipeline { pipeline, layouts }
    }

    pub(super) fn wgpu(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }

    pub(super) fn bind_group_layout(&self, index: usize) -> &wgpu::BindGroupLayout {
        &self.layouts[index]
    }
}

impl fmt::Debug for ComputePipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ComputePipeline").finish()
    }
}

impl crate::traits::Resource for ComputePipeline {}
