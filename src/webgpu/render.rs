use std::{fmt, rc::Rc};

use smallvec::SmallVec;

use crate::{
    backend::from::{IntoWgpu, WgpuInto},
    generic::{ArgumentGroupLayout, ArgumentKind, ArgumentLayout, ShaderStages},
};

#[derive(Clone)]
pub struct RenderPipeline {
    pipeline: wgpu::RenderPipeline,
    layouts: Rc<[wgpu::BindGroupLayout]>,
}

impl RenderPipeline {
    pub(super) fn new(
        pipeline: wgpu::RenderPipeline,
        layouts: Rc<[wgpu::BindGroupLayout]>,
    ) -> Self {
        RenderPipeline { pipeline, layouts }
    }

    pub(super) fn wgpu(&self) -> &wgpu::RenderPipeline {
        &self.pipeline
    }

    pub(super) fn bind_group_layout(&self, index: usize) -> &wgpu::BindGroupLayout {
        &self.layouts[index]
    }
}

impl fmt::Debug for RenderPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("RenderPipeline").finish()
    }
}

impl crate::traits::Resource for RenderPipeline {}
