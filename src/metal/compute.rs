use std::sync::Arc;

use super::shader::Bindings;

#[derive(Clone)]
pub struct ComputePipeline {
    state: metal::ComputePipelineState,
    bindings: Option<Arc<Bindings>>,
    workgroup_size: Option<[u32; 3]>,
}

unsafe impl Send for ComputePipeline {}
unsafe impl Sync for ComputePipeline {}

impl ComputePipeline {
    #[inline]
    pub(super) fn new(
        state: metal::ComputePipelineState,
        bindings: Option<Arc<Bindings>>,
        workgroup_size: Option<[u32; 3]>,
    ) -> Self {
        ComputePipeline {
            state,
            bindings,
            workgroup_size,
        }
    }

    #[inline(always)]
    pub(super) fn metal(&self) -> &metal::ComputePipelineState {
        &self.state
    }

    #[inline(always)]
    pub(super) fn bindings(&self) -> Option<Arc<Bindings>> {
        self.bindings.clone()
    }

    #[inline(always)]
    pub(super) fn workgroup_size(&self) -> Option<[u32; 3]> {
        self.workgroup_size
    }
}
