use std::sync::Arc;

use ash::vk;
use ash::vk::Handle;

use super::{device::WeakDevice, layout::PipelineLayout, shader::Library};

struct Inner {
    owner: WeakDevice,
    layout: PipelineLayout,
    idx: usize,
    shader_library: Library,
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.owner.drop_pipeline(self.idx);
    }
}

#[derive(Clone)]
pub struct ComputePipeline {
    handle: vk::Pipeline,
    layout: vk::PipelineLayout,
    inner: Arc<Inner>,
}

impl ComputePipeline {
    pub(super) fn new(
        owner: WeakDevice,
        handle: vk::Pipeline,
        idx: usize,
        layout: PipelineLayout,
        shader_library: Library,
    ) -> Self {
        ComputePipeline {
            handle,
            layout: layout.handle(),
            inner: Arc::new(Inner {
                owner,
                layout,
                idx,
                shader_library,
            }),
        }
    }

    /// Creates a null/invalid ComputePipeline for use when device OOM occurs.
    pub(super) fn null() -> Self {
        ComputePipeline {
            handle: vk::Pipeline::null(),
            layout: vk::PipelineLayout::null(),
            inner: Arc::new(Inner {
                owner: WeakDevice::null(),
                layout: PipelineLayout::null(),
                idx: 0,
                shader_library: Library::null(),
            }),
        }
    }

    pub fn is_null(&self) -> bool {
        self.handle.is_null()
    }

    pub(super) fn handle(&self) -> vk::Pipeline {
        self.handle
    }

    pub(super) fn layout(&self) -> &PipelineLayout {
        &self.inner.layout
    }
}
