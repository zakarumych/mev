use std::{error::Error, fmt, sync::Arc};

use ash::vk;
use ash::vk::Handle;

use super::{device::WeakDevice, layout::PipelineLayout, shader::Library};

struct Inner {
    owner: WeakDevice,
    layout: PipelineLayout,
    idx: usize,
    vertex_library: Library,
    fragment_library: Option<Library>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.owner.drop_pipeline(self.idx);
    }
}

#[derive(Clone)]
pub struct RenderPipeline {
    handle: vk::Pipeline,
    layout: vk::PipelineLayout,
    inner: Arc<Inner>,
}

impl RenderPipeline {
    pub(super) fn new(
        owner: WeakDevice,
        handle: vk::Pipeline,
        idx: usize,
        layout: PipelineLayout,
        vertex_library: Library,
        fragment_library: Option<Library>,
    ) -> Self {
        RenderPipeline {
            handle,
            layout: layout.handle(),
            inner: Arc::new(Inner {
                owner,
                layout,
                idx,
                vertex_library,
                fragment_library,
            }),
        }
    }

    /// Creates a null/invalid RenderPipeline for use when device OOM occurs.
    pub(super) fn null() -> Self {
        RenderPipeline {
            handle: vk::Pipeline::null(),
            layout: vk::PipelineLayout::null(),
            inner: Arc::new(Inner {
                owner: WeakDevice::null(),
                layout: PipelineLayout::null(),
                idx: 0,
                vertex_library: Library::null(),
                fragment_library: None,
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
