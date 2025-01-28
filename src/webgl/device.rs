use std::fmt;
use web_sys::WebGl2RenderingContext;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Device {
    context: WebGl2RenderingContext,
}

unsafe impl Sync for Device {}
unsafe impl Send for Device {}

impl Device {
    pub fn new(context: WebGl2RenderingContext) -> Self {
        Device { context }
    }

    pub fn context(&self) -> &WebGl2RenderingContext {
        &self.context
    }
}

impl crate::traits::Resource for Device {}

// #[hidden_trait::expose]
impl crate::traits::Device for Device {
    fn new_buffer(
        &self,
        desc: crate::BufferDesc,
    ) -> Result<crate::backend::Buffer, crate::OutOfMemory> {
        todo!()
    }

    fn new_buffer_init(
        &self,
        desc: crate::BufferInitDesc,
    ) -> Result<crate::backend::Buffer, crate::OutOfMemory> {
        todo!()
    }

    fn new_image(
        &self,
        desc: crate::ImageDesc,
    ) -> Result<crate::backend::Image, crate::OutOfMemory> {
        todo!()
    }

    fn new_compute_pipeline(
        &self,
        desc: crate::ComputePipelineDesc,
    ) -> Result<crate::backend::ComputePipeline, crate::CreatePipelineError> {
        unimplemented!("WebGL backend does not support compute pipelines")
    }

    fn new_render_pipeline(
        &self,
        desc: crate::RenderPipelineDesc,
    ) -> Result<crate::backend::RenderPipeline, crate::CreatePipelineError> {
        todo!()
    }

    fn new_sampler(
        &self,
        desc: crate::SamplerDesc,
    ) -> Result<crate::backend::Sampler, crate::OutOfMemory> {
        todo!()
    }

    fn new_shader_library(
        &self,
        desc: crate::LibraryDesc,
    ) -> Result<crate::backend::Library, crate::CreateLibraryError> {
        todo!()
    }

    fn new_surface(
        &self,
        window: &impl raw_window_handle::HasWindowHandle,
        display: &impl raw_window_handle::HasDisplayHandle,
    ) -> Result<crate::backend::Surface, crate::SurfaceError> {
        unimplemented!("WebGL backend does not support creating surfaces after device creation")
    }

    fn new_blas(&self, desc: crate::BlasDesc) -> Result<crate::backend::Blas, crate::OutOfMemory> {
        unimplemented!("WebGL backend does not support BLAS")
    }

    fn new_tlas(&self, desc: crate::TlasDesc) -> Result<crate::backend::Tlas, crate::OutOfMemory> {
        unimplemented!("WebGL backend does not support TLAS")
    }

    fn wait_idle(&self) -> Result<(), crate::OutOfMemory> {
        self.context.client_wait_sync()
    }
}
