use std::fmt;
use web_sys::WebGl2RenderingContext;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Device {
    gl: WebGl2RenderingContext,
}

unsafe impl Sync for Device {}
unsafe impl Send for Device {}

impl Device {
    pub fn new(gl: WebGl2RenderingContext) -> Self {
        Device { gl }
    }

    pub fn context(&self) -> &WebGl2RenderingContext {
        &self.gl
    }
}

impl crate::traits::Resource for Device {}

#[hidden_trait::expose]
impl crate::traits::Device for Device {
    fn new_buffer(&self, desc: crate::BufferDesc) -> crate::backend::Buffer {
        self.gl.create_buffer();

        todo!()
    }

    fn new_buffer_init(&self, desc: crate::BufferInitDesc) -> crate::backend::Buffer {
        todo!()
    }

    fn new_image(&self, desc: crate::ImageDesc) -> crate::backend::Image {
        todo!()
    }

    fn new_sampler(&self, desc: crate::SamplerDesc) -> crate::backend::Sampler {
        todo!()
    }

    fn new_compute_pipeline(
        &self,
        desc: crate::ComputePipelineDesc,
    ) -> Result<crate::backend::ComputePipeline, crate::PipelineError> {
        unimplemented!("WebGL backend does not support compute pipelines")
    }

    fn new_render_pipeline(
        &self,
        desc: crate::RenderPipelineDesc,
    ) -> Result<crate::backend::RenderPipeline, crate::PipelineError> {
        todo!()
    }

    fn new_shader_library(
        &self,
        desc: crate::LibraryDesc,
    ) -> Result<crate::backend::Library, crate::ShaderLibraryError> {
        todo!()
    }

    fn new_surface(
        &self,
        window: &impl raw_window_handle::HasWindowHandle,
        display: &impl raw_window_handle::HasDisplayHandle,
    ) -> Result<crate::backend::Surface, crate::SurfaceError> {
        unimplemented!("WebGL backend does not support creating surfaces after device creation")
    }

    fn new_blas(&self, desc: crate::BlasDesc) -> crate::backend::Blas {
        unimplemented!("WebGL backend does not support BLAS")
    }

    fn new_tlas(&self, desc: crate::TlasDesc) -> crate::backend::Tlas {
        unimplemented!("WebGL backend does not support TLAS")
    }

    fn wait_idle(&self) -> Result<(), crate::DeviceError> {
        self.context.client_wait_sync()
    }
}
