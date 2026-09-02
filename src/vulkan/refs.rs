use super::{Buffer, CommandBuffer, ComputePipeline, Image, RenderPipeline, Sampler};

/// Stores references to vulkan objects
/// to keep them alive.
#[derive(Default)]
pub struct Refs {
    buffers: Vec<Buffer>,
    images: Vec<Image>,
    samplers: Vec<Sampler>,
    render_pipelines: Vec<RenderPipeline>,
    compute_pipelines: Vec<ComputePipeline>,
}

impl Refs {
    pub fn new() -> Self {
        Refs::default()
    }

    pub fn clear(&mut self) {
        // FIXME: Clear buffers // self.buffers.clear();
        self.images.clear();
        self.samplers.clear();
        self.render_pipelines.clear();
    }

    pub fn add_buffer(&mut self, buffer: Buffer) {
        self.buffers.push(buffer);
    }

    pub fn add_buffers(&mut self, buffers: &[Buffer]) {
        self.buffers.extend_from_slice(buffers);
    }

    pub fn add_image(&mut self, image: Image) {
        self.images.push(image);
    }

    pub fn add_images(&mut self, images: &[Image]) {
        self.images.extend_from_slice(images);
    }

    pub fn add_sampler(&mut self, sampler: Sampler) {
        self.samplers.push(sampler);
    }

    pub fn add_samplers(&mut self, samplers: &[Sampler]) {
        self.samplers.extend_from_slice(samplers);
    }

    pub fn add_render_pipeline(&mut self, pipeline: RenderPipeline) {
        self.render_pipelines.push(pipeline);
    }

    pub fn add_compute_pipeline(&mut self, pipeline: ComputePipeline) {
        self.compute_pipelines.push(pipeline);
    }

    // pub fn add_cbuf(&mut self, cbuf: CommandBuffer) {
    //     self.cbufs.push(cbuf);
    // }

    // pub fn add_refs(&mut self, refs: Refs) {
    //     self.refs.push(refs);
    // }
}
