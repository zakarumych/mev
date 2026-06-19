use std::{
    fmt::Debug,
    hash::Hash,
    ops::{Deref, Range},
};

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::{
    generic::{
        Arguments, AsBufferSlice, BlasBuildDesc, BlasDesc, BufferDesc, BufferInitDesc,
        Capabilities, ComputePipelineDesc, CreateError, CreatePipelineError,
        CreateShaderLibraryError, CreateWithSurfaceError, DeviceDesc, DeviceError, DeviceRepr,
        Extent2, Extent3, ImageDesc, ImageExtent, ImageUsage, LibraryDesc, Offset2, Offset3,
        PipelineStages, PixelFormat, RenderPassDesc, RenderPipelineDesc, SamplerDesc, Shader,
        SurfaceError, TlasBuildDesc, TlasDesc, ViewDesc,
    },
    BufferMappedRange, BufferMappedRangeMut,
};

#[cfg(mev_backend = "metal")]
pub trait Resource: Send + Sync + 'static {}

#[cfg(mev_backend = "vulkan")]
pub trait Resource: Send + Sync + 'static {}

#[cfg(mev_backend = "webgl")]
pub trait Resource: 'static {}

pub trait Instance: Debug + Resource {
    fn capabilities(&self) -> &Capabilities;

    fn new_device(
        &self,
        info: DeviceDesc,
    ) -> Result<(crate::backend::Device, Vec<crate::backend::Queue>), CreateError>;

    fn new_device_with_surface(
        &self,
        info: DeviceDesc,
        window: &impl HasWindowHandle,
        display: &impl HasDisplayHandle,
    ) -> Result<
        (
            crate::backend::Device,
            Vec<crate::backend::Queue>,
            crate::backend::Surface,
        ),
        CreateWithSurfaceError,
    > {
        let (device, queues) = self.new_device(info)?;
        let surface = device.new_surface(window, display)?;
        Ok((device, queues, surface))
    }
}

pub trait Device: Clone + Debug + Eq + Resource {
    /// Create a new shader library.
    fn new_shader_library(
        &self,
        desc: LibraryDesc,
    ) -> Result<crate::backend::Library, CreateShaderLibraryError>;

    /// Create a new compute pipeline.
    fn new_compute_pipeline(
        &self,
        desc: ComputePipelineDesc,
    ) -> Result<crate::backend::ComputePipeline, CreatePipelineError>;

    /// Create a new render pipeline.
    fn new_render_pipeline(
        &self,
        desc: RenderPipelineDesc,
    ) -> Result<crate::backend::RenderPipeline, CreatePipelineError>;

    /// Create a new buffer with uninitialized contents.
    fn new_buffer(&self, desc: BufferDesc) -> crate::backend::Buffer;

    /// Create a new buffer and initialize it with the given data.
    fn new_buffer_init(&self, desc: BufferInitDesc) -> crate::backend::Buffer;

    /// Create a new image.
    fn new_image(&self, desc: ImageDesc) -> crate::backend::Image;

    /// Create a new sampler.
    fn new_sampler(&self, desc: SamplerDesc) -> crate::backend::Sampler;

    /// Create a new surface associated with given window.
    fn new_surface(
        &self,
        window: &impl HasWindowHandle,
        display: &impl HasDisplayHandle,
    ) -> Result<crate::backend::Surface, SurfaceError>;

    /// Create a new fake surface associated with image.
    fn new_fake_surface(
        &self,
        image: crate::backend::Image,
    ) -> Result<crate::backend::Surface, SurfaceError>;

    /// Create a new bottom-level acceleration structure.
    fn new_blas(&self, desc: BlasDesc) -> crate::backend::Blas;

    /// Create a new top-level acceleration structure.
    fn new_tlas(&self, desc: TlasDesc) -> crate::backend::Tlas;
}

pub trait Queue: Deref<Target = crate::backend::Device> + Debug + Resource {
    /// Get the device associated with this queue.
    fn device(&self) -> &crate::backend::Device;

    /// Get the queue family index.
    fn family(&self) -> u32;

    /// Create a new command encoder associated with this queue.
    /// The encoder must be submitted to the queue it was created from.
    fn new_command_encoder(&mut self) -> crate::backend::CommandEncoder;

    /// Submit command buffers to the queue.
    ///
    /// If `check_point` is `true`, inserts a checkpoint into queue and check previous checkpoints.
    /// Checkpoints are required for resource reclamation.
    fn submit<I>(&mut self, command_buffers: I, check_point: bool) -> Result<(), DeviceError>
    where
        I: IntoIterator<Item = crate::backend::CommandBuffer>;

    /// Synchronize the access to the frame resources.
    fn sync_frame(&mut self, frame: &mut crate::backend::Frame, before: PipelineStages);

    /// Wait for all operations on the queue to complete.
    fn wait_idle(&self) -> Result<(), DeviceError>;
}

pub trait SyncCommandEncoder {
    /// Synchronizes the access to the resources.
    /// Commands in `before` stages of subsequent commands will be
    /// executed only after commands in `after` stages of previous commands
    /// are finished.
    fn barrier(&mut self, after: PipelineStages, before: PipelineStages);

    /// Synchronizes the access to the image.
    /// Commands in `before` stages of subsequent commands will be
    /// executed only after commands in `after` stages of previous commands
    /// are finished.
    /// Image content is discarded.
    fn init_image(
        &mut self,
        after: PipelineStages,
        before: PipelineStages,
        image: &crate::backend::Image,
    );
}

pub trait CommandEncoder: SyncCommandEncoder {
    /// Presents the frame to the surface.
    fn present(&mut self, frame: crate::backend::Frame, after: PipelineStages);

    /// Finishes encoding and returns the command buffer.
    fn finish(self) -> crate::backend::CommandBuffer;

    /// Returns encoder for copy commands.
    fn copy(&mut self) -> crate::backend::CopyCommandEncoder<'_>;

    fn acceleration_structure(&mut self)
        -> crate::backend::AccelerationStructureCommandEncoder<'_>;

    fn compute(&mut self) -> crate::backend::ComputeCommandEncoder<'_>;

    /// Starts rendering and returns encoder for render commands.
    fn render(&mut self, desc: RenderPassDesc) -> crate::backend::RenderCommandEncoder<'_>;
}

pub trait ComputeCommandEncoder: SyncCommandEncoder {
    /// Sets the current compute pipeline.
    fn with_pipeline(&mut self, pipeline: &crate::backend::ComputePipeline);

    /// Sets arguments group for the current pipeline.
    fn with_arguments(&mut self, group: u32, arguments: &impl Arguments);

    /// Sets constants for the current pipeline.
    fn with_constants(&mut self, constants: &impl DeviceRepr);

    /// Dispatches compute work.
    fn dispatch(&mut self, groups: Extent3);
}

pub trait CopyCommandEncoder: SyncCommandEncoder {
    /// Fills the buffer slice with the given byte.
    fn fill_buffer(&mut self, slice: impl AsBufferSlice, byte: u8);

    /// Writes data to the buffer.
    fn write_buffer_raw(&mut self, slice: impl AsBufferSlice, data: &[u8]);

    /// Writes data to the buffer.
    fn write_buffer(&mut self, slice: impl AsBufferSlice, data: &impl bytemuck::Pod);

    /// Writes data to the buffer.
    fn write_buffer_slice(&mut self, slice: impl AsBufferSlice, data: &[impl bytemuck::Pod]);

    /// Copies bytes from src buffer to dst buffer.
    fn copy_buffer_to_buffer(
        &mut self,
        src: &crate::backend::Buffer,
        src_offset: usize,
        dst: &crate::backend::Buffer,
        dst_offset: usize,
        size: usize,
    );

    /// Copies pixels from src image to dst image.
    fn copy_buffer_to_image(
        &mut self,
        src: &crate::backend::Buffer,
        src_offset: usize,
        bytes_per_line: usize,
        bytes_per_plane: usize,
        dst: &crate::backend::Image,
        dst_offset: Offset3<u32>,
        extent: Extent3<u32>,
        layers: Range<u32>,
        level: u32,
    );

    /// Copies pixels from src image to dst image.
    fn copy_image_region(
        &mut self,
        src: &crate::backend::Image,
        src_level: u32,
        src_base_layer: u32,
        src_offset: Offset3<u32>,
        dst: &crate::backend::Image,
        dst_level: u32,
        dst_base_layer: u32,
        dst_offset: Offset3<u32>,
        extent: Extent3<u32>,
        layers: u32,
    );
}

pub trait RenderCommandEncoder {
    /// Sets the current render pipeline.
    fn with_pipeline(&mut self, pipeline: &crate::backend::RenderPipeline);

    fn with_viewport(&mut self, offset: Offset3<f32>, extent: Extent3<f32>);

    fn with_scissor(&mut self, offset: Offset2<i32>, extent: Extent2<u32>);

    /// Sets arguments group for the current pipeline.
    fn with_arguments(&mut self, group: u32, arguments: &impl Arguments);

    /// Sets constants for the current pipeline.
    fn with_constants(&mut self, constants: &impl DeviceRepr);

    /// Bind vertex buffer to the current pipeline.
    fn bind_vertex_buffers(&mut self, start: u32, slices: &[impl AsBufferSlice]);

    /// Bind index buffer to the current pipeline.
    fn bind_index_buffer(&mut self, slice: impl AsBufferSlice);

    /// Draws primitives.
    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>);

    /// Draws primitives with indices.
    fn draw_indexed(&mut self, vertex_offset: i32, indices: Range<u32>, instances: Range<u32>);
}

pub trait AccelerationStructureCommandEncoder {
    fn build_blas(
        &mut self,
        blas: &crate::backend::Blas,
        desc: BlasBuildDesc,
        scratch: impl AsBufferSlice,
    );

    fn build_tlas(
        &mut self,
        tlas: &crate::backend::Tlas,
        desc: TlasBuildDesc,
        scratch: impl AsBufferSlice,
    );
}

pub trait Surface: Send + Sync + 'static {
    /// Acquires next frame from the surface.
    fn next_frame(&mut self) -> Result<crate::backend::Frame, SurfaceError>;
}

pub trait Frame: Send + Sync + 'static {
    fn image(&self) -> &crate::backend::Image;
}

pub trait Image: Clone + Debug + Eq + Hash + Resource {
    /// Returns the pixel format of the image.
    fn format(&self) -> PixelFormat;

    /// Returns the extent of the image.
    fn extent(&self) -> ImageExtent;

    /// Returns the number of layers in the image.
    fn layers(&self) -> u32;

    /// Returns the number of mip levels in the image.
    fn levels(&self) -> u32;

    /// Returns the usage of the image.
    fn usage(&self) -> ImageUsage;

    /// Returns new image that is a view into this image.
    fn view(&self, device: &crate::backend::Device, desc: ViewDesc) -> crate::backend::Image;

    /// Returns `true` if the image is not shared,
    /// meaning that there are no other references to the image
    /// including references that tracks that GPU may be using the image.
    ///
    /// If this method returns `true` then it is safe to write to the image
    /// from host and use in any way.
    ///
    /// If old content is not needed then no synchronization is required.
    /// Otherwise memory barrier with is required.
    fn detached(&self) -> bool;
}

pub trait Buffer: Clone + Debug + Eq + Hash + Resource {
    /// Returns the buffer size in bytes.
    fn size(&self) -> usize;

    /// Returns the buffer usage.
    fn usage(&self) -> crate::generic::BufferUsage;

    /// Returns the name of the buffer.
    fn name(&self) -> &str;

    /// Returns `true` if the buffer is not shared,
    /// meaning that there are no other references to the buffer
    /// including references that tracks that GPU may be using the buffer.
    ///
    /// If this method returns `true` then it is safe to write to or read from the buffer from host.
    fn detached(&self) -> bool;

    /// Maps the buffer region for host access.
    ///
    /// Requires that the buffer was created with `Memory::Shared`, `Memory::Upload` or `Memory::Download`.
    /// The buffer must be in detached state - the [`detached`](Buffer::detached) must return `true`.
    /// This function should be called only once at a time, and [`unmap`](Buffer::unmap) should be called after the mapping is no longer needed.
    fn map<R>(&mut self, range: R) -> Result<(), DeviceError>
    where
        R: crate::generic::BufferRange;

    /// Unmaps the buffer.
    ///
    /// This is no-op for persistently mapped buffers.
    /// This function should be called only after [`map`](Buffer::map) finishes successfully.
    fn unmap(&mut self);

    /// Return an inmutable slice of the mapped buffer region.
    ///
    /// This function should be called only between [`map`](Buffer::map) and [`unmap`](Buffer::unmap).
    fn read_mapped_range<R>(&mut self, range: R) -> Result<BufferMappedRange<'_>, DeviceError>
    where
        R: crate::generic::BufferRange;

    /// Return a mutable slice of the mapped buffer region.
    ///
    /// This function should be called only between [`map`](Buffer::map) and [`unmap`](Buffer::unmap).
    fn write_mapped_range<R>(&mut self, range: R) -> Result<BufferMappedRangeMut<'_>, DeviceError>
    where
        R: crate::generic::BufferRange;

    /// Writes data to the buffer at the given offset.
    ///
    /// This function will map and unmap the buffer, so it is not suitable for frequent writes.
    /// For frequent writes, use `map` and `write_mapped_range` instead.
    fn write(&mut self, offset: usize, data: &[u8]) -> Result<(), DeviceError>;

    /// Reads data from the buffer at the given offset.
    ///
    /// This function will map and unmap the buffer, so it is not suitable for frequent reads.
    /// For frequent reads, use `map` and `read_mapped_range` instead.
    fn read(&mut self, offset: usize, data: &mut [u8]) -> Result<(), DeviceError>;
}

pub trait Library: Clone + Debug + Resource {
    /// Returns shader entry point.
    fn entry<'a>(&self, entry: &'a str) -> Shader<'a>;
}
