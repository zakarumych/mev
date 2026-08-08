use std::{fmt::Debug, hash::Hash, ops::Deref, range::Range};

use mev_proc::match_backend;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::generic::{
    Arguments, AsBufferSlice, BlasBuildDesc, BlasDesc, BufferDesc, BufferInit,
    BufferInitDesc, BufferMappedRange, BufferMappedRangeMut, BufferUsage, Capabilities, ColorSpace,
    ComputePipelineDesc, DeviceDesc, DeviceError, DeviceRepr, Extent2, Extent3, Image2D, ImageDesc,
    ImageDims, ImageExtent, ImageRegion, ImageSlice, ImageUsage, LibraryDesc, Offset2, Offset3,
    PipelineError, PipelineStages, PixelFormat, PresentMode, RenderPassDesc, RenderPipelineDesc,
    SamplerDesc, Shader, ShaderLibraryError, SurfaceError, TexelShape, TlasBuildDesc, TlasDesc,
    ViewDesc,
};

// match_backend! {
//     metal => {
//         pub trait Resource: Send + Sync + 'static {}
//     }
//     vulkan => {
//         pub trait Resource: Send + Sync + 'static {}
//     }
//     webgpu => {
//         pub trait Resource: 'static {}
//     }
// }

with_metal! {
    pub trait Resource: Send + Sync + Sized + 'static {}
}

with_vulkan! {
    pub trait Resource: Send + Sync + Sized + 'static {}
}

with_webgpu! {
    pub trait Resource: Sized + 'static {}
}

pub(crate) trait Instance: Debug + Resource {
    fn capabilities(&self) -> &Capabilities;

    fn new_device(
        &self,
        info: DeviceDesc<'_>,
    ) -> Result<(crate::backend::Device, Vec<crate::backend::Queue>), DeviceError>;

    async fn new_device_async(
        &self,
        info: DeviceDesc<'_>,
    ) -> Result<(crate::backend::Device, Vec<crate::backend::Queue>), DeviceError> {
        self.new_device(info)
    }

    fn new_device_with_surface(
        &self,
        info: DeviceDesc<'_>,
        window: &impl HasWindowHandle,
        display: &impl HasDisplayHandle,
    ) -> Result<
        (
            crate::backend::Device,
            Vec<crate::backend::Queue>,
            crate::backend::Surface,
        ),
        SurfaceError,
    > {
        let (device, queues) = self.new_device(info)?;
        let surface = device.new_surface(window, display)?;
        Ok((device, queues, surface))
    }

    async fn new_device_with_surface_async(
        &self,
        info: DeviceDesc<'_>,
        window: &impl HasWindowHandle,
        display: &impl HasDisplayHandle,
    ) -> Result<
        (
            crate::backend::Device,
            Vec<crate::backend::Queue>,
            crate::backend::Surface,
        ),
        SurfaceError,
    > {
        let (device, queues) = self.new_device_async(info).await?;
        let surface = device.new_surface(window, display)?;
        Ok((device, queues, surface))
    }
}

pub(crate) trait Device: Clone + Debug + Eq + Resource {
    /// Create a new shader library.
    fn new_shader_library(
        &self,
        desc: LibraryDesc,
    ) -> Result<crate::backend::Library, ShaderLibraryError>;

    /// Create a new compute pipeline.
    fn new_compute_pipeline(
        &self,
        desc: ComputePipelineDesc,
    ) -> Result<crate::backend::ComputePipeline, PipelineError>;

    /// Create a new render pipeline.
    fn new_render_pipeline(
        &self,
        desc: RenderPipelineDesc,
    ) -> Result<crate::backend::RenderPipeline, PipelineError>;

    /// Create a new buffer with uninitialized contents.
    fn new_buffer(&self, desc: BufferDesc) -> crate::backend::Buffer;

    /// Create a new buffer and initialize it with the given data.
    fn new_buffer_init<D: BufferInit>(
        &self,
        desc: BufferInitDesc<'_, D>,
    ) -> crate::backend::Buffer {
        let size = desc.data.size();

        let mut buffer = self.new_buffer(BufferDesc {
            size,
            usage: desc.usage | BufferUsage::HOST_WRITE,
            name: desc.name,
        });

        if size > 0 {
            if buffer.map(..).is_ok() {
                if let Ok(mut mapped) = buffer.write_mapped_range(..) {
                    desc.data.write(mapped.as_mut());
                }
                buffer.unmap();
            }
        }

        buffer
    }

    /// Create a new image.
    fn new_image<T: ImageDims>(&self, desc: ImageDesc<T>) -> T::Image;

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

pub(crate) trait Queue: Deref<Target = crate::backend::Device> + Debug + Resource {
    /// Get the device associated with this queue.
    fn device(&self) -> &crate::backend::Device;

    /// Get the queue family index.
    fn family(&self) -> u32;

    /// Create a new command encoder associated with this queue.
    /// The encoder must be submitted to the queue it was created from.
    fn new_command_encoder(&mut self) -> crate::backend::CommandEncoder;

    /// Submit command buffers to the queue.
    ///
    /// Returns the epoch id this submission belongs to.
    fn submit<I>(&mut self, command_buffers: I) -> Result<u64, DeviceError>
    where
        I: IntoIterator<Item = crate::backend::CommandBuffer>;

    /// Submit command buffers to the queue.
    ///
    /// Returns the epoch id this submission belongs to.
    ///
    /// This method is similar to `submit`,
    /// but it also ends the current epoch and starts a new one.
    ///
    /// Queue may have fixed number of epochs in flight,
    /// so next submit may block until one of the previous epochs is finished.
    ///
    /// Therefore it is recommended to use this method with last submission of the frame,
    /// allowing a few frames to be in flight at the same time.
    fn submit_checkpoint<I>(&mut self, command_buffers: I) -> Result<u64, DeviceError>
    where
        I: IntoIterator<Item = crate::backend::CommandBuffer>;

    /// Synchronize the access to the frame resources.
    fn sync_frame(&mut self, frame: &mut crate::backend::Frame, before: PipelineStages);

    /// Wait for all operations on the queue to complete.
    fn wait_idle(&self) -> Result<(), DeviceError>;

    /// Returns the id of the last finished epoch.
    ///
    /// Returns id of the last epoch that was finished on the GPU.
    ///
    /// If resource was used in submits that returned this or smaller epoch ids,
    /// then it is safe to use the resource from host.
    fn last_finished_epoch(&self) -> u64;
}

pub(crate) trait SyncCommandEncoder {
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
        image: impl AsRef<crate::backend::Image>,
    );
}

pub(crate) trait CommandEncoder: SyncCommandEncoder {
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

pub(crate) trait ComputeCommandEncoder: SyncCommandEncoder {
    /// Sets the current compute pipeline.
    fn with_pipeline(&mut self, pipeline: &crate::backend::ComputePipeline);

    /// Sets arguments group for the current pipeline.
    fn with_arguments(&mut self, group: u32, arguments: &impl Arguments);

    /// Sets constants for the current pipeline.
    fn with_constants(&mut self, constants: &impl DeviceRepr);

    /// Dispatches compute work.
    fn dispatch(&mut self, groups: Extent3);
}

pub(crate) trait CopyCommandEncoder: SyncCommandEncoder {
    /// Fills the buffer slice with the given byte.
    fn fill_buffer(&mut self, slice: impl AsBufferSlice, byte: u8);

    /// Writes data to the buffer.
    fn write_buffer_raw(&mut self, slice: impl AsBufferSlice, data: &[u8]);

    /// Writes data to the buffer.
    fn write_buffer(&mut self, slice: impl AsBufferSlice, data: &impl DeviceRepr);

    /// Writes data to the buffer.
    fn write_buffer_slice(&mut self, slice: impl AsBufferSlice, data: &[impl bytemuck::NoUninit]);

    /// Copies bytes from src buffer to dst buffer.
    fn copy_buffer_to_buffer(
        &mut self,
        src: &impl crate::AsBufferSlice,
        dst: &impl crate::AsBufferSlice,
    );

    /// Copies pixels from `src` (a buffer region, described by `src_shape`)
    /// into `dst`. `N` (1..=3) is how many axes `src_shape` actually
    /// specifies: `N == 1` gives just an element stride/format (row and
    /// plane pitch are both inferred tightly-packed from `src_shape.extent`
    /// and `dst`'s pixel format), `N == 2` additionally gives a row stride
    /// (plane pitch is inferred), and `N == 3` gives row and plane strides
    /// explicitly. Within whatever `src_shape` does specify, any individual
    /// axis's stride may still be left unset for a tightly-packed default.
    ///
    /// `src_shape`'s element (axis 0 / X) stride is NOT actually configurable
    /// here: native buffer<->image copy commands (Vulkan/Metal/wgpu) always
    /// read a row's elements tightly packed - if set, it must equal `dst`'s
    /// pixel format's own tightly-packed element size (asserted; panics
    /// otherwise, in all build profiles). For a transfer with padded/strided
    /// elements, use `Cvt` instead - its software-addressed compute kernel
    /// supports arbitrary per-axis byte strides.
    fn copy_buffer_to_image<const N: usize>(
        &mut self,
        src: &impl crate::AsBufferSlice,
        shape: TexelShape<N>,
        dst: &impl crate::AsImageSlice<N>,
    );

    /// Copies pixels from src image to dst image.
    fn copy_image_region<const N: usize>(
        &mut self,
        src: &impl crate::AsImageSlice<N>,
        dst: &impl crate::AsImageSlice<N>,
    );

    /// Copies pixels from `src` into `dst` (a buffer region, described by
    /// `dst_shape`). Same `dst_shape` semantics as `copy_buffer_to_image`,
    /// including `N` (1..=3) meaning how many axes `dst_shape` specifies
    /// (row/plane stride may be left unset, or omitted entirely via a
    /// smaller `N`, for tight packing; element stride, if set, must equal
    /// `src`'s pixel format's tight element size, asserted in all build
    /// profiles; use `Cvt` for padded/strided elements).
    fn copy_image_to_buffer<const N: usize>(
        &mut self,
        src: &impl crate::AsImageSlice<N>,
        dst: &impl crate::AsBufferSlice,
        shape: TexelShape<N>,
    );
}

pub(crate) trait RenderCommandEncoder {
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
    fn draw(&mut self, vertices: impl Into<Range<u32>>, instances: impl Into<Range<u32>>);

    /// Draws primitives with indices.
    fn draw_indexed(
        &mut self,
        vertex_offset: i32,
        indices: impl Into<Range<u32>>,
        instances: impl Into<Range<u32>>,
    );
}

pub(crate) trait AccelerationStructureCommandEncoder {
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

pub(crate) trait Surface: Resource {
    /// List of available formats for the surface.
    /// It may change during the lifetime of the surface, so it is not recommended to cache the result.
    fn available_formats(&self) -> &[(PixelFormat, ColorSpace)];

    /// Sets the preferred pixel format for the surface.
    fn preferred_format(&mut self, format: PixelFormat);

    /// Sets the preferred color space for the surface.
    fn preferred_color_space(&mut self, color_space: ColorSpace);

    /// Sets the preferred extent for the surface.
    fn preferred_extent(&mut self, extent: Extent2);

    /// Sets the preferred present mode for the surface.
    fn preferred_present_mode(&mut self, mode: PresentMode);

    /// Sets the preferred usage for the surface.
    fn preferred_usage(&mut self, usage: ImageUsage);

    /// Sets the preferred number of images for the surface.
    fn preferred_image_count(&mut self, count: u32);

    /// Acquires next frame from the surface.
    fn next_frame(&mut self) -> Result<crate::backend::Frame, SurfaceError>;
}

pub(crate) trait Frame: Resource {
    fn image(&self) -> &Image2D;
}

pub(crate) trait Image: Clone + Debug + Eq + Hash + Resource {
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
    fn view(&self, device: &crate::backend::Device, desc: ViewDesc) -> Self;

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

pub(crate) trait Buffer: Clone + Debug + Eq + Hash + Resource {
    /// Returns the buffer size in bytes.
    fn size(&self) -> usize;

    /// Returns the buffer usage.
    fn usage(&self) -> crate::generic::BufferUsage;

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

pub(crate) trait Library: Clone + Debug + Resource {
    /// Returns shader entry point.
    fn entry<'a>(&self, entry: &'a str) -> Shader<'a>;
}
