use std::range::Range;

use ash::vk::Handle;
use ash::vk::{self, Rect2D};
use smallvec::SmallVec;

use crate::{
    AsImageSlice,
    backend::from::TryIntoAsh,
    generic::{
        Arguments, AsBufferSlice, BlasBuildDesc, ClearColor, ClearDepthStencil, DeviceRepr, Extent,
        Extent2, Extent3, LoadOp, Offset2, Offset3, PipelineStages, PixelFormat, RenderPassDesc,
        StoreOp, TexelShape, TlasBuildDesc,
    },
};

use super::{
    Blas, Buffer, ComputePipeline, Device, Frame, Image, RenderPipeline, Tlas,
    access::access_for_stages, format_aspect, from::IntoAsh, handle_host_oom,
    layout::PipelineLayout, refs::Refs, unexpected_error,
};

/// Pads `shape`'s extent/resolved stride out to the 3 axes (element/row/
/// plane) every native buffer<->image copy command needs, regardless of
/// how many axes `shape` itself specifies (`N` is `1..=3` at these call
/// sites). Any axis at or beyond `N` defaults to a tightly-packed value
/// (extent `1`, stride derived the same way `TexelShape::byte_stride` derives
/// any unset axis).
fn pad_shape_to_3<const N: usize>(shape: &TexelShape<N>) -> (Extent3<u32>, [usize; 3]) {
    let resolved = shape.byte_stride();

    let mut extent = [1u32; 3];
    for i in 0..N {
        extent[i] = shape.extent.0[i];
    }

    let mut out = [0usize; 3];
    for i in 0..3 {
        out[i] = if i < N {
            resolved[i]
        } else if i == 0 {
            shape.format.block_size()
        } else {
            out[i - 1] * extent[i - 1] as usize
        };
    }

    (Extent(extent), out)
}

pub struct CommandBuffer {
    pub(super) handle: vk::CommandBuffer,
    pub(super) pool: vk::CommandPool,
    pub(super) present: SmallVec<[Frame; 2]>,
    pub(super) refs: Refs,
}

pub struct CommandEncoder {
    device: Device,
    handle: vk::CommandBuffer,
    pool: vk::CommandPool,
    present: SmallVec<[Frame; 2]>,
    refs: Refs,
}

impl CommandEncoder {
    pub(super) fn new(
        device: Device,
        handle: vk::CommandBuffer,
        pool: vk::CommandPool,
        refs: Refs,
    ) -> Self {
        CommandEncoder {
            device,
            handle,
            pool,
            present: SmallVec::new(),
            refs,
        }
    }

    /// Creates a null encoder for use when command pool allocation fails.
    pub(super) fn null(device: Device) -> Self {
        CommandEncoder {
            device,
            handle: vk::CommandBuffer::null(),
            pool: vk::CommandPool::null(),
            present: SmallVec::new(),
            refs: Refs::new(),
        }
    }
}

#[hidden_trait::expose]
impl crate::traits::SyncCommandEncoder for CommandEncoder {
    #[inline(always)]
    fn barrier(&mut self, prev_stages: PipelineStages, next_stages: PipelineStages) {
        if self.handle.is_null() {
            return;
        }

        barrier(&self.device, self.handle, prev_stages, next_stages);
    }

    #[inline(always)]
    fn init_image(
        &mut self,
        prev_use: PipelineStages,
        next_use: PipelineStages,
        image: impl AsRef<Image>,
    ) {
        let image = image.as_ref();
        if self.handle.is_null() || image.handle().is_null() {
            return;
        }

        image_barrier(&self.device, self.handle, prev_use, next_use, image);
        self.refs.add_image(image.clone());
    }
}

#[hidden_trait::expose]
impl crate::traits::CommandEncoder for CommandEncoder {
    #[inline]
    fn present(&mut self, frame: Frame, prev_use: PipelineStages) {
        // Always collect present frames even in OOM state so swapchain images are released.
        if !self.handle.is_null() {
            unsafe {
                self.device.ash().cmd_pipeline_barrier(
                    self.handle,
                    ash::vk::PipelineStageFlags::BOTTOM_OF_PIPE | prev_use.into_ash(),
                    ash::vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[ash::vk::ImageMemoryBarrier::default()
                        .src_access_mask(access_for_stages(prev_use))
                        .dst_access_mask(ash::vk::AccessFlags::empty())
                        .old_layout(ash::vk::ImageLayout::GENERAL)
                        .new_layout(frame.present_layout())
                        .image(frame.image().as_image().handle())
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })],
                )
            }
            self.refs.add_image(frame.image().as_image().clone());
        }
        self.present.push(frame);
    }

    #[inline]
    fn finish(self) -> CommandBuffer {
        if !self.handle.is_null() {
            let result = unsafe { self.device.ash().end_command_buffer(self.handle) };
            match result {
                Ok(()) => {}
                Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY) => handle_host_oom(),
                Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                    self.device.set_oom();
                }
                Err(err) => unexpected_error(err),
            }
        }

        CommandBuffer {
            handle: self.handle,
            pool: self.pool,
            present: self.present,
            refs: self.refs,
        }
    }

    #[inline]
    fn copy(&mut self) -> CopyCommandEncoder<'_> {
        CopyCommandEncoder {
            device: self.device.clone(),
            handle: self.handle,
            refs: &mut self.refs,
        }
    }

    #[inline]
    fn acceleration_structure(&mut self) -> AccelerationStructureCommandEncoder<'_> {
        AccelerationStructureCommandEncoder {
            device: self.device.clone(),
            handle: self.handle,
            refs: &mut self.refs,
        }
    }

    #[inline]
    fn compute(&mut self) -> ComputeCommandEncoder<'_> {
        ComputeCommandEncoder {
            device: self.device.clone(),
            handle: self.handle,
            refs: &mut self.refs,
            current_layout: None,
        }
    }

    fn render(&mut self, desc: RenderPassDesc) -> RenderCommandEncoder<'_> {
        if self.handle.is_null() {
            return RenderCommandEncoder {
                device: self.device.clone(),
                handle: self.handle,
                current_layout: None,
                refs: &mut self.refs,
            };
        }

        assert!(
            !desc.color_attachments.is_empty() || desc.depth_stencil_attachment.is_some(),
            "Render pass must have at least one color attachment or a depth-stencil attachment"
        );

        let mut render_area = None::<ash::vk::Rect2D>;

        let mut color_attachments = Vec::with_capacity(desc.color_attachments.len());
        for color in desc.color_attachments.iter() {
            color.image.assert_fits_image();

            assert_eq!(
                color.image.layers.start, 0,
                "Temporary restriction: color attachment layers must start at 0"
            );
            assert_eq!(
                color.image.layers.end, 1,
                "Temporary restriction: color attachment layers must end at 1"
            );

            let format = color.image.image().format();
            debug_assert!(format.is_color());

            let color_offset = color.image.offset().into_ash();
            let color_extent = color.image.extent().into_ash();

            match render_area {
                Some(area) => {
                    assert_eq!(
                        area.offset, color_offset,
                        "All attachments in a render pass must have the same offset"
                    );
                    assert_eq!(
                        area.extent, color_extent,
                        "All attachments in a render pass must have the same extent"
                    );
                }
                None => {
                    render_area = Some(vk::Rect2D {
                        offset: color_offset,
                        extent: color_extent,
                    });
                }
            }

            let mut attachment = vk::RenderingAttachmentInfo::default();

            self.refs.add_image(color.image.image().clone());

            attachment.image_view = color.image.image().view_handle();
            attachment.image_layout = vk::ImageLayout::GENERAL;
            attachment.load_op = match color.load {
                LoadOp::Load => vk::AttachmentLoadOp::LOAD,
                LoadOp::Clear(ClearColor(r, g, b, a)) => {
                    attachment.clear_value = vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [r, g, b, a],
                        },
                    };
                    vk::AttachmentLoadOp::CLEAR
                }
                LoadOp::DontCare => vk::AttachmentLoadOp::DONT_CARE,
            };
            attachment.store_op = match color.store {
                StoreOp::Store => vk::AttachmentStoreOp::STORE,
                StoreOp::DontCare => vk::AttachmentStoreOp::DONT_CARE,
            };
            color_attachments.push(attachment);
        }

        let mut info = vk::RenderingInfo::default().color_attachments(&color_attachments);

        let depth_attachment;
        let stencil_attachment;

        if let Some(depth) = &desc.depth_stencil_attachment {
            depth.image.assert_fits_image();

            assert_eq!(
                depth.image.layers.start, 0,
                "Temporary restriction: depth attachment layers must start at 0"
            );
            assert_eq!(
                depth.image.layers.end, 1,
                "Temporary restriction: depth attachment layers must end at 1"
            );

            let format = depth.image.image().format();
            debug_assert!(format.is_depth() || format.is_stencil());

            let depth_offset = depth.image.offset().into_ash();
            let depth_extent = depth.image.extent().into_ash();

            match render_area {
                Some(area) => {
                    assert_eq!(
                        area.offset, depth_offset,
                        "All attachments in a render pass must have the same offset"
                    );
                    assert_eq!(
                        area.extent, depth_extent,
                        "All attachments in a render pass must have the same extent"
                    );
                }
                None => {
                    render_area = Some(vk::Rect2D {
                        offset: depth_offset,
                        extent: depth_extent,
                    });
                }
            }

            if format.is_depth() {
                let mut attachment = vk::RenderingAttachmentInfo::default();

                self.refs.add_image(depth.image.image().clone());

                attachment.image_view = depth.image.image().view_handle();
                attachment.image_layout = vk::ImageLayout::GENERAL;
                attachment.load_op = match depth.load {
                    LoadOp::Load => vk::AttachmentLoadOp::LOAD,
                    LoadOp::Clear(ClearDepthStencil { depth, stencil }) => {
                        attachment.clear_value = vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
                        };
                        vk::AttachmentLoadOp::CLEAR
                    }
                    LoadOp::DontCare => vk::AttachmentLoadOp::DONT_CARE,
                };
                attachment.store_op = match depth.store {
                    StoreOp::Store => vk::AttachmentStoreOp::STORE,
                    StoreOp::DontCare => vk::AttachmentStoreOp::DONT_CARE,
                };
                depth_attachment = attachment;
                info.p_depth_attachment = &depth_attachment;
            }
            if format.is_stencil() {
                let mut attachment = vk::RenderingAttachmentInfo::default();

                self.refs.add_image(depth.image.image().clone());

                attachment.image_view = depth.image.image().view_handle();
                attachment.load_op = match depth.load {
                    LoadOp::Load => vk::AttachmentLoadOp::LOAD,
                    LoadOp::Clear(ClearDepthStencil { depth, stencil }) => {
                        attachment.clear_value = vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
                        };
                        vk::AttachmentLoadOp::CLEAR
                    }
                    LoadOp::DontCare => vk::AttachmentLoadOp::DONT_CARE,
                };
                attachment.store_op = match depth.store {
                    StoreOp::Store => vk::AttachmentStoreOp::STORE,
                    StoreOp::DontCare => vk::AttachmentStoreOp::DONT_CARE,
                };
                stencil_attachment = attachment;
                info.p_stencil_attachment = &stencil_attachment;
            }
        }

        unsafe {
            self.device.ash().cmd_begin_rendering(
                self.handle,
                &info
                    .render_area(render_area.expect("Render area must be specified"))
                    .layer_count(1),
            )
        }

        let mut render = RenderCommandEncoder {
            device: self.device.clone(),
            handle: self.handle,
            current_layout: None,
            refs: &mut self.refs,
        };

        // Default the viewport and scissor to the full extent of the first
        // attachment (first color attachment, else depth-stencil), matching
        // an explicit `with_viewport`/`with_scissor` call to the same effect.
        let first_attachment = desc
            .color_attachments
            .first()
            .map(|color| &color.image)
            .or_else(|| {
                desc.depth_stencil_attachment
                    .as_ref()
                    .map(|depth| &depth.image)
            });
        if let Some(image) = first_attachment {
            let extent = image.extent();
            render.with_viewport(Offset3::ZERO, extent.to_3d().cast_as_f32());
            render.with_scissor(Offset2::ZERO, extent);
        }

        render
    }
}

pub struct ComputeCommandEncoder<'a> {
    device: Device,
    handle: vk::CommandBuffer,
    refs: &'a mut Refs,
    current_layout: Option<PipelineLayout>,
}

impl ComputeCommandEncoder<'_> {
    #[inline(always)]
    pub(super) fn handle(&self) -> vk::CommandBuffer {
        self.handle
    }

    #[inline(always)]
    pub(super) fn device(&self) -> &Device {
        &self.device
    }

    #[inline(always)]
    pub(super) fn current_layout(&self) -> Option<&PipelineLayout> {
        self.current_layout.as_ref()
    }

    #[inline(always)]
    pub(super) fn refs_mut(&mut self) -> &mut Refs {
        &mut self.refs
    }
}

#[hidden_trait::expose]
impl crate::traits::SyncCommandEncoder for ComputeCommandEncoder<'_> {
    #[inline]
    fn barrier(&mut self, prev_stages: PipelineStages, next_stages: PipelineStages) {
        if self.handle.is_null() {
            return;
        }
        barrier(&self.device, self.handle, prev_stages, next_stages);
    }

    #[inline]
    fn init_image(
        &mut self,
        prev_use: PipelineStages,
        next_use: PipelineStages,
        image: impl AsRef<Image>,
    ) {
        let image = image.as_ref();

        if self.handle.is_null() || image.handle().is_null() {
            return;
        }

        image_barrier(&self.device, self.handle, prev_use, next_use, image);
        self.refs.add_image(image.clone());
    }
}

#[hidden_trait::expose]
impl crate::traits::ComputeCommandEncoder for ComputeCommandEncoder<'_> {
    #[inline]
    fn with_pipeline(&mut self, pipeline: &ComputePipeline) {
        if self.handle.is_null() || pipeline.is_null() {
            return;
        }
        unsafe {
            self.device.ash().cmd_bind_pipeline(
                self.handle,
                ash::vk::PipelineBindPoint::COMPUTE,
                pipeline.handle(),
            );
        }
        self.current_layout = Some(pipeline.layout().clone());
        self.refs.add_compute_pipeline(pipeline.clone());
    }

    #[inline]
    fn with_arguments(&mut self, group: u32, arguments: &impl Arguments) {
        if self.handle.is_null() {
            return;
        }
        arguments.bind_compute(group, self);
    }

    #[inline]
    fn with_constants(&mut self, constants: &impl DeviceRepr) {
        if self.handle.is_null() {
            return;
        }
        let Some(layout) = self.current_layout.as_ref() else {
            panic!("Constants binding requires a pipeline to be bound to the encoder");
        };

        let data = constants.as_repr();

        unsafe {
            self.device.ash().cmd_push_constants(
                self.handle,
                layout.handle(),
                ash::vk::ShaderStageFlags::ALL,
                0,
                bytemuck::bytes_of(&data),
            )
        }
    }

    #[inline]
    fn dispatch(&mut self, groups: Extent3) {
        if self.handle.is_null() {
            return;
        }
        unsafe {
            self.device.ash().cmd_dispatch(
                self.handle,
                groups.width(),
                groups.height(),
                groups.depth(),
            )
        }
    }
}

pub struct RenderCommandEncoder<'a> {
    device: Device,
    handle: vk::CommandBuffer,
    refs: &'a mut Refs,
    current_layout: Option<PipelineLayout>,
}

impl RenderCommandEncoder<'_> {
    #[inline(always)]
    pub(super) fn handle(&self) -> vk::CommandBuffer {
        self.handle
    }

    #[inline(always)]
    pub(super) fn device(&self) -> &Device {
        &self.device
    }

    #[inline(always)]
    pub(super) fn current_layout(&self) -> Option<&PipelineLayout> {
        self.current_layout.as_ref()
    }

    #[inline(always)]
    pub(super) fn refs_mut(&mut self) -> &mut Refs {
        &mut self.refs
    }
}

impl Drop for RenderCommandEncoder<'_> {
    #[inline]
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { self.device.ash().cmd_end_rendering(self.handle) }
        }
    }
}

#[hidden_trait::expose]
impl crate::traits::RenderCommandEncoder for RenderCommandEncoder<'_> {
    #[inline]
    fn with_pipeline(&mut self, pipeline: &RenderPipeline) {
        if self.handle.is_null() || pipeline.is_null() {
            return;
        }
        unsafe {
            self.device.ash().cmd_bind_pipeline(
                self.handle,
                ash::vk::PipelineBindPoint::GRAPHICS,
                pipeline.handle(),
            );
        }
        self.current_layout = Some(pipeline.layout().clone());
        self.refs.add_render_pipeline(pipeline.clone());
    }

    #[inline]
    fn with_viewport(&mut self, offset: Offset3<f32>, extent: Extent3<f32>) {
        if self.handle.is_null() {
            return;
        }
        unsafe {
            self.device.ash().cmd_set_viewport(
                self.handle,
                0,
                &[ash::vk::Viewport::default()
                    .x(offset.x())
                    .y(offset.y())
                    .width(extent.width())
                    .height(extent.height())
                    .min_depth(offset.z())
                    .max_depth(extent.depth())],
            );
        }
    }

    #[inline]
    fn with_scissor(&mut self, offset: Offset2<i32>, extent: Extent2<u32>) {
        if self.handle.is_null() {
            return;
        }
        unsafe {
            self.device.ash().cmd_set_scissor(
                self.handle,
                0,
                &[ash::vk::Rect2D::default()
                    .offset(ash::vk::Offset2D {
                        x: offset.x(),
                        y: offset.y(),
                    })
                    .extent(ash::vk::Extent2D {
                        width: extent.width(),
                        height: extent.height(),
                    })],
            );
        }
    }

    #[inline]
    fn with_arguments(&mut self, group: u32, arguments: &impl Arguments) {
        if self.handle.is_null() {
            return;
        }
        arguments.bind_render(group, self);
    }

    #[inline]
    fn with_constants(&mut self, constants: &impl DeviceRepr) {
        if self.handle.is_null() {
            return;
        }
        let Some(layout) = self.current_layout.as_ref() else {
            panic!("Constants binding requires a pipeline to be bound to the encoder");
        };

        let data = constants.as_repr();

        unsafe {
            self.device.ash().cmd_push_constants(
                self.handle,
                layout.handle(),
                ash::vk::ShaderStageFlags::ALL,
                0,
                bytemuck::bytes_of(&data),
            )
        }
    }

    #[inline]
    fn bind_vertex_buffers(&mut self, start: u32, slices: &[impl AsBufferSlice]) {
        if self.handle.is_null() {
            return;
        }
        let mut handles = smallvec::SmallVec::<[_; 8]>::with_capacity(slices.len());
        let mut offsets = smallvec::SmallVec::<[_; 8]>::with_capacity(slices.len());
        for slice in slices.iter() {
            let slice: crate::generic::BufferSlice = slice.as_buffer_slice();
            handles.push(slice.buffer.handle());
            offsets.push(slice.offset as u64);
            self.refs.add_buffer(slice.buffer.clone().into_owned());
        }

        unsafe {
            self.device
                .ash()
                .cmd_bind_vertex_buffers(self.handle, start, &handles, &offsets)
        }
    }

    #[inline]
    fn bind_index_buffer(&mut self, slice: impl AsBufferSlice) {
        if self.handle.is_null() {
            return;
        }
        let slice: crate::generic::BufferSlice = slice.as_buffer_slice();
        unsafe {
            self.device.ash().cmd_bind_index_buffer(
                self.handle,
                slice.buffer.handle(),
                slice.offset as u64,
                vk::IndexType::UINT32,
            )
        }
        self.refs.add_buffer(slice.buffer.clone().into_owned());
    }

    #[inline]
    fn draw(&mut self, vertices: impl Into<Range<u32>>, instances: impl Into<Range<u32>>) {
        let vertices = vertices.into();
        let instances = instances.into();

        if self.handle.is_null() {
            return;
        }
        unsafe {
            self.device.ash().cmd_draw(
                self.handle,
                vertices.end - vertices.start,
                instances.end - instances.start,
                vertices.start,
                instances.start,
            );
        }
    }

    #[inline]
    fn draw_indexed(
        &mut self,
        vertex_offset: i32,
        indices: impl Into<Range<u32>>,
        instances: impl Into<Range<u32>>,
    ) {
        let indices = indices.into();
        let instances = instances.into();

        if self.handle.is_null() {
            return;
        }
        unsafe {
            self.device.ash().cmd_draw_indexed(
                self.handle,
                indices.end - indices.start,
                instances.end - instances.start,
                indices.start,
                vertex_offset,
                instances.start,
            );
        }
    }
}

pub struct CopyCommandEncoder<'a> {
    device: Device,
    handle: vk::CommandBuffer,
    refs: &'a mut Refs,
}

#[hidden_trait::expose]
impl crate::traits::SyncCommandEncoder for CopyCommandEncoder<'_> {
    #[inline]
    fn barrier(&mut self, prev_stages: PipelineStages, next_stages: PipelineStages) {
        if self.handle.is_null() {
            return;
        }
        barrier(&self.device, self.handle, prev_stages, next_stages);
    }

    #[inline]
    fn init_image(
        &mut self,
        prev_use: PipelineStages,
        next_use: PipelineStages,
        image: impl AsRef<Image>,
    ) {
        let image = image.as_ref();

        if self.handle.is_null() || image.handle().is_null() {
            return;
        }

        image_barrier(&self.device, self.handle, prev_use, next_use, image);
        self.refs.add_image(image.clone());
    }
}

#[hidden_trait::expose]
impl crate::traits::CopyCommandEncoder for CopyCommandEncoder<'_> {
    #[inline]
    fn copy_buffer_to_buffer(
        &mut self,
        src: &impl crate::AsBufferSlice,
        dst: &impl crate::AsBufferSlice,
    ) {
        let src = src.as_buffer_slice();
        let dst = dst.as_buffer_slice();

        assert_eq!(
            src.size(),
            dst.size(),
            "copy_buffer_to_buffer requires src and dst to have the same size"
        );

        if self.handle.is_null()
            || src.buffer().handle().is_null()
            || dst.buffer().handle().is_null()
        {
            return;
        }

        self.refs.add_buffer(src.buffer().clone());
        self.refs.add_buffer(dst.buffer().clone());

        unsafe {
            self.device.ash().cmd_copy_buffer(
                self.handle,
                src.buffer().handle(),
                dst.buffer().handle(),
                &[vk::BufferCopy {
                    src_offset: src.offset() as u64,
                    dst_offset: dst.offset() as u64,
                    size: src.size() as u64,
                }],
            )
        }
    }

    #[inline]
    fn copy_buffer_to_image<const N: usize>(
        &mut self,
        src: &impl AsBufferSlice,
        shape: TexelShape<N>,
        dst: &impl AsImageSlice<N>,
    ) {
        const { assert!(N >= 1 && N <= 3, "copy_buffer_to_image: Rank must be 1..=3") };

        let src = src.as_buffer_slice();
        let dst = dst.as_image_slice();
        dst.assert_fits_image();

        // The buffer's texel extent must match image slice extent.
        assert_eq!(
            shape.extent, dst.extent,
            "copy_buffer_to_image: TexelShape extent must match the destination image slice extent"
        );

        let block_size = dst.image.format().block_size();
        let block_extent = dst.image.format().block_extent();
        let (extent, resolved) = pad_shape_to_3(&shape);
        assert_eq!(
            resolved[0], block_size,
            "copy_buffer_to_image: element stride must equal the format's tightly-packed size ({block_size}) - native copy commands can't vary it; use Cvt for padded/strided element transfers"
        );
        let bytes_per_line = resolved[1];
        let bytes_per_plane = resolved[2];

        debug_assert_eq!(bytes_per_line % block_size, 0);
        debug_assert_eq!(bytes_per_plane % block_size, 0);
        let buffer_row_length = (bytes_per_line / block_size) as u32 * block_extent.width();
        let buffer_image_height = (bytes_per_plane / bytes_per_line) as u32 * block_extent.height();

        if self.handle.is_null() || src.buffer().handle().is_null() || dst.image.handle().is_null()
        {
            return;
        }

        self.refs.add_buffer(src.buffer().clone());
        self.refs.add_image(dst.image.clone().into_owned());

        let dst_offset = dst.offset.to_3d();

        unsafe {
            self.device.ash().cmd_copy_buffer_to_image(
                self.handle,
                src.buffer().handle(),
                dst.image.handle(),
                ash::vk::ImageLayout::GENERAL,
                &[vk::BufferImageCopy {
                    buffer_offset: src.offset() as u64,
                    buffer_row_length,
                    buffer_image_height,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: format_aspect(dst.image.format()),
                        mip_level: dst.image.base_level() + dst.level,
                        base_array_layer: dst.image.base_layer() + dst.layers.start,
                        layer_count: dst.layers.end - dst.layers.start,
                    },
                    image_offset: dst_offset
                        .try_into_ash()
                        .expect("copy_buffer_to_image: dst offset exceeds i32 range"),
                    image_extent: extent.into_ash(),
                }],
            )
        }
    }

    #[inline]
    fn copy_image_region<const N: usize>(
        &mut self,
        src: &impl crate::AsImageSlice<N>,
        dst: &impl crate::AsImageSlice<N>,
    ) {
        let src = src.as_image_slice();
        src.assert_fits_image();
        let dst = dst.as_image_slice();
        dst.assert_fits_image();

        assert_eq!(
            src.extent, dst.extent,
            "copy_image_region: source and destination image slices must have the same extent"
        );

        if self.handle.is_null() || src.image.handle().is_null() || dst.image.handle().is_null() {
            return;
        }

        self.refs.add_image(src.image.clone().into_owned());
        self.refs.add_image(dst.image.clone().into_owned());

        let src_offset = src.offset.to_3d();
        let dst_offset = dst.offset.to_3d();
        let extent = src.extent.to_3d();

        unsafe {
            self.device.ash().cmd_copy_image(
                self.handle,
                src.image.handle(),
                ash::vk::ImageLayout::GENERAL,
                dst.image.handle(),
                ash::vk::ImageLayout::GENERAL,
                &[vk::ImageCopy {
                    src_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: format_aspect(src.image.format()),
                        mip_level: src.image.base_level() + src.level,
                        base_array_layer: src.image.base_layer() + src.layers.start,
                        layer_count: src.layers.end - src.layers.start,
                    },
                    src_offset: src_offset
                        .try_into_ash()
                        .expect("copy_image_region: src offset exceeds i32 range"),
                    dst_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: format_aspect(dst.image.format()),
                        mip_level: dst.image.base_level() + dst.level,
                        base_array_layer: dst.image.base_layer() + dst.layers.start,
                        layer_count: dst.layers.end - dst.layers.start,
                    },
                    dst_offset: dst_offset
                        .try_into_ash()
                        .expect("copy_image_region: dst offset exceeds i32 range"),
                    extent: extent.into_ash(),
                }],
            )
        }
    }

    #[inline]
    fn copy_image_to_buffer<const N: usize>(
        &mut self,
        src: &impl crate::AsImageSlice<N>,
        dst: &impl crate::AsBufferSlice,
        shape: TexelShape<N>,
    ) {
        const {
            assert!(
                N >= 1 && N <= 3,
                "copy_image_to_buffer: TexelShape rank must be 1..=3"
            )
        };

        let src = src.as_image_slice();
        src.assert_fits_image();
        let dst = dst.as_buffer_slice();

        assert_eq!(
            shape.extent, src.extent,
            "copy_image_to_buffer: TexelShape extent must match the source image slice extent"
        );

        let block_size = src.image.format().block_size();
        let block_extent = src.image.format().block_extent();
        let (extent, resolved) = pad_shape_to_3(&shape);
        assert_eq!(
            resolved[0], block_size,
            "copy_image_to_buffer: element stride must equal the format's tightly-packed size ({block_size}) - native copy commands can't vary it; use Cvt for padded/strided element transfers"
        );
        let bytes_per_line = resolved[1];
        let bytes_per_plane = resolved[2];

        debug_assert_eq!(bytes_per_line % block_size, 0);
        debug_assert_eq!(bytes_per_plane % block_size, 0);
        let buffer_row_length = (bytes_per_line / block_size) as u32 * block_extent.width();
        let buffer_image_height = (bytes_per_plane / bytes_per_line) as u32 * block_extent.height();

        if self.handle.is_null() || src.image.handle().is_null() || dst.buffer().handle().is_null()
        {
            return;
        }

        self.refs.add_image(src.image.clone().into_owned());
        self.refs.add_buffer(dst.buffer().clone());

        let src_offset = src.offset.to_3d();

        unsafe {
            self.device.ash().cmd_copy_image_to_buffer(
                self.handle,
                src.image.handle(),
                ash::vk::ImageLayout::GENERAL,
                dst.buffer().handle(),
                &[vk::BufferImageCopy {
                    buffer_offset: dst.offset() as u64,
                    buffer_row_length,
                    buffer_image_height,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: format_aspect(src.image.format()),
                        mip_level: src.image.base_level() + src.level,
                        base_array_layer: src.image.base_layer() + src.layers.start,
                        layer_count: src.layers.end - src.layers.start,
                    },
                    image_offset: src_offset
                        .try_into_ash()
                        .expect("copy_image_to_buffer: src offset exceeds i32 range"),
                    image_extent: extent.into_ash(),
                }],
            )
        }
    }

    #[inline]
    fn fill_buffer(&mut self, slice: impl AsBufferSlice, byte: u8) {
        let slice = slice.as_buffer_slice();

        if self.handle.is_null() || slice.buffer.handle().is_null() {
            return;
        }

        self.refs.add_buffer(slice.buffer.clone().into_owned());

        let data = u32::from_ne_bytes([byte; 4]);

        unsafe {
            self.device.ash().cmd_fill_buffer(
                self.handle,
                slice.buffer.handle(),
                slice.offset as u64,
                slice.size as u64,
                data,
            );
        }
    }

    #[inline]
    fn write_buffer_raw(&mut self, slice: impl AsBufferSlice, data: &[u8]) {
        let slice = slice.as_buffer_slice();

        if self.handle.is_null() || slice.buffer.handle().is_null() {
            return;
        }

        if data.is_empty() {
            return;
        }

        assert!(slice.size >= data.len());

        self.refs.add_buffer(slice.buffer.clone().into_owned());

        const CHUNK_SIZE: usize = 65536;

        let full_chunks = data.len() / CHUNK_SIZE;

        for i in 0..full_chunks {
            unsafe {
                self.device.ash().cmd_update_buffer(
                    self.handle,
                    slice.buffer.handle(),
                    (slice.offset + i * CHUNK_SIZE) as u64,
                    &data[i * CHUNK_SIZE..(i + 1) * CHUNK_SIZE],
                )
            }
        }

        let remainder = data.len() % CHUNK_SIZE;
        if remainder > 0 {
            unsafe {
                self.device.ash().cmd_update_buffer(
                    self.handle,
                    slice.buffer.handle(),
                    (slice.offset + full_chunks * CHUNK_SIZE) as u64,
                    &data[full_chunks * CHUNK_SIZE..],
                )
            }
        }
    }

    #[inline]
    fn write_buffer(&mut self, slice: impl AsBufferSlice, data: &impl bytemuck::NoUninit) {
        self.write_buffer_slice(slice, bytemuck::bytes_of(data))
    }

    /// Writes data to the buffer.
    #[inline]
    fn write_buffer_slice(&mut self, slice: impl AsBufferSlice, data: &[impl bytemuck::NoUninit]) {
        self.write_buffer_raw(slice, bytemuck::cast_slice(data))
    }

    /// Writes data and emits the surrounding barriers. `prev_use` = stages
    /// that previously used the buffer, `next_use` = stages that will use it
    /// next.
    #[inline]
    fn write_buffer_sync(
        &mut self,
        slice: impl AsBufferSlice,
        data: &impl bytemuck::NoUninit,
        prev_use: PipelineStages,
        next_use: PipelineStages,
    ) {
        self.barrier(prev_use, PipelineStages::TRANSFER);
        self.write_buffer(slice, data);
        self.barrier(PipelineStages::TRANSFER, next_use);
    }
}

pub struct AccelerationStructureCommandEncoder<'a> {
    device: Device,
    handle: vk::CommandBuffer,
    refs: &'a mut Refs,
}

#[hidden_trait::expose]
impl crate::traits::AccelerationStructureCommandEncoder
    for AccelerationStructureCommandEncoder<'_>
{
    fn build_blas(&mut self, blas: &Blas, desc: BlasBuildDesc, scratch: impl AsBufferSlice) {
        todo!();
    }

    fn build_tlas(&mut self, tlas: &Tlas, desc: TlasBuildDesc, scratch: impl AsBufferSlice) {
        todo!();
    }
}

#[inline]
fn barrier(
    device: &Device,
    handle: ash::vk::CommandBuffer,
    prev_stages: PipelineStages,
    next_stages: PipelineStages,
) {
    unsafe {
        device.ash().cmd_pipeline_barrier(
            handle,
            ash::vk::PipelineStageFlags::BOTTOM_OF_PIPE | prev_stages.into_ash(),
            ash::vk::PipelineStageFlags::TOP_OF_PIPE | next_stages.into_ash(),
            vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::default()
                .src_access_mask(access_for_stages(prev_stages))
                .dst_access_mask(access_for_stages(next_stages))],
            &[],
            &[],
        )
    }
}

#[inline]
fn image_barrier(
    device: &Device,
    handle: ash::vk::CommandBuffer,
    prev_use: PipelineStages,
    next_use: PipelineStages,
    image: &Image,
) {
    let mut aspect_mask = ash::vk::ImageAspectFlags::empty();
    if image.format().is_color() {
        aspect_mask |= ash::vk::ImageAspectFlags::COLOR;
    }
    if image.format().is_depth() {
        aspect_mask |= ash::vk::ImageAspectFlags::DEPTH;
    }
    if image.format().is_stencil() {
        aspect_mask |= ash::vk::ImageAspectFlags::STENCIL;
    }

    unsafe {
        device.ash().cmd_pipeline_barrier(
            handle,
            ash::vk::PipelineStageFlags::BOTTOM_OF_PIPE | prev_use.into_ash(),
            ash::vk::PipelineStageFlags::TOP_OF_PIPE | next_use.into_ash(),
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[ash::vk::ImageMemoryBarrier::default()
                .src_access_mask(access_for_stages(prev_use))
                .dst_access_mask(access_for_stages(next_use))
                .old_layout(ash::vk::ImageLayout::UNDEFINED)
                .new_layout(ash::vk::ImageLayout::GENERAL)
                .image(image.handle())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask,
                    base_mip_level: image.base_level(),
                    level_count: image.levels(),
                    base_array_layer: image.base_layer(),
                    layer_count: image.layers(),
                })],
        )
    }
}
