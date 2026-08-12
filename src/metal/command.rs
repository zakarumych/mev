use std::{marker::PhantomData, ptr::NonNull, range::Range, sync::Arc};

use objc2::{Message, msg_send, rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSArray, NSInteger, NSRange, NSUInteger};
use objc2_metal::{
    MTLAccelerationStructureBoundingBoxGeometryDescriptor, MTLAccelerationStructureCommandEncoder,
    MTLAccelerationStructureGeometryDescriptor, MTLAccelerationStructureTriangleGeometryDescriptor,
    MTLAccelerationStructureUsage, MTLBlitCommandEncoder, MTLBlitOption, MTLBuffer, MTLClearColor,
    MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLDevice, MTLIndexType,
    MTLLoadAction, MTLOrigin, MTLPrimitiveAccelerationStructureDescriptor, MTLPrimitiveType,
    MTLRenderCommandEncoder, MTLRenderPassColorAttachmentDescriptor, MTLRenderPassDescriptor,
    MTLResourceOptions, MTLScissorRect, MTLSize, MTLStoreAction, MTLViewport,
};
use smallvec::SmallVec;

use crate::{
    AsImageSlice,
    generic::{
        AccelerationStructureBuildFlags, AccelerationStructurePerformance, Arguments,
        AsBufferSlice, BlasBuildDesc, BlasGeometryDesc, ClearColor, ClearDepthStencil, DeviceRepr,
        Extent, Extent2, Extent3, LoadOp, Offset2, Offset3, OutOfMemory, PipelineStages,
        PixelFormat, RenderPassDesc, StoreOp, TexelShape, TlasBuildDesc,
    },
    traits,
};

use super::{
    Blas, Buffer, Frame, Image, RenderPipeline, Tlas, from::TryIntoMetal, shader::Bindings,
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
    buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

impl CommandBuffer {
    pub(super) fn commit(self) -> Retained<ProtocolObject<dyn MTLCommandBuffer>> {
        self.buffer.commit();
        self.buffer
    }
}

pub struct CommandEncoder {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

impl CommandEncoder {
    pub(super) fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    ) -> Self {
        CommandEncoder { device, buffer }
    }
}

#[hidden_trait::expose]
impl crate::traits::SyncCommandEncoder for CommandEncoder {
    #[inline(always)]
    fn barrier(&mut self, _prev_stages: PipelineStages, _next_stages: PipelineStages) {}

    #[inline(always)]
    fn init_image(
        &mut self,
        _prev_use: PipelineStages,
        _next_use: PipelineStages,
        _image: impl AsRef<Image>,
    ) {
    }
}

#[hidden_trait::expose]
impl crate::traits::CommandEncoder for CommandEncoder {
    #[inline(always)]
    fn copy(&mut self) -> CopyCommandEncoder<'_> {
        let encoder = self
            .buffer
            .blitCommandEncoder()
            .expect("Failed to create blit command encoder");
        CopyCommandEncoder {
            device: &self.device,
            encoder: encoder.to_owned(),
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    fn compute(&mut self) -> ComputeCommandEncoder<'_> {
        let encoder = self
            .buffer
            .computeCommandEncoder()
            .expect("Failed to create compute command encoder");
        ComputeCommandEncoder {
            device: &self.device,
            encoder: encoder.to_owned(),
            bindings: None,
            workgroup_size: None,
            _marker: PhantomData,
        }
    }

    fn render(&mut self, desc: RenderPassDesc) -> RenderCommandEncoder<'_> {
        let mdesc = MTLRenderPassDescriptor::new();
        let color_attachments = mdesc.colorAttachments();
        for (idx, color) in desc.color_attachments.iter().enumerate() {
            let format = color.image.format();
            debug_assert!(format.is_color());

            let attachment = MTLRenderPassColorAttachmentDescriptor::new();
            attachment.setTexture(Some(color.image.metal()));
            attachment.setLoadAction(match color.load {
                LoadOp::Load => MTLLoadAction::Load,
                LoadOp::Clear(ClearColor(r, g, b, a)) => {
                    attachment.setClearColor(MTLClearColor {
                        red: r.into(),
                        green: g.into(),
                        blue: b.into(),
                        alpha: a.into(),
                    });
                    MTLLoadAction::Clear
                }
                LoadOp::DontCare => MTLLoadAction::DontCare,
            });
            attachment.setStoreAction(match color.store {
                StoreOp::Store => MTLStoreAction::Store,
                StoreOp::DontCare => MTLStoreAction::DontCare,
            });
            attachment.setLevel(0);
            attachment.setSlice(0);
            unsafe {
                color_attachments.setObject_atIndexedSubscript(Some(&attachment), idx);
            }
        }

        if let Some(depth) = desc.depth_stencil_attachment {
            let format = depth.image.format();
            debug_assert!(format.is_depth() || format.is_stencil());

            if format.is_depth() {
                let attachment = mdesc.depthAttachment();
                attachment.setTexture(Some(depth.image.metal()));
                attachment.setLoadAction(match depth.load {
                    LoadOp::Load => MTLLoadAction::Load,
                    LoadOp::Clear(ClearDepthStencil { depth, .. }) => {
                        attachment.setClearDepth(depth.into());
                        MTLLoadAction::Clear
                    }
                    LoadOp::DontCare => MTLLoadAction::DontCare,
                });
                attachment.setStoreAction(match depth.store {
                    StoreOp::Store => MTLStoreAction::Store,
                    StoreOp::DontCare => MTLStoreAction::DontCare,
                });
                attachment.setLevel(0);
                attachment.setSlice(0);
            }
            if format.is_stencil() {
                let attachment = mdesc.stencilAttachment();
                attachment.setTexture(Some(depth.image.metal()));
                attachment.setLoadAction(match depth.load {
                    LoadOp::Load => MTLLoadAction::Load,
                    LoadOp::Clear(ClearDepthStencil { stencil, .. }) => {
                        attachment.setClearStencil(stencil.into());
                        MTLLoadAction::Clear
                    }
                    LoadOp::DontCare => MTLLoadAction::DontCare,
                });
                attachment.setStoreAction(match depth.store {
                    StoreOp::Store => MTLStoreAction::Store,
                    StoreOp::DontCare => MTLStoreAction::DontCare,
                });
                attachment.setLevel(0);
                attachment.setSlice(0);
            }
        }

        let encoder = self
            .buffer
            .renderCommandEncoderWithDescriptor(&mdesc)
            .expect("Failed to create render command encoder");
        let mut render = RenderCommandEncoder {
            encoder: encoder.to_owned(),
            primitive: MTLPrimitiveType::Triangle,
            index_buffer: None,
            index_buffer_offset: 0,
            vertex_bindings: None,
            fragment_bindings: None,
            vertex_buffers_count: 0,
            _marker: PhantomData,
        };

        // Default the viewport and scissor to the full extent of the first
        // attachment (first color attachment, else depth-stencil), matching
        // an explicit `with_viewport`/`with_scissor` call to the same effect.
        let first_attachment = desc
            .color_attachments
            .first()
            .map(|color| color.image)
            .or_else(|| desc.depth_stencil_attachment.as_ref().map(|depth| depth.image));
        if let Some(image) = first_attachment {
            let extent = image.extent().expect_2d();
            render.with_viewport(Offset3::ZERO, extent.to_3d().cast_as_f32());
            render.with_scissor(Offset2::ZERO, extent);
        }

        render
    }

    #[inline(always)]
    fn acceleration_structure(&mut self) -> AccelerationStructureCommandEncoder<'_> {
        let encoder = self
            .buffer
            .accelerationStructureCommandEncoder()
            .expect("Failed to create acceleration structure command encoder");
        AccelerationStructureCommandEncoder {
            device: &mut self.device,
            encoder: encoder.to_owned(),
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    fn present(&mut self, frame: Frame, _prev_use: PipelineStages) {
        if let Some(drawable) = frame.drawable() {
            self.buffer.presentDrawable(drawable);
        }
    }

    #[inline(always)]
    fn finish(self) -> CommandBuffer {
        CommandBuffer {
            buffer: self.buffer,
        }
    }
}

pub struct CopyCommandEncoder<'a> {
    device: &'a ProtocolObject<dyn MTLDevice>,
    encoder: Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>,
    _marker: PhantomData<&'a mut CommandBuffer>,
}

impl Drop for CopyCommandEncoder<'_> {
    #[inline(always)]
    fn drop(&mut self) {
        self.encoder.endEncoding();
    }
}

#[hidden_trait::expose]
impl crate::traits::SyncCommandEncoder for CopyCommandEncoder<'_> {
    #[inline(always)]
    fn barrier(&mut self, _prev_stages: PipelineStages, _next_stages: PipelineStages) {}

    #[inline(always)]
    fn init_image(
        &mut self,
        _prev_use: PipelineStages,
        _next_use: PipelineStages,
        _image: impl AsRef<Image>,
    ) {
    }
}

#[hidden_trait::expose]
impl crate::traits::CopyCommandEncoder for CopyCommandEncoder<'_> {
    #[inline]
    fn copy_buffer_to_image<const N: usize>(
        &mut self,
        src: &impl AsBufferSlice,
        shape: TexelShape<N>,
        dst: &impl AsImageSlice<N>,
    ) {
        const {
            assert!(
                N >= 1 && N <= 3,
                "copy_buffer_to_image: TexelShape rank must be 1..=3"
            )
        };

        let src = src.as_buffer_slice();
        let dst = dst.as_image_slice();
        dst.assert_fits_image();

        assert_eq!(
            shape.extent, dst.extent,
            "copy_buffer_to_image: TexelShape extent must match the destination image slice extent"
        );

        let (extent, resolved) = pad_shape_to_3(&shape);

        let block_size = dst.image.format().block_size();
        assert_eq!(
            resolved[0], block_size,
            "copy_buffer_to_image: element stride must equal the format's tightly-packed size ({block_size}) - native copy commands can't vary it; use Cvt for padded/strided element transfers"
        );
        let bytes_per_line = resolved[1];
        let bytes_per_plane = resolved[2];

        let layers = dst.layers;
        debug_assert!(layers.end > layers.start);
        debug_assert!(layers.end == layers.start + 1);

        let dst_offset = dst.offset.to_3d();

        unsafe {
            self.encoder.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
            src.buffer().metal(),
                src.offset() as NSUInteger,
                bytes_per_line as NSUInteger,
                bytes_per_plane as NSUInteger,
                MTLSize {
                    width: extent.width() as NSUInteger,
                    height: extent.height() as NSUInteger,
                    depth: extent.depth() as NSUInteger,
                },
                dst.image.metal(),
                layers.start as NSUInteger,
                dst.level as NSUInteger,
                MTLOrigin {
                    x: dst_offset.x() as NSUInteger,
                    y: dst_offset.y() as NSUInteger,
                    z: dst_offset.z() as NSUInteger,
                },
            );
        }
    }

    #[inline]
    fn copy_image_region<const N: usize>(
        &mut self,
        src: &impl AsImageSlice<N>,
        dst: &impl AsImageSlice<N>,
    ) {
        let src = src.as_image_slice();
        src.assert_fits_image();
        let dst = dst.as_image_slice();
        dst.assert_fits_image();

        assert_eq!(
            src.extent, dst.extent,
            "copy_image_region: source and destination image slices must have the same extent"
        );

        let src_layers = src.layers;
        let dst_layers = dst.layers;
        assert_eq!(
            src_layers.end - src_layers.start,
            dst_layers.end - dst_layers.start,
            "copy_image_region: source and destination image slices must have the same layer count"
        );
        let layers = src_layers.end - src_layers.start;

        let src_offset = src.offset.to_3d();
        let dst_offset = dst.offset.to_3d();
        let extent = src.extent.to_3d();

        // If copying entire slices, use optimized method
        if src_offset == Offset3::ZERO
            && dst_offset == Offset3::ZERO
            && src.image.extent().into_3d() == extent
            && dst.image.extent().into_3d() == extent
        {
            unsafe {
                self.encoder.copyFromTexture_sourceSlice_sourceLevel_toTexture_destinationSlice_destinationLevel_sliceCount_levelCount(
                    src.image.metal(),
                    src_layers.start as NSUInteger,
                    src.level as NSUInteger,
                    dst.image.metal(),
                    dst_layers.start as NSUInteger,
                    dst.level as NSUInteger,
                    layers as NSUInteger,
                    1,
                );
            }

            return;
        }

        // Otherwise, copy slice by slice, level by level
        for layer in 0..layers {
            unsafe {
                self.encoder.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                    src.image.metal(),
                    (src_layers.start + layer) as NSUInteger,
                    src.level as NSUInteger,
                    MTLOrigin {
                        x: src_offset.x() as NSUInteger,
                        y: src_offset.y() as NSUInteger,
                        z: src_offset.z() as NSUInteger,
                    },
                    MTLSize {
                        width: extent.width() as NSUInteger,
                        height: extent.height() as NSUInteger,
                        depth: extent.depth() as NSUInteger,
                    },
                    dst.image.metal(),
                    (dst_layers.start + layer) as NSUInteger,
                    dst.level as NSUInteger,
                    MTLOrigin {
                        x: dst_offset.x() as NSUInteger,
                        y: dst_offset.y() as NSUInteger,
                        z: dst_offset.z() as NSUInteger,
                    },
                );
            }
        }
    }

    #[inline]
    fn copy_image_to_buffer<const N: usize>(
        &mut self,
        src: &impl AsImageSlice<N>,
        dst: &impl AsBufferSlice,
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

        let (extent, resolved) = pad_shape_to_3(&shape);

        let block_size = src.image.format().block_size();
        assert_eq!(
            resolved[0], block_size,
            "copy_image_to_buffer: element stride must equal the format's tightly-packed size ({block_size}) - native copy commands can't vary it; use Cvt for padded/strided element transfers"
        );
        let bytes_per_line = resolved[1];
        let bytes_per_plane = resolved[2];

        let layers = src.layers;
        debug_assert!(layers.end > layers.start);
        debug_assert!(layers.end == layers.start + 1);

        let src_offset = src.offset.to_3d();

        unsafe {
            self.encoder.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                src.image.metal(),
                layers.start as NSUInteger,
                src.level as NSUInteger,
                MTLOrigin {
                    x: src_offset.x() as NSUInteger,
                    y: src_offset.y() as NSUInteger,
                    z: src_offset.z() as NSUInteger,
                },
                MTLSize {
                    width: extent.width() as NSUInteger,
                    height: extent.height() as NSUInteger,
                    depth: extent.depth() as NSUInteger,
                },
                dst.buffer().metal(),
                dst.offset() as NSUInteger,
                bytes_per_line as NSUInteger,
                bytes_per_plane as NSUInteger,
            );
        }
    }

    #[inline]
    fn fill_buffer(&mut self, slice: impl AsBufferSlice, byte: u8) {
        let slice = slice.as_buffer_slice();

        self.encoder.fillBuffer_range_value(
            slice.buffer.metal(),
            NSRange {
                location: slice.offset as NSUInteger,
                length: slice.size as NSUInteger,
            },
            byte,
        );
    }

    #[inline]
    fn write_buffer_raw(&mut self, slice: impl AsBufferSlice, data: &[u8]) {
        if data.is_empty() {
            return;
        }

        let slice = slice.as_buffer_slice();
        assert_eq!(
            data.len(),
            slice.size,
            "Data length does not match buffer slice size"
        );

        let staging = unsafe {
            self.device.newBufferWithBytes_length_options(
                NonNull::from(data).cast(),
                data.len() as NSUInteger,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .expect("Failed to allocate staging buffer");

        unsafe {
            self.encoder
                .copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &staging,
                    0,
                    slice.buffer.metal(),
                    slice.offset as NSUInteger,
                    data.len() as NSUInteger,
                );
        }
    }

    #[inline(always)]
    fn write_buffer(&mut self, slice: impl AsBufferSlice, data: &impl bytemuck::NoUninit) {
        self.write_buffer_raw(slice, bytemuck::bytes_of(data))
    }

    /// Writes data to the buffer.
    #[inline(always)]
    fn write_buffer_slice(&mut self, slice: impl AsBufferSlice, data: &[impl bytemuck::NoUninit]) {
        self.write_buffer_raw(slice, bytemuck::cast_slice(data))
    }

    /// Writes data and emits the surrounding barriers. `prev_use` = stages
    /// that previously used the buffer, `next_use` = stages that will use it
    /// next.
    #[inline(always)]
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

    #[inline(always)]
    fn copy_buffer_to_buffer(&mut self, src: &impl AsBufferSlice, dst: &impl AsBufferSlice) {
        let src = src.as_buffer_slice();
        let dst = dst.as_buffer_slice();

        assert_eq!(
            src.size(),
            dst.size(),
            "copy_buffer_to_buffer requires src and dst to have the same size (mirrors slice::copy_from_slice)"
        );

        unsafe {
            self.encoder
                .copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    src.buffer().metal(),
                    src.offset() as NSUInteger,
                    dst.buffer().metal(),
                    dst.offset() as NSUInteger,
                    src.size() as NSUInteger,
                );
        }
    }
}

pub struct ComputeCommandEncoder<'a> {
    device: &'a ProtocolObject<dyn MTLDevice>,
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    bindings: Option<Arc<Bindings>>,
    workgroup_size: Option<[u32; 3]>,
    _marker: PhantomData<&'a mut CommandBuffer>,
}

impl ComputeCommandEncoder<'_> {
    #[doc(hidden)]
    #[inline(always)]
    pub fn bindings(&self) -> Option<&Bindings> {
        self.bindings.as_deref()
    }

    #[doc(hidden)]
    #[inline(always)]
    pub fn metal(&self) -> &ProtocolObject<dyn MTLComputeCommandEncoder> {
        self.encoder.as_ref()
    }
}

impl Drop for ComputeCommandEncoder<'_> {
    #[inline(always)]
    fn drop(&mut self) {
        self.encoder.endEncoding();
    }
}

#[hidden_trait::expose]
impl traits::SyncCommandEncoder for ComputeCommandEncoder<'_> {
    #[inline(always)]
    fn barrier(&mut self, _prev_stages: PipelineStages, _next_stages: PipelineStages) {}

    #[inline(always)]
    fn init_image(
        &mut self,
        _prev_use: PipelineStages,
        _next_use: PipelineStages,
        _image: impl AsRef<Image>,
    ) {
    }
}

#[hidden_trait::expose]
impl traits::ComputeCommandEncoder for ComputeCommandEncoder<'_> {
    #[inline(always)]
    fn with_pipeline(&mut self, pipeline: &crate::backend::ComputePipeline) {
        pipeline.metal();
        self.encoder.setComputePipelineState(pipeline.metal());
        self.bindings = pipeline.bindings();
        self.workgroup_size = pipeline.workgroup_size();
    }

    #[inline(always)]
    fn with_arguments(&mut self, group: u32, arguments: &impl Arguments) {
        arguments.bind_compute(group, self);
    }

    #[inline(always)]
    fn with_constants(&mut self, constants: &impl DeviceRepr) {
        let data = constants.as_repr();
        let data_bytes = bytemuck::bytes_of(&data);

        unsafe {
            self.encoder.setBytes_length_atIndex(
                NonNull::from(data_bytes).cast(),
                data_bytes.len() as NSUInteger,
                0,
            );
        }
    }

    #[inline(always)]
    fn dispatch(&mut self, groups: Extent3) {
        let group_size = self.workgroup_size.unwrap_or([1, 1, 1]);

        unsafe {
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: groups.width() as usize,
                    height: groups.height() as usize,
                    depth: groups.depth() as usize,
                },
                MTLSize {
                    width: group_size[0] as usize,
                    height: group_size[1] as usize,
                    depth: group_size[2] as usize,
                },
            );
        }
    }
}

pub struct RenderCommandEncoder<'a> {
    encoder: Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>,
    primitive: MTLPrimitiveType,
    index_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    index_buffer_offset: NSUInteger,
    vertex_bindings: Option<Arc<Bindings>>,
    fragment_bindings: Option<Arc<Bindings>>,
    vertex_buffers_count: u32,
    _marker: PhantomData<&'a mut CommandBuffer>,
}

impl RenderCommandEncoder<'_> {
    #[doc(hidden)]
    #[inline(always)]
    pub fn vertex_bindings(&self) -> Option<&Bindings> {
        self.vertex_bindings.as_deref()
    }

    #[doc(hidden)]
    #[inline(always)]
    pub fn fragment_bindings(&self) -> Option<&Bindings> {
        self.fragment_bindings.as_deref()
    }

    #[doc(hidden)]
    #[inline(always)]
    pub fn metal(&self) -> &ProtocolObject<dyn MTLRenderCommandEncoder> {
        &self.encoder
    }
}

impl Drop for RenderCommandEncoder<'_> {
    #[inline(always)]
    fn drop(&mut self) {
        self.encoder.endEncoding();
    }
}

#[hidden_trait::expose]
impl crate::traits::RenderCommandEncoder for RenderCommandEncoder<'_> {
    #[inline(always)]
    fn with_pipeline(&mut self, pipeline: &RenderPipeline) {
        self.encoder.setRenderPipelineState(pipeline.metal());
        self.primitive = pipeline.primitive();
        self.vertex_bindings = pipeline.vertex_bindings();
        self.fragment_bindings = pipeline.fragment_bindings();
        self.vertex_buffers_count = pipeline.vertex_buffers_count();
    }

    #[inline(always)]
    fn with_viewport(&mut self, offset: Offset3<f32>, extent: Extent3<f32>) {
        let viewport = MTLViewport {
            originX: offset.x().into(),
            originY: offset.y().into(),
            width: extent.width().into(),
            height: extent.height().into(),
            znear: offset.z().into(),
            zfar: (offset.z() + extent.depth()).into(),
        };
        self.encoder.setViewport(viewport);
    }

    #[inline(always)]
    fn with_scissor(&mut self, offset: Offset2<i32>, extent: Extent2<u32>) {
        debug_assert!(offset.x() >= 0);
        debug_assert!(offset.y() >= 0);

        let scissor = MTLScissorRect {
            x: offset.x() as NSUInteger,
            y: offset.y() as NSUInteger,
            width: extent.width() as NSUInteger,
            height: extent.height() as NSUInteger,
        };
        self.encoder.setScissorRect(scissor);
    }

    /// Sets arguments group for the current pipeline.
    #[inline(always)]
    fn with_arguments(&mut self, group: u32, arguments: &impl Arguments) {
        arguments.bind_render(group, self);
    }

    /// Sets constants for the current pipeline.
    #[inline]
    fn with_constants(&mut self, constants: &impl DeviceRepr) {
        let data = constants.as_repr();
        let data_bytes = bytemuck::bytes_of(&data);

        if let Some(vb) = &self.vertex_bindings {
            if let Some(slot) = vb.push_constants {
                unsafe {
                    self.encoder.setVertexBytes_length_atIndex(
                        NonNull::from(data_bytes).cast(),
                        data_bytes.len() as NSUInteger,
                        slot as NSUInteger,
                    );
                }
            }
        }

        if let Some(fb) = &self.fragment_bindings {
            if let Some(slot) = fb.push_constants {
                unsafe {
                    self.encoder.setFragmentBytes_length_atIndex(
                        NonNull::from(data_bytes).cast(),
                        data_bytes.len() as NSUInteger,
                        slot as NSUInteger,
                    );
                }
            }
        }
    }

    /// Bind vertex buffer to the current pipeline.
    #[inline]
    fn bind_vertex_buffers(&mut self, start: u32, buffers: &[(impl AsBufferSlice)]) {
        let slices = buffers
            .iter()
            .map(|slice| slice.as_buffer_slice())
            .collect::<SmallVec<[_; 8]>>();

        let (buffers, offsets) = slices
            .iter()
            .map(|slice| {
                let offset = slice.offset() as NSUInteger;
                let buffer = slice.buffer().metal();
                (Some(buffer), offset)
            })
            .unzip::<_, _, SmallVec<[_; 8]>, SmallVec<[_; 8]>>();

        let first = self.vertex_buffers_count + start;

        unsafe {
            self.encoder.setVertexBuffers_offsets_withRange(
                NonNull::from(&buffers[..]).cast(),
                NonNull::from(&offsets[..]).cast(),
                NSRange {
                    location: first as NSUInteger,
                    length: buffers.len() as NSUInteger,
                },
            );
        }
    }

    /// Bind index buffer to the current pipeline.
    #[inline(always)]
    fn bind_index_buffer(&mut self, buffer: impl AsBufferSlice) {
        let buffer_slice = buffer.as_buffer_slice();

        self.index_buffer = Some(Retained::from(buffer_slice.buffer.metal()));
        self.index_buffer_offset = buffer_slice.offset as NSUInteger;
    }

    #[inline]
    fn draw(&mut self, vertices: impl Into<Range<u32>>, instances: impl Into<Range<u32>>) {
        let vertices = vertices.into();
        let instances = instances.into();

        if vertices.end <= vertices.start {
            // Rendering no vertices is a no-op
            return;
        }

        if instances.end <= instances.start {
            // Rendering no instances is a no-op
            return;
        }

        let first_vertex = vertices.start as NSUInteger;
        let vertex_count = (vertices.end - vertices.start) as NSUInteger;
        let first_instance = instances.start as NSUInteger;
        let instance_count = (instances.end - instances.start) as NSUInteger;

        match (first_instance, instance_count) {
            (0, 1) => {
                // Rendering single instance
                unsafe {
                    self.encoder.drawPrimitives_vertexStart_vertexCount(
                        self.primitive,
                        first_vertex,
                        vertex_count,
                    );
                }
            }
            (0, _) => {
                // Rendering multiple instances
                unsafe {
                    self.encoder
                        .drawPrimitives_vertexStart_vertexCount_instanceCount(
                            self.primitive,
                            first_vertex,
                            vertex_count,
                            instance_count,
                        );
                }
            }
            _ => {
                // Rendering multiple instances with a base instance
                unsafe {
                    self.encoder
                        .drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                            self.primitive,
                            first_vertex,
                            vertex_count,
                            instance_count,
                            first_instance,
                        );
                }
            }
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

        debug_assert!(vertex_offset >= 0);

        let index_buffer = self.index_buffer.as_deref().unwrap();

        if indices.end <= indices.start {
            // Rendering no indices is a no-op
            return;
        }
        if instances.end <= instances.start {
            // Rendering no instances is a no-op
            return;
        }

        let first_vertex = vertex_offset as NSInteger;
        let first_index = indices.start as NSUInteger;
        let indices_count = (indices.end - indices.start) as NSUInteger;
        let first_instance = instances.start as NSUInteger;
        let instance_count = (instances.end - instances.start) as NSUInteger;

        match (first_vertex, first_instance, instance_count) {
            (0, 0, 1) => {
                // Rendering single instance
                unsafe {
                    self.encoder
                        .drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset(
                            self.primitive,
                            indices_count,
                            MTLIndexType::UInt32,
                            index_buffer,
                            (self.index_buffer_offset + (first_index * 4)) as NSUInteger,
                        );
                }
            }
            (0, 0, _) => {
                // Rendering multiple instances
                unsafe {
                    self.encoder
                        .drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount(
                            self.primitive,
                            indices_count,
                            MTLIndexType::UInt32,
                            index_buffer,
                            (self.index_buffer_offset + (first_index * 4)) as NSUInteger,
                            instance_count,
                        );
                }
            }
            (_, _, _) => {
                // Rendering multiple instances with a base instance
                unsafe {
                    self.encoder
                        .drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                            self.primitive,
                            indices_count,
                            MTLIndexType::UInt32,
                            index_buffer,
                            (self.index_buffer_offset + (first_index * 4)) as NSUInteger,
                            instance_count,
                            first_vertex,
                            first_instance,
                        );
                }
            }
        }
    }
}

pub struct AccelerationStructureCommandEncoder<'a> {
    device: &'a ProtocolObject<dyn MTLDevice>,
    encoder: Retained<ProtocolObject<dyn MTLAccelerationStructureCommandEncoder>>,
    _marker: PhantomData<&'a mut CommandBuffer>,
}

#[hidden_trait::expose]
impl crate::traits::AccelerationStructureCommandEncoder
    for AccelerationStructureCommandEncoder<'_>
{
    fn build_blas(&mut self, blas: &Blas, desc: BlasBuildDesc, scratch: impl AsBufferSlice) {
        let mdesc = MTLPrimitiveAccelerationStructureDescriptor::descriptor();
        if let AccelerationStructurePerformance::FastBuild = desc.performance {
            mdesc.setUsage(MTLAccelerationStructureUsage::PreferFastBuild);
        }

        let mut geometry_descs = Vec::<Retained<MTLAccelerationStructureGeometryDescriptor>>::new();

        for geometry in desc.geometry {
            match geometry {
                BlasGeometryDesc::Triangles(triangles) => {
                    let mdesc = MTLAccelerationStructureTriangleGeometryDescriptor::descriptor();

                    mdesc.setOpaque(triangles.opaque);

                    let mut count =
                        (triangles.vertices.size / triangles.vertex_stride) as NSUInteger;

                    if let Some(indices) = &triangles.indices {
                        mdesc.setIndexType(MTLIndexType::UInt32);
                        unsafe {
                            mdesc.setIndexBufferOffset(indices.offset as NSUInteger);
                        }
                        mdesc.setIndexBuffer(Some(indices.buffer().metal()));
                        count = indices.size() as NSUInteger / 4;
                    }

                    mdesc.setTriangleCount(count as NSUInteger);

                    if let Some(transform) = &triangles.transform {
                        unsafe {
                            mdesc.setTransformationMatrixBufferOffset(
                                transform.offset() as NSUInteger
                            );
                            mdesc.setTransformationMatrixBuffer(Some(transform.buffer().metal()));
                        }
                    }

                    mdesc.setVertexFormat(triangles.vertex_format.try_into_metal().unwrap());
                    mdesc.setVertexStride(triangles.vertex_stride as NSUInteger);
                    unsafe {
                        mdesc.setVertexBufferOffset(triangles.vertices.offset() as NSUInteger);
                    }
                    mdesc.setVertexBuffer(Some(triangles.vertices.buffer().metal()));

                    geometry_descs.push(mdesc.into_super());
                }
                BlasGeometryDesc::AABBs(aabbs) => {
                    let mdesc = MTLAccelerationStructureBoundingBoxGeometryDescriptor::descriptor();

                    mdesc.setOpaque(aabbs.opaque);

                    mdesc
                        .setBoundingBoxCount((aabbs.boxes.size() / aabbs.box_stride) as NSUInteger);
                    unsafe {
                        mdesc.setBoundingBoxStride(aabbs.box_stride as NSUInteger);
                        mdesc.setBoundingBoxBufferOffset(aabbs.boxes.offset() as NSUInteger);
                    }
                    mdesc.setBoundingBoxBuffer(Some(aabbs.boxes.buffer().metal()));

                    geometry_descs.push(mdesc.into_super());
                }
            }
        }

        let geometry_descs = NSArray::from_retained_slice(&geometry_descs);

        mdesc.setGeometryDescriptors(Some(&geometry_descs));
    }

    fn build_tlas(&mut self, tlas: &Tlas, desc: TlasBuildDesc, scratch: impl AsBufferSlice) {}
}
