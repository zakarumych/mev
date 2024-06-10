use std::{marker::PhantomData, ops::Range, sync::Arc};

use metal::NSUInteger;
use objc::{msg_send, Message};
use smallvec::SmallVec;

use crate::{
    generic::{
        AccelerationStructureBuildFlags, AccelerationStructurePerformance, Arguments,
        AsBufferSlice, BlasBuildDesc, BlasGeometryDesc, ClearColor, ClearDepthStencil, DeviceRepr,
        Extent2, Extent3, LoadOp, Offset2, Offset3, OutOfMemory, PipelineStages, RenderPassDesc,
        StoreOp, TlasBuildDesc,
    },
    traits,
};

use super::{
    from::TryIntoMetal, out_of_bounds, shader::Bindings, Blas, Buffer, Frame, Image,
    RenderPipeline, Tlas,
};

pub struct CommandBuffer {
    buffer: metal::CommandBuffer,
}

impl CommandBuffer {
    pub(super) fn commit(self) {
        self.buffer.commit();
    }
}

pub struct CommandEncoder {
    device: metal::Device,
    buffer: metal::CommandBuffer,
}

impl CommandEncoder {
    pub(super) fn new(device: metal::Device, buffer: metal::CommandBuffer) -> Self {
        CommandEncoder { device, buffer }
    }
}

#[hidden_trait::expose]
impl crate::traits::CommandEncoder for CommandEncoder {
    #[inline(always)]
    fn barrier(&mut self, _after: PipelineStages, _before: PipelineStages) {}

    #[inline(always)]
    fn init_image(&mut self, _after: PipelineStages, _before: PipelineStages, _image: &Image) {}

    #[inline(always)]
    fn copy(&mut self) -> CopyCommandEncoder {
        let encoder = self.buffer.new_blit_command_encoder();
        CopyCommandEncoder {
            device: &mut self.device,
            encoder: encoder.to_owned(),
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    fn compute(&mut self) -> ComputeCommandEncoder<'_> {
        let encoder = self.buffer.new_compute_command_encoder();
        ComputeCommandEncoder {
            device: &mut self.device,
            encoder: encoder.to_owned(),
            bindings: None,
            workgroup_size: None,
            _marker: PhantomData,
        }
    }

    fn render(&mut self, desc: RenderPassDesc) -> RenderCommandEncoder<'_> {
        let mdesc = metal::RenderPassDescriptor::new();
        let color_attachments = mdesc.color_attachments();
        for (idx, color) in desc.color_attachments.iter().enumerate() {
            let format = color.image.format();
            debug_assert!(format.is_color());

            let attachment = metal::RenderPassColorAttachmentDescriptor::new();
            attachment.set_texture(Some(color.image.metal()));
            attachment.set_load_action(match color.load {
                LoadOp::Load => metal::MTLLoadAction::Load,
                LoadOp::Clear(ClearColor(r, g, b, a)) => {
                    attachment.set_clear_color(metal::MTLClearColor {
                        red: r.into(),
                        green: g.into(),
                        blue: b.into(),
                        alpha: a.into(),
                    });
                    metal::MTLLoadAction::Clear
                }
                LoadOp::DontCare => metal::MTLLoadAction::DontCare,
            });
            attachment.set_store_action(match color.store {
                StoreOp::Store => metal::MTLStoreAction::Store,
                StoreOp::DontCare => metal::MTLStoreAction::DontCare,
            });
            attachment.set_level(0);
            attachment.set_slice(0);
            color_attachments.set_object_at(idx as _, Some(&attachment));
        }

        if let Some(depth) = desc.depth_stencil_attachment {
            let format = depth.image.format();
            debug_assert!(format.is_depth() || format.is_stencil());

            if format.is_depth() {
                let attachment = mdesc.depth_attachment().unwrap();
                attachment.set_texture(Some(depth.image.metal()));
                attachment.set_load_action(match depth.load {
                    LoadOp::Load => metal::MTLLoadAction::Load,
                    LoadOp::Clear(ClearDepthStencil { depth, .. }) => {
                        attachment.set_clear_depth(depth.into());
                        metal::MTLLoadAction::Clear
                    }
                    LoadOp::DontCare => metal::MTLLoadAction::DontCare,
                });
                attachment.set_store_action(match depth.store {
                    StoreOp::Store => metal::MTLStoreAction::Store,
                    StoreOp::DontCare => metal::MTLStoreAction::DontCare,
                });
                attachment.set_level(0);
                attachment.set_slice(0);
            }
            if format.is_stencil() {
                let attachment = mdesc.stencil_attachment().unwrap();
                attachment.set_texture(Some(depth.image.metal()));
                attachment.set_load_action(match depth.load {
                    LoadOp::Load => metal::MTLLoadAction::Load,
                    LoadOp::Clear(ClearDepthStencil { stencil, .. }) => {
                        attachment.set_clear_stencil(stencil.into());
                        metal::MTLLoadAction::Clear
                    }
                    LoadOp::DontCare => metal::MTLLoadAction::DontCare,
                });
                attachment.set_store_action(match depth.store {
                    StoreOp::Store => metal::MTLStoreAction::Store,
                    StoreOp::DontCare => metal::MTLStoreAction::DontCare,
                });
                attachment.set_level(0);
                attachment.set_slice(0);
            }
        }

        let encoder = self.buffer.new_render_command_encoder(&mdesc);
        RenderCommandEncoder {
            encoder: encoder.to_owned(),
            primitive: metal::MTLPrimitiveType::Triangle,
            index_buffer: None,
            index_buffer_offset: 0,
            vertex_bindings: None,
            fragment_bindings: None,
            vertex_buffers_count: 0,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    fn acceleration_structure(&mut self) -> AccelerationStructureCommandEncoder<'_> {
        let encoder = self.buffer.new_acceleration_structure_command_encoder();
        AccelerationStructureCommandEncoder {
            device: &mut self.device,
            encoder: encoder.to_owned(),
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    fn present(&mut self, frame: Frame, _after: PipelineStages) {
        self.buffer.present_drawable(frame.drawable());
    }

    #[inline(always)]
    fn finish(self) -> Result<CommandBuffer, OutOfMemory> {
        Ok(CommandBuffer {
            buffer: self.buffer,
        })
    }
}

pub struct CopyCommandEncoder<'a> {
    device: &'a mut metal::DeviceRef,
    encoder: metal::BlitCommandEncoder,
    _marker: PhantomData<&'a mut CommandBuffer>,
}

impl Drop for CopyCommandEncoder<'_> {
    #[inline(always)]
    fn drop(&mut self) {
        self.encoder.end_encoding();
    }
}

#[hidden_trait::expose]
impl crate::traits::CopyCommandEncoder for CopyCommandEncoder<'_> {
    #[inline(always)]
    fn barrier(&mut self, _after: PipelineStages, _before: PipelineStages) {}

    #[inline(always)]
    fn init_image(&mut self, _after: PipelineStages, _before: PipelineStages, _image: &Image) {}

    #[cfg_attr(feature = "inline-more", inline(always))]
    fn copy_buffer_to_image(
        &mut self,
        src: &Buffer,
        start: usize,
        bytes_per_line: usize,
        bytes_per_plane: usize,
        dst: &Image,
        offset: Offset3<u32>,
        extent: Extent3<u32>,
        layers: Range<u32>,
        level: u32,
    ) {
        debug_assert!(layers.end > layers.start);
        debug_assert!(layers.end == layers.start + 1);

        self.encoder.copy_from_buffer_to_texture(
            src.metal(),
            start as NSUInteger,
            bytes_per_line as NSUInteger,
            bytes_per_plane as NSUInteger,
            metal::MTLSize {
                width: extent.width() as NSUInteger,
                height: extent.height() as NSUInteger,
                depth: extent.depth() as NSUInteger,
            },
            dst.metal(),
            layers.start as NSUInteger,
            level as NSUInteger,
            metal::MTLOrigin {
                x: offset.x() as NSUInteger,
                y: offset.y() as NSUInteger,
                z: offset.z() as NSUInteger,
            },
            metal::MTLBlitOption::empty(),
        );
    }

    #[cfg_attr(feature = "inline-more", inline(always))]
    fn copy_image_region(
        &mut self,
        src: &Image,
        src_level: u32,
        src_base_layer: u32,
        src_offset: Offset3<u32>,
        dst: &Image,
        dst_level: u32,
        dst_base_layer: u32,
        dst_offset: Offset3<u32>,
        extent: Extent3<u32>,
        layers: u32,
    ) {
        use objc::{sel, sel_impl};

        // If copying entire slices, use optimized method
        if src_offset == Offset3::ZERO
            && dst_offset == Offset3::ZERO
            && src.dimensions().into_3d() == extent
            && dst.dimensions().into_3d() == extent
        {
            unsafe {
                let () = msg_send![self.encoder,
                    copyFromTexture: src.metal()
                    sourceSlice: src_base_layer as NSUInteger
                    sourceLevel: src_level as NSUInteger
                    toTexture: dst.metal()
                    destinationSlice: dst_base_layer as NSUInteger
                    destinationLevel: dst_level as NSUInteger
                    sliceCount: layers as NSUInteger
                    levelCount: 1
                ];
            }

            return;
        }

        // Otherwise, copy slice by slice, level by level
        for layer in 0..layers {
            self.encoder.copy_from_texture(
                src.metal(),
                (src_base_layer + layer) as NSUInteger,
                src_level as NSUInteger,
                metal::MTLOrigin {
                    x: src_offset.x() as NSUInteger,
                    y: src_offset.y() as NSUInteger,
                    z: src_offset.z() as NSUInteger,
                },
                metal::MTLSize {
                    width: extent.width() as NSUInteger,
                    height: extent.height() as NSUInteger,
                    depth: extent.depth() as NSUInteger,
                },
                dst.metal(),
                (dst_base_layer + layer) as NSUInteger,
                dst_level as NSUInteger,
                metal::MTLOrigin {
                    x: dst_offset.x() as NSUInteger,
                    y: dst_offset.y() as NSUInteger,
                    z: dst_offset.z() as NSUInteger,
                },
            );
        }
    }

    #[cfg_attr(feature = "inline-more", inline(always))]
    fn write_buffer_raw(&mut self, buffer: impl AsBufferSlice, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        let buffer_slice = buffer.as_buffer_slice();
        if data.len() > buffer_slice.size {
            out_of_bounds();
        }

        let staged = self.device.new_buffer_with_data(
            data.as_ptr().cast(),
            data.len() as NSUInteger,
            metal::MTLResourceOptions::StorageModeShared,
        );

        self.encoder.copy_from_buffer(
            &staged,
            0,
            buffer_slice.buffer.metal(),
            buffer_slice.offset as NSUInteger,
            data.len() as NSUInteger,
        );
    }

    #[cfg_attr(feature = "inline-more", inline(always))]
    fn write_buffer(&mut self, slice: impl AsBufferSlice, data: &impl bytemuck::Pod) {
        self.write_buffer_slice(slice, bytemuck::bytes_of(data))
    }

    /// Writes data to the buffer.
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn write_buffer_slice(&mut self, slice: impl AsBufferSlice, data: &[impl bytemuck::Pod]) {
        self.write_buffer_raw(slice, bytemuck::cast_slice(data))
    }
}

pub struct ComputeCommandEncoder<'a> {
    device: &'a mut metal::DeviceRef,
    encoder: metal::ComputeCommandEncoder,
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
    pub fn metal(&self) -> &metal::ComputeCommandEncoderRef {
        &self.encoder
    }
}

impl Drop for ComputeCommandEncoder<'_> {
    #[inline(always)]
    fn drop(&mut self) {
        self.encoder.end_encoding();
    }
}

#[hidden_trait::expose]
impl traits::ComputeCommandEncoder for ComputeCommandEncoder<'_> {
    #[inline(always)]
    fn barrier(&mut self, _after: PipelineStages, _before: PipelineStages) {}

    #[inline(always)]
    fn init_image(&mut self, _after: PipelineStages, _before: PipelineStages, _image: &Image) {}

    #[inline(always)]
    fn with_pipeline(&mut self, pipeline: &crate::backend::ComputePipeline) {
        self.encoder.set_compute_pipeline_state(pipeline.metal());
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

        self.encoder
            .set_bytes(0, data_bytes.len() as NSUInteger, data_bytes.as_ptr() as _);
    }

    #[inline(always)]
    fn dispatch(&mut self, groups: Extent3) {
        let group_size = self.workgroup_size.unwrap_or([1, 1, 1]);

        self.encoder.dispatch_thread_groups(
            metal::MTLSize {
                width: groups.width().into(),
                height: groups.height().into(),
                depth: groups.depth().into(),
            },
            metal::MTLSize {
                width: group_size[0].into(),
                height: group_size[1].into(),
                depth: group_size[2].into(),
            },
        );
    }
}

pub struct RenderCommandEncoder<'a> {
    encoder: metal::RenderCommandEncoder,
    primitive: metal::MTLPrimitiveType,
    index_buffer: Option<metal::Buffer>,
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
    pub fn metal(&self) -> &metal::RenderCommandEncoderRef {
        &self.encoder
    }
}

impl Drop for RenderCommandEncoder<'_> {
    #[inline(always)]
    fn drop(&mut self) {
        self.encoder.end_encoding();
    }
}

#[hidden_trait::expose]
impl crate::traits::RenderCommandEncoder for RenderCommandEncoder<'_> {
    #[inline(always)]
    fn with_pipeline(&mut self, pipeline: &RenderPipeline) {
        self.encoder.set_render_pipeline_state(pipeline.metal());
        self.primitive = pipeline.primitive();
        self.vertex_bindings = pipeline.vertex_bindings();
        self.fragment_bindings = pipeline.fragment_bindings();
        self.vertex_buffers_count = pipeline.vertex_buffers_count();
    }

    #[inline(always)]
    fn with_viewport(&mut self, offset: Offset3<f32>, extent: Extent3<f32>) {
        let viewport = metal::MTLViewport {
            originX: offset.x().into(),
            originY: offset.y().into(),
            width: extent.width().into(),
            height: extent.height().into(),
            znear: offset.z().into(),
            zfar: (offset.z() + extent.depth()).into(),
        };
        self.encoder.set_viewport(viewport);
    }

    #[inline(always)]
    fn with_scissor(&mut self, offset: Offset2<i32>, extent: Extent2<u32>) {
        debug_assert!(offset.x() >= 0);
        debug_assert!(offset.y() >= 0);

        let scissor = metal::MTLScissorRect {
            x: offset.x() as NSUInteger,
            y: offset.y() as NSUInteger,
            width: extent.width() as NSUInteger,
            height: extent.height() as NSUInteger,
        };
        self.encoder.set_scissor_rect(scissor);
    }

    /// Sets arguments group for the current pipeline.
    #[inline(always)]
    fn with_arguments(&mut self, group: u32, arguments: &impl Arguments) {
        arguments.bind_render(group, self);
    }

    /// Sets constants for the current pipeline.
    #[cfg_attr(feature = "inline-more", inline)]
    fn with_constants(&mut self, constants: &impl DeviceRepr) {
        let data = constants.as_repr();
        let data_bytes = bytemuck::bytes_of(&data);

        if let Some(vb) = &self.vertex_bindings {
            if let Some(slot) = vb.push_constants {
                self.encoder.set_vertex_bytes(
                    slot as u64,
                    data_bytes.len() as u64,
                    data_bytes.as_ptr() as _,
                );
            }
        }

        if let Some(fb) = &self.fragment_bindings {
            if let Some(slot) = fb.push_constants {
                self.encoder.set_fragment_bytes(
                    slot as u64,
                    data_bytes.len() as u64,
                    data_bytes.as_ptr() as _,
                );
            }
        }
    }

    /// Bind vertex buffer to the current pipeline.
    #[cfg_attr(feature = "inline-more", inline)]
    fn bind_vertex_buffers(&mut self, start: u32, buffers: &[(impl AsBufferSlice)]) {
        let (buffers, offsets) = buffers
            .iter()
            .map(|slice| {
                let slice = slice.as_buffer_slice();
                let offset = slice.offset as NSUInteger;
                let buffer = slice.buffer.metal();
                (Some(buffer), offset)
            })
            .unzip::<_, _, SmallVec<[_; 8]>, SmallVec<[_; 8]>>();

        let first = self.vertex_buffers_count + start;

        self.encoder
            .set_vertex_buffers(first as NSUInteger, &buffers, &offsets);
    }

    /// Bind index buffer to the current pipeline.
    #[inline(always)]
    fn bind_index_buffer(&mut self, buffer: impl AsBufferSlice) {
        let buffer_slice = buffer.as_buffer_slice();

        self.index_buffer = Some(buffer_slice.buffer.metal().to_owned());
        self.index_buffer_offset = buffer_slice.offset as NSUInteger;
    }

    #[cfg_attr(feature = "inline-more", inline)]
    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        if vertices.end <= vertices.start {
            // Rendering no vertices is a no-op
            return;
        }
        if instances.end <= instances.start {
            // Rendering no instances is a no-op
            return;
        }

        if instances.end - 1 == instances.start {
            // Rendering single instance
            self.encoder.draw_primitives(
                self.primitive,
                vertices.start.into(),
                (vertices.end - vertices.start).into(),
            );
        } else if instances.start == 0 {
            // Rendering multiple instances
            self.encoder.draw_primitives_instanced(
                self.primitive,
                vertices.start.into(),
                (vertices.end - vertices.start).into(),
                instances.end.into(),
            );
        } else {
            // Rendering multiple instances with a base instance
            self.encoder.draw_primitives_instanced_base_instance(
                self.primitive,
                vertices.start.into(),
                (vertices.end - vertices.start).into(),
                (instances.end - instances.start).into(),
                instances.start.into(),
            );
        }
    }

    #[cfg_attr(feature = "inline-more", inline)]
    fn draw_indexed(&mut self, vertex_offset: i32, indices: Range<u32>, instances: Range<u32>) {
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

        if instances.end - 1 == instances.start && vertex_offset == 0 {
            // Rendering single instance
            self.encoder.draw_indexed_primitives(
                self.primitive,
                (indices.end - indices.start).into(),
                metal::MTLIndexType::UInt32,
                index_buffer,
                (self.index_buffer_offset + (indices.start as NSUInteger * 4)).into(),
            );
        } else if instances.start == 0 && vertex_offset == 0 {
            // Rendering multiple instances
            self.encoder.draw_indexed_primitives_instanced(
                self.primitive,
                (indices.end - indices.start).into(),
                metal::MTLIndexType::UInt32,
                index_buffer,
                (self.index_buffer_offset + (indices.start as NSUInteger * 4)).into(),
                instances.end.into(),
            );
        } else {
            // Rendering multiple instances with a base instance
            self.encoder
                .draw_indexed_primitives_instanced_base_instance(
                    self.primitive,
                    (indices.end - indices.start).into(),
                    metal::MTLIndexType::UInt32,
                    self.index_buffer.as_ref().unwrap(),
                    (self.index_buffer_offset + (indices.start as NSUInteger * 4)).into(),
                    (instances.end - instances.start).into(),
                    instances.start.into(),
                    vertex_offset as NSUInteger,
                );
        }
    }
}

pub struct AccelerationStructureCommandEncoder<'a> {
    device: &'a mut metal::DeviceRef,
    encoder: metal::AccelerationStructureCommandEncoder,
    _marker: PhantomData<&'a mut CommandBuffer>,
}

#[hidden_trait::expose]
impl crate::traits::AccelerationStructureCommandEncoder
    for AccelerationStructureCommandEncoder<'_>
{
    fn build_blas(&mut self, blas: &Blas, desc: BlasBuildDesc, scratch: impl AsBufferSlice) {
        use objc::{
            class, msg_send,
            runtime::{Object, BOOL, YES},
            sel, sel_impl,
        };

        let mut mdesc = metal::PrimitiveAccelerationStructureDescriptor::descriptor();
        match desc.performance {
            AccelerationStructurePerformance::FastBuild => unsafe {
                msg_send![mdesc, usage: 0x2u64]
            },
            _ => {}
        }

        let mut geometry_descs = Vec::<metal::AccelerationStructureGeometryDescriptor>::new();

        for geometry in desc.geometry {
            match geometry {
                BlasGeometryDesc::Triangles(triangles) => {
                    let mut mdesc =
                        metal::AccelerationStructureTriangleGeometryDescriptor::descriptor();

                    mdesc.set_opaque(triangles.opaque);

                    let mut count =
                        (triangles.vertices.size / triangles.vertex_stride) as NSUInteger;

                    if let Some(indices) = triangles.indices {
                        mdesc.set_index_type(metal::MTLIndexType::UInt32);
                        mdesc.set_index_buffer_offset(indices.offset as _);
                        mdesc.set_index_buffer(Some(indices.buffer.metal()));
                        count = indices.size as NSUInteger / 4;
                    }

                    mdesc.set_triangle_count(count as NSUInteger);

                    if let Some(transform) = triangles.transform {
                        mdesc.set_transformation_matrix_buffer_offset(
                            transform.offset as NSUInteger,
                        );
                        mdesc.set_transformation_matrix_buffer(Some(transform.buffer.metal()));
                    }

                    mdesc.set_vertex_format(triangles.vertex_format.try_into_metal().unwrap());
                    mdesc.set_vertex_stride(triangles.vertex_stride as NSUInteger);
                    mdesc.set_vertex_buffer_offset(triangles.vertices.offset as NSUInteger);
                    mdesc.set_vertex_buffer(Some(triangles.vertices.buffer.metal()));

                    geometry_descs.push((**mdesc).to_owned());
                }
                BlasGeometryDesc::AABBs(aabbs) => {
                    let mut mdesc =
                        metal::AccelerationStructureBoundingBoxGeometryDescriptor::descriptor();

                    mdesc.set_opaque(aabbs.opaque);

                    mdesc.set_bounding_box_count(
                        (aabbs.boxes.size / aabbs.box_stride) as NSUInteger,
                    );
                    unsafe {
                        msg_send![mdesc, setBoundingBoxStride: (aabbs.box_stride as NSUInteger)]
                    }
                    unsafe {
                        msg_send![mdesc, setBoundingBoxBufferOffset: (aabbs.boxes.offset as NSUInteger)]
                    }
                    mdesc.set_bounding_box_buffer(Some(aabbs.boxes.buffer.metal()));

                    geometry_descs.push((**mdesc).to_owned());
                }
            }
        }

        let geometry_descs = metal::Array::from_owned_slice(&*geometry_descs);

        mdesc.set_geometry_descriptors(geometry_descs);
    }

    fn build_tlas(&mut self, tlas: &Tlas, desc: TlasBuildDesc, scratch: impl AsBufferSlice) {}
}
