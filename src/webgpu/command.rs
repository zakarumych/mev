use std::{marker::PhantomData, range::Range, rc::Rc};

use mev_proc::hidden_trait;
use smallvec::SmallVec;
use wgpu::util::DeviceExt;

use crate::{
    AsImageSlice,
    backend::from::IntoWgpu,
    generic::{
        Arguments, AsBufferSlice, BlasBuildDesc, ClearColor, ClearDepthStencil, DeviceRepr, Extent,
        Extent2, Extent3, LoadOp, Offset2, Offset3, PipelineStages, PixelFormat, RenderPassDesc,
        StoreOp, TexelShape, TlasBuildDesc,
    },
};

use super::{
    Device,
    acst::{Blas, Tlas},
    buffer::Buffer,
    compute::ComputePipeline,
    image::{Image, ImageInner},
    render::RenderPipeline,
    surface::Frame,
};

/// Pads `shape`'s extent/resolved stride out to the 3 axes (element/row/plane) every native buffer<->image copy command needs,
/// regardless of how many axes `shape` itself specifies (`N` is `1..=3` at these call sites).
///
/// Any axis at or beyond `N` defaults to a tightly-packed value
/// (extent `1`, stride derived the same way `TexelShape::byte_stride` derives any unset axis).
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
    pub(super) buffer: wgpu::CommandBuffer,
    pub(super) surface_textures: Vec<wgpu::SurfaceTexture>,
}

pub struct CommandEncoder {
    encoder: wgpu::CommandEncoder,
    device: Device,
    queue: wgpu::Queue,
    pending_presents: Vec<wgpu::SurfaceTexture>,
}

impl CommandEncoder {
    pub(super) fn new(encoder: wgpu::CommandEncoder, device: Device, queue: wgpu::Queue) -> Self {
        CommandEncoder {
            encoder,
            device,
            queue,
            pending_presents: Vec::new(),
        }
    }
}

#[hidden_trait]
impl crate::traits::SyncCommandEncoder for CommandEncoder {
    fn barrier(&mut self, _prev_stages: PipelineStages, _next_stages: PipelineStages) {}

    fn init_image(
        &mut self,
        _prev_use: PipelineStages,
        _next_use: PipelineStages,
        _image: impl AsRef<Image>,
    ) {
    }
}

#[hidden_trait]
impl crate::traits::CommandEncoder for CommandEncoder {
    fn copy(&mut self) -> CopyCommandEncoder<'_> {
        CopyCommandEncoder {
            encoder: &mut self.encoder,
            device: self.device.clone(),
            queue: self.queue.clone(),
            _marker: PhantomData,
        }
    }

    fn compute(&mut self) -> ComputeCommandEncoder<'_> {
        let pass = self
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
        ComputeCommandEncoder {
            pass,
            current_pipeline: None,
            device: self.device.clone(),
        }
    }

    fn render(&mut self, desc: RenderPassDesc) -> RenderCommandEncoder<'_> {
        let color_attachments = desc
            .color_attachments
            .iter()
            .map(|att| {
                Some(wgpu::RenderPassColorAttachment {
                    view: att.image.image().wgpu_view(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: match att.load {
                            LoadOp::Load => wgpu::LoadOp::Load,
                            LoadOp::Clear(color) => wgpu::LoadOp::Clear(color.into_wgpu()),
                            LoadOp::DontCare => wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        },
                        store: match att.store {
                            StoreOp::Store => wgpu::StoreOp::Store,
                            StoreOp::DontCare => wgpu::StoreOp::Discard,
                        },
                    },
                })
            })
            .collect::<SmallVec<[_; 4]>>();

        let depth_stencil_attachment = desc.depth_stencil_attachment.as_ref().map(|att| {
            wgpu::RenderPassDepthStencilAttachment {
                view: att.image.image().wgpu_view(),
                depth_ops: Some(wgpu::Operations {
                    load: match att.load {
                        LoadOp::Load => wgpu::LoadOp::Load,
                        LoadOp::Clear(clear) => wgpu::LoadOp::Clear(clear.depth),
                        LoadOp::DontCare => wgpu::LoadOp::Clear(0.0),
                    },
                    store: match att.store {
                        StoreOp::Store => wgpu::StoreOp::Store,
                        StoreOp::DontCare => wgpu::StoreOp::Discard,
                    },
                }),
                stencil_ops: Some(wgpu::Operations {
                    load: match att.load {
                        LoadOp::Load => wgpu::LoadOp::Load,
                        LoadOp::Clear(clear) => wgpu::LoadOp::Clear(clear.stencil),
                        LoadOp::DontCare => wgpu::LoadOp::Clear(0),
                    },
                    store: match att.store {
                        StoreOp::Store => wgpu::StoreOp::Store,
                        StoreOp::DontCare => wgpu::StoreOp::Discard,
                    },
                }),
            }
        });

        let encoder = self.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(desc.name),
            color_attachments: &color_attachments,
            depth_stencil_attachment: depth_stencil_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        let mut render = RenderCommandEncoder {
            pass: encoder,
            current_pipeline: None,
            device: self.device.clone(),
        };

        // Default the viewport and scissor to the full extent of the first
        // attachment (first color attachment, else depth-stencil), matching
        // an explicit `with_viewport`/`with_scissor` call to the same effect.
        // wgpu already defaults to the full target, but this keeps all
        // backends' behavior identical.
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

    fn acceleration_structure(&mut self) -> AccelerationStructureCommandEncoder<'_> {
        AccelerationStructureCommandEncoder {
            _marker: PhantomData,
        }
    }

    fn present(&mut self, frame: Frame, _prev_use: PipelineStages) {
        self.pending_presents.push(frame.texture);
    }

    fn finish(self) -> CommandBuffer {
        CommandBuffer {
            buffer: self.encoder.finish(),
            surface_textures: self.pending_presents,
        }
    }
}

// CopyCommandEncoder is declared LAST (dropped first) so encoder is valid during its lifetime
pub struct CopyCommandEncoder<'enc> {
    encoder: &'enc mut wgpu::CommandEncoder,
    device: Device,
    queue: wgpu::Queue,
    _marker: PhantomData<&'enc mut CommandEncoder>,
}

#[hidden_trait]
impl crate::traits::SyncCommandEncoder for CopyCommandEncoder<'_> {
    fn barrier(&mut self, _prev_stages: PipelineStages, _next_stages: PipelineStages) {}

    fn init_image(
        &mut self,
        _prev_use: PipelineStages,
        _next_use: PipelineStages,
        _image: impl AsRef<Image>,
    ) {
    }
}

#[hidden_trait]
impl crate::traits::CopyCommandEncoder for CopyCommandEncoder<'_> {
    fn fill_buffer(&mut self, slice: impl AsBufferSlice, byte: u8) {
        let slice = slice.as_buffer_slice();

        if slice.size() == 0 {
            return;
        }

        if byte == 0 {
            self.encoder.clear_buffer(
                slice.buffer().wgpu(),
                slice.offset() as u64,
                Some(slice.size() as u64),
            );
        } else {
            // Create a staging buffer filled with the byte
            let data: Vec<u8> = vec![byte; slice.size()];

            let (buffer, offset) = self.device.allocate_scratch(data.len() as u64);

            self.queue.write_buffer(&buffer, offset, &data[..]);

            self.encoder.copy_buffer_to_buffer(
                &buffer,
                offset,
                slice.buffer().wgpu(),
                slice.offset() as u64,
                slice.size() as u64,
            );
        }
    }

    fn write_buffer_raw(&mut self, slice: impl AsBufferSlice, data: &[u8]) {
        if data.is_empty() {
            return;
        }

        let slice = slice.as_buffer_slice();

        assert_eq!(
            data.len(),
            slice.size(),
            "data length does not match buffer slice size"
        );

        let (buffer, offset) = self.device.allocate_scratch(data.len() as u64);

        self.queue.write_buffer(&buffer, offset, data);

        self.encoder.copy_buffer_to_buffer(
            &buffer,
            offset,
            slice.buffer().wgpu(),
            slice.offset() as u64,
            slice.size() as u64,
        );
    }

    fn write_buffer(&mut self, slice: impl AsBufferSlice, data: &impl bytemuck::NoUninit) {
        self.write_buffer_raw(slice, bytemuck::bytes_of(data));
    }

    fn write_buffer_slice(&mut self, slice: impl AsBufferSlice, data: &[impl bytemuck::NoUninit]) {
        self.write_buffer_raw(slice, bytemuck::cast_slice(data));
    }

    /// Writes data and emits the surrounding barriers. `prev_use` = stages
    /// that previously used the buffer, `next_use` = stages that will use it
    /// next.
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

    fn copy_buffer_to_buffer(&mut self, src: &impl AsBufferSlice, dst: &impl AsBufferSlice) {
        let src = src.as_buffer_slice();
        let dst = dst.as_buffer_slice();

        assert_eq!(
            src.size(),
            dst.size(),
            "copy_buffer_to_buffer requires src and dst to have the same size (mirrors slice::copy_from_slice)"
        );

        self.encoder.copy_buffer_to_buffer(
            src.buffer().wgpu(),
            src.offset() as u64,
            dst.buffer().wgpu(),
            dst.offset() as u64,
            src.size() as u64,
        );
    }

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
            shape.extent,
            dst.extent(),
            "copy_buffer_to_image: TexelShape extent must match the destination image slice extent"
        );

        let block_size = dst.image().format().block_size();
        let (extent, resolved) = pad_shape_to_3(&shape);
        assert_eq!(
            resolved[0], block_size,
            "copy_buffer_to_image: element stride must equal the format's tightly-packed size ({block_size}) - native copy commands can't vary it; use Cvt for padded/strided element transfers"
        );
        let bytes_per_line = resolved[1];
        let bytes_per_plane = resolved[2];

        let dst_offset = dst.offset().to_3d().map(|v| v as u32);
        let dst_layers = dst.layers();

        self.encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: src.buffer().wgpu(),
                layout: wgpu::TexelCopyBufferLayout {
                    offset: src.offset() as u64,
                    bytes_per_row: Some(bytes_per_line as u32),
                    rows_per_image: Some((bytes_per_plane / bytes_per_line) as u32),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: dst.image().wgpu_view().texture(),
                mip_level: dst.level() + dst.image().base_level(),
                origin: wgpu::Origin3d {
                    x: dst_offset.x(),
                    y: dst_offset.y(),
                    z: dst_layers.start + dst.image().base_layer(),
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: extent.width(),
                height: extent.height(),
                depth_or_array_layers: extent.depth() * (dst_layers.end - dst_layers.start),
            },
        );
    }

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
            src.extent(),
            dst.extent(),
            "copy_image_region: source and destination image slices must have the same extent"
        );

        let src_offset = src.offset().to_3d().map(|v| v as u32);
        let dst_offset = dst.offset().to_3d().map(|v| v as u32);
        let extent = src.extent().to_3d();
        let src_layers = src.layers();
        let dst_layers = dst.layers();

        self.encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: src.image().wgpu_view().texture(),
                mip_level: src.level() + src.image().base_level(),
                origin: wgpu::Origin3d {
                    x: src_offset.x(),
                    y: src_offset.y(),
                    z: src_layers.start + src.image().base_layer(),
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: dst.image().wgpu_view().texture(),
                mip_level: dst.level() + dst.image().base_level(),
                origin: wgpu::Origin3d {
                    x: dst_offset.x(),
                    y: dst_offset.y(),
                    z: dst_layers.start + dst.image().base_layer(),
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: extent.width(),
                height: extent.height(),
                depth_or_array_layers: extent.depth() * (src_layers.end - src_layers.start),
            },
        );
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
            shape.extent,
            src.extent(),
            "copy_image_to_buffer: TexelShape extent must match the source image slice extent"
        );

        let block_size = src.image().format().block_size();
        let (extent, resolved) = pad_shape_to_3(&shape);
        assert_eq!(
            resolved[0], block_size,
            "copy_image_to_buffer: element stride must equal the format's tightly-packed size ({block_size}) - native copy commands can't vary it; use Cvt for padded/strided element transfers"
        );
        let bytes_per_line = resolved[1];
        let bytes_per_plane = resolved[2];

        let src_offset = src.offset().to_3d().map(|v| v as u32);
        let src_layers = src.layers();

        self.encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: src.image().wgpu_view().texture(),
                mip_level: src.level() + src.image().base_level(),
                origin: wgpu::Origin3d {
                    x: src_offset.x(),
                    y: src_offset.y(),
                    z: src_layers.start + src.image().base_layer(),
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: dst.buffer().wgpu(),
                layout: wgpu::TexelCopyBufferLayout {
                    offset: dst.offset() as u64,
                    bytes_per_row: Some(bytes_per_line as u32),
                    rows_per_image: Some((bytes_per_plane / bytes_per_line) as u32),
                },
            },
            wgpu::Extent3d {
                width: extent.width(),
                height: extent.height(),
                depth_or_array_layers: extent.depth() * (src_layers.end - src_layers.start),
            },
        );
    }
}

pub struct ComputeCommandEncoder<'enc> {
    pass: wgpu::ComputePass<'enc>,
    current_pipeline: Option<ComputePipeline>,
    device: Device,
}

impl ComputeCommandEncoder<'_> {
    #[doc(hidden)]
    pub fn wgpu_device(&self) -> &wgpu::Device {
        self.device.wgpu()
    }

    #[doc(hidden)]
    pub fn set_bind_group(&mut self, group: u32, bind_group: &wgpu::BindGroup) {
        self.pass.set_bind_group(group, bind_group, &[]);
    }

    #[doc(hidden)]
    pub fn bind_group_layout(&self, group: u32) -> &wgpu::BindGroupLayout {
        self.current_pipeline
            .as_ref()
            .expect("No pipeline set")
            .bind_group_layout(group as usize)
    }
}

#[hidden_trait]
impl crate::traits::SyncCommandEncoder for ComputeCommandEncoder<'_> {
    fn barrier(&mut self, _prev_stages: PipelineStages, _next_stages: PipelineStages) {}

    fn init_image(
        &mut self,
        _prev_use: PipelineStages,
        _next_use: PipelineStages,
        _image: impl AsRef<Image>,
    ) {
    }
}

#[hidden_trait]
impl crate::traits::ComputeCommandEncoder for ComputeCommandEncoder<'_> {
    fn with_pipeline(&mut self, pipeline: &ComputePipeline) {
        self.pass.set_pipeline(pipeline.wgpu());
        self.current_pipeline = Some(pipeline.clone());
    }

    fn with_arguments(&mut self, group: u32, arguments: &impl Arguments) {
        arguments.bind_compute(group, self);
    }

    fn with_constants(&mut self, constants: &impl DeviceRepr) {
        let data = constants.as_repr();
        let bytes = bytemuck::bytes_of(&data);
        if !bytes.is_empty() {
            self.pass.set_immediates(0, bytes);
        }
    }

    fn dispatch(&mut self, groups: Extent3) {
        self.pass
            .dispatch_workgroups(groups.width(), groups.height(), groups.depth());
    }

    fn dispatch_indirect(&mut self, args: impl AsBufferSlice) {
        let s = args.as_buffer_slice();
        self.pass
            .dispatch_workgroups_indirect(s.buffer().wgpu(), s.offset() as u64);
    }
}

// Fields declared in this order so `pass` drops before the view arcs (Rust drops in reverse order)
pub struct RenderCommandEncoder<'enc> {
    pass: wgpu::RenderPass<'enc>,
    current_pipeline: Option<RenderPipeline>,
    device: Device,
}

impl RenderCommandEncoder<'_> {
    #[doc(hidden)]
    pub fn wgpu_device(&self) -> &wgpu::Device {
        self.device.wgpu()
    }

    #[doc(hidden)]
    pub fn set_bind_group(&mut self, group: u32, bind_group: &wgpu::BindGroup) {
        self.pass.set_bind_group(group, bind_group, &[]);
    }

    #[doc(hidden)]
    pub fn bind_group_layout(&self, group: u32) -> &wgpu::BindGroupLayout {
        self.current_pipeline
            .as_ref()
            .expect("No pipeline set")
            .bind_group_layout(group as usize)
    }
}

#[hidden_trait]
impl crate::traits::RenderCommandEncoder for RenderCommandEncoder<'_> {
    fn with_pipeline(&mut self, pipeline: &RenderPipeline) {
        self.pass.set_pipeline(pipeline.wgpu());
        self.current_pipeline = Some(pipeline.clone());
    }

    fn with_viewport(&mut self, offset: Offset3<f32>, extent: Extent3<f32>) {
        self.pass.set_viewport(
            offset.x(),
            offset.y(),
            extent.width(),
            extent.height(),
            offset.z(),
            offset.z() + extent.depth(),
        );
    }

    fn with_scissor(&mut self, offset: Offset2<i32>, extent: Extent2<u32>) {
        debug_assert!(offset.x() >= 0);
        debug_assert!(offset.y() >= 0);
        self.pass.set_scissor_rect(
            offset.x() as u32,
            offset.y() as u32,
            extent.width(),
            extent.height(),
        );
    }

    fn with_arguments(&mut self, group: u32, arguments: &impl Arguments) {
        arguments.bind_render(group, self);
    }

    fn with_constants(&mut self, constants: &impl DeviceRepr) {
        let data = constants.as_repr();
        let bytes = bytemuck::bytes_of(&data);
        if !bytes.is_empty() {
            self.pass.set_immediates(0, bytes);
        }
    }

    fn bind_vertex_buffers(&mut self, start: u32, slices: &[impl AsBufferSlice]) {
        for (i, slice) in slices.iter().enumerate() {
            let s = slice.as_buffer_slice();
            self.pass.set_vertex_buffer(
                start + i as u32,
                s.buffer().wgpu().slice(s.offset() as u64..),
            );
        }
    }

    fn bind_index_buffer(&mut self, slice: impl AsBufferSlice) {
        let s = slice.as_buffer_slice();
        self.pass.set_index_buffer(
            s.buffer().wgpu().slice(s.offset() as u64..),
            wgpu::IndexFormat::Uint32,
        );
    }

    fn draw(&mut self, vertices: impl Into<Range<u32>>, instances: impl Into<Range<u32>>) {
        let vertices = vertices.into();
        let instances = instances.into();
        self.pass
            .draw(vertices.start..vertices.end, instances.start..instances.end);
    }

    fn draw_indexed(
        &mut self,
        vertex_offset: i32,
        indices: impl Into<Range<u32>>,
        instances: impl Into<Range<u32>>,
    ) {
        let indices = indices.into();
        let instances = instances.into();
        self.pass.draw_indexed(
            indices.start..indices.end,
            vertex_offset,
            instances.start..instances.end,
        );
    }

    fn draw_indirect(&mut self, args: impl AsBufferSlice) {
        let s = args.as_buffer_slice();
        self.pass.draw_indirect(s.buffer().wgpu(), s.offset() as u64);
    }

    fn draw_indexed_indirect(&mut self, args: impl AsBufferSlice) {
        let s = args.as_buffer_slice();
        self.pass
            .draw_indexed_indirect(s.buffer().wgpu(), s.offset() as u64);
    }
}

pub struct AccelerationStructureCommandEncoder<'enc> {
    _marker: PhantomData<&'enc mut ()>,
}

#[hidden_trait]
impl crate::traits::AccelerationStructureCommandEncoder
    for AccelerationStructureCommandEncoder<'_>
{
    fn build_blas(&mut self, _blas: &Blas, _desc: BlasBuildDesc, _scratch: impl AsBufferSlice) {
        panic!("acceleration structures are not supported in the WebGPU backend")
    }

    fn build_tlas(&mut self, _tlas: &Tlas, _desc: TlasBuildDesc, _scratch: impl AsBufferSlice) {
        panic!("acceleration structures are not supported in the WebGPU backend")
    }
}
