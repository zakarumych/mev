use std::{marker::PhantomData, ops::Range, rc::Rc};

use smallvec::SmallVec;
use wgpu::util::DeviceExt;

use crate::{
    backend::from::IntoWgpu,
    generic::{
        Arguments, AsBufferSlice, BlasBuildDesc, ClearColor, ClearDepthStencil, DeviceRepr,
        Extent2, Extent3, LoadOp, Offset2, Offset3, PipelineStages, RenderPassDesc, StoreOp,
        TlasBuildDesc,
    },
};

use super::{
    acst::{Blas, Tlas},
    buffer::Buffer,
    compute::ComputePipeline,
    image::{Image, ImageInner},
    render::RenderPipeline,
    surface::Frame,
    Device,
};

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

#[hidden_trait::expose]
impl crate::traits::SyncCommandEncoder for CommandEncoder {
    fn barrier(&mut self, _after: PipelineStages, _before: PipelineStages) {}

    fn init_image(&mut self, _after: PipelineStages, _before: PipelineStages, _image: &Image) {}
}

#[hidden_trait::expose]
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
                    view: att.image.wgpu_view(),
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
                view: att.image.wgpu_view(),
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

        RenderCommandEncoder {
            pass: encoder,
            current_pipeline: None,
            device: self.device.clone(),
        }
    }

    fn acceleration_structure(&mut self) -> AccelerationStructureCommandEncoder<'_> {
        AccelerationStructureCommandEncoder {
            _marker: PhantomData,
        }
    }

    fn present(&mut self, frame: Frame, _after: PipelineStages) {
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

#[hidden_trait::expose]
impl crate::traits::SyncCommandEncoder for CopyCommandEncoder<'_> {
    fn barrier(&mut self, _after: PipelineStages, _before: PipelineStages) {}

    fn init_image(&mut self, _after: PipelineStages, _before: PipelineStages, _image: &Image) {}
}

#[hidden_trait::expose]
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

    fn write_buffer(&mut self, slice: impl AsBufferSlice, data: &impl bytemuck::Pod) {
        self.write_buffer_raw(slice, bytemuck::bytes_of(data));
    }

    fn write_buffer_slice(&mut self, slice: impl AsBufferSlice, data: &[impl bytemuck::Pod]) {
        self.write_buffer_raw(slice, bytemuck::cast_slice(data));
    }

    fn copy_buffer_to_buffer(
        &mut self,
        src: &Buffer,
        src_offset: usize,
        dst: &Buffer,
        dst_offset: usize,
        size: usize,
    ) {
        self.encoder.copy_buffer_to_buffer(
            src.wgpu(),
            src_offset as u64,
            dst.wgpu(),
            dst_offset as u64,
            size as u64,
        );
    }

    fn copy_buffer_to_image(
        &mut self,
        src: &Buffer,
        src_offset: usize,
        bytes_per_line: usize,
        bytes_per_plane: usize,
        dst: &Image,
        dst_offset: Offset3<u32>,
        extent: Extent3<u32>,
        layers: Range<u32>,
        level: u32,
    ) {
        self.encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: src.wgpu(),
                layout: wgpu::TexelCopyBufferLayout {
                    offset: src_offset as u64,
                    bytes_per_row: Some(bytes_per_line as u32),
                    rows_per_image: Some((bytes_per_plane / bytes_per_line) as u32),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: dst.wgpu_view().texture(),
                mip_level: level + dst.base_level(),
                origin: wgpu::Origin3d {
                    x: dst_offset.x(),
                    y: dst_offset.y(),
                    z: layers.start + dst.base_layer(),
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: extent.width(),
                height: extent.height(),
                depth_or_array_layers: extent.depth() * (layers.end - layers.start),
            },
        );
    }

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
        self.encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: src.wgpu_view().texture(),
                mip_level: src_level + src.base_level(),
                origin: wgpu::Origin3d {
                    x: src_offset.x(),
                    y: src_offset.y(),
                    z: src_base_layer + src.base_layer(),
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: dst.wgpu_view().texture(),
                mip_level: dst_level + dst.base_level(),
                origin: wgpu::Origin3d {
                    x: dst_offset.x(),
                    y: dst_offset.y(),
                    z: dst_base_layer + dst.base_layer(),
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: extent.width(),
                height: extent.height(),
                depth_or_array_layers: layers,
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

#[hidden_trait::expose]
impl crate::traits::SyncCommandEncoder for ComputeCommandEncoder<'_> {
    fn barrier(&mut self, _after: PipelineStages, _before: PipelineStages) {}

    fn init_image(&mut self, _after: PipelineStages, _before: PipelineStages, _image: &Image) {}
}

#[hidden_trait::expose]
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

#[hidden_trait::expose]
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

    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.pass.draw(vertices, instances);
    }

    fn draw_indexed(&mut self, vertex_offset: i32, indices: Range<u32>, instances: Range<u32>) {
        self.pass.draw_indexed(indices, vertex_offset, instances);
    }
}

pub struct AccelerationStructureCommandEncoder<'enc> {
    _marker: PhantomData<&'enc mut ()>,
}

#[hidden_trait::expose]
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
