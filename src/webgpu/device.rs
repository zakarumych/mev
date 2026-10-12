use std::{fmt, num::NonZero, rc::Rc};

use mev_proc::hidden_trait;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use smallvec::SmallVec;
use wgpu::util::DeviceExt;

use crate::{
    ArgumentGroupLayout, PixelFormat,
    backend::from::{IntoWgpu, TryIntoWgpu, TryWgpuFrom, WgpuFrom},
    generic::{
        BlasDesc, BufferDesc, BufferInit, BufferInitDesc, ComputePipelineDesc, ImageDesc,
        ImageDims, ImageExtent, ImageUsage, LibraryDesc, LibraryInput, PipelineError,
        RenderPipelineDesc, SamplerDesc, ShaderLanguage, ShaderLibraryError, SurfaceError,
        TlasDesc,
    },
};

use super::{
    acst::{Blas, Tlas},
    buffer::Buffer,
    compute::ComputePipeline,
    image::Image,
    render::RenderPipeline,
    sampler::Sampler,
    shader::Library,
    surface::Surface,
};

struct DeviceInner {
    device: wgpu::Device,
    instance: Rc<wgpu::Instance>,
    adapter: wgpu::Adapter,
}

#[derive(Clone)]
pub struct Device {
    inner: Rc<DeviceInner>,
}

impl Device {
    pub(crate) fn compute_kernel_limits(&self) -> crate::generic::ComputeKernelLimits {
        let limits = self.inner.device.limits();
        crate::generic::ComputeKernelLimits {
            max_workgroup_size: limits
                .max_compute_workgroup_size_x
                .min(limits.max_compute_invocations_per_workgroup),
            max_workgroup_memory: limits.max_compute_workgroup_storage_size as usize,
            max_storage_buffers: limits.max_storage_buffers_per_shader_stage,
            max_uniform_buffers: limits.max_uniform_buffers_per_shader_stage,
            max_buffer_bindings: limits.max_bindings_per_bind_group.min(
                limits.max_storage_buffers_per_shader_stage
                    + limits.max_uniform_buffers_per_shader_stage,
            ),
            max_storage_buffer_range: limits.max_storage_buffer_binding_size as usize,
            max_uniform_buffer_range: limits.max_uniform_buffer_binding_size as usize,
            storage_offset_alignment: limits.min_storage_buffer_offset_alignment as usize,
            uniform_offset_alignment: limits.min_uniform_buffer_offset_alignment as usize,
            max_dispatch_count: limits.max_compute_workgroups_per_dimension,
            max_subgroups_per_workgroup: None,
            subgroup_arithmetic: self
                .inner
                .device
                .features()
                .contains(wgpu::Features::SUBGROUP),
            shader_f16: self
                .inner
                .device
                .features()
                .contains(wgpu::Features::SHADER_F16),
        }
    }

    pub(super) fn new(
        device: wgpu::Device,
        instance: Rc<wgpu::Instance>,
        adapter: wgpu::Adapter,
    ) -> Self {
        Device {
            inner: Rc::new(DeviceInner {
                device,
                instance,
                adapter,
            }),
        }
    }

    pub(super) fn wgpu(&self) -> &wgpu::Device {
        &self.inner.device
    }

    pub(super) fn wgpu_instance(&self) -> &wgpu::Instance {
        &self.inner.instance
    }

    pub(super) fn wgpu_adapter(&self) -> &wgpu::Adapter {
        &self.inner.adapter
    }

    pub(crate) fn new_compute_pipeline_with_buffer_access(
        &self,
        desc: ComputePipelineDesc,
        storage_read_only: &[bool],
    ) -> Result<ComputePipeline, PipelineError> {
        let bind_group_layouts = create_bind_group_layouts_with_buffer_access(
            &self.inner.device,
            desc.arguments,
            storage_read_only,
        );

        let pipeline = {
            let bgl_refs = bind_group_layouts
                .iter()
                .map(Some)
                .collect::<SmallVec<[Option<&wgpu::BindGroupLayout>; 4]>>();
            let push_constant_size = u32::try_from(desc.constants)
                .map_err(|_| PipelineError::Failure("Invalid push constant size".to_owned()))?;
            let layout =
                self.inner
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &bgl_refs,
                        immediate_size: push_constant_size,
                    });
            self.inner
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(desc.name),
                    layout: Some(&layout),
                    module: desc.shader.library.wgpu(),
                    entry_point: Some(&desc.shader.entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };

        Ok(ComputePipeline::new(pipeline, bind_group_layouts.into()))
    }
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Device").finish()
    }
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for Device {}

impl crate::traits::Resource for Device {}

#[hidden_trait]
impl crate::traits::Device for Device {
    /// Create an independently recorded file I/O batch.
    ///
    /// # Panics
    /// Panics unless `Features::IO_COMMANDS` was enabled when creating this device.
    /// This backend does not support that feature.
    fn new_io_batch(&self) -> super::IoBatch {
        panic!("Features::IO_COMMANDS must be enabled to create an I/O batch")
    }

    fn new_shader_library(&self, desc: LibraryDesc) -> Result<Library, ShaderLibraryError> {
        match desc.input {
            LibraryInput::Source(source) => {
                if source.language != ShaderLanguage::Wgsl {
                    let code =
                        std::str::from_utf8(&source.code).map_err(ShaderLibraryError::NonUtf8)?;
                    // Try to parse and compile from other formats via naga
                    return Err(ShaderLibraryError::ValidationFailed);
                }
                let code =
                    std::str::from_utf8(&source.code).map_err(ShaderLibraryError::NonUtf8)?;
                let module = self
                    .inner
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(desc.name),
                        source: wgpu::ShaderSource::Wgsl(code.into()),
                    });
                Ok(Library::new(module))
            }
        }
    }

    fn new_compute_pipeline(
        &self,
        desc: ComputePipelineDesc,
    ) -> Result<ComputePipeline, PipelineError> {
        self.new_compute_pipeline_with_buffer_access(desc, &[])
    }

    fn new_render_pipeline(
        &self,
        desc: RenderPipelineDesc,
    ) -> Result<RenderPipeline, PipelineError> {
        let bind_group_layouts = create_bind_group_layouts(&self.inner.device, desc.arguments);
        let pipeline = {
            let bgl_refs = bind_group_layouts
                .iter()
                .map(Some)
                .collect::<SmallVec<[Option<&wgpu::BindGroupLayout>; 4]>>();

            let push_constant_size = u32::try_from(desc.constants)
                .map_err(|_| PipelineError::Failure("Invalid push constant size".to_owned()))?;

            let layout =
                self.inner
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &bgl_refs,
                        immediate_size: push_constant_size,
                    });

            let mut attrs: SmallVec<[SmallVec<[wgpu::VertexAttribute; 8]>; 4]> = SmallVec::new();
            for i in 0..desc.vertex_layouts.len() {
                attrs.push(
                    desc.vertex_attributes
                        .iter()
                        .enumerate()
                        .filter(|(_location, a)| a.buffer_index as usize == i)
                        .map(|(location, a)| wgpu::VertexAttribute {
                            format: a.format.into_wgpu(),
                            offset: a.offset as u64,
                            shader_location: location as u32,
                        })
                        .collect(),
                );
            }

            let vertex_buffers: Vec<Option<wgpu::VertexBufferLayout>> = desc
                .vertex_layouts
                .iter()
                .enumerate()
                .map(|(i, layout)| {
                    Some(wgpu::VertexBufferLayout {
                        array_stride: layout.stride as u64,
                        step_mode: layout.step_mode.into_wgpu(),
                        attributes: &attrs[i],
                    })
                })
                .collect();

            let targets: Vec<Option<wgpu::ColorTargetState>>;
            let depth_stencil: Option<wgpu::DepthStencilState>;
            let fragment: Option<wgpu::FragmentState>;

            if let Some(raster) = &desc.raster {
                targets = raster
                    .color_targets
                    .iter()
                    .map(|ct| {
                        let blend = ct.blend.as_ref().map(|b| wgpu::BlendState {
                            color: b.color.into_wgpu(),
                            alpha: b.alpha.into_wgpu(),
                        });
                        Some(wgpu::ColorTargetState {
                            format: ct.format.expect_into_wgpu(),
                            blend,
                            write_mask: ct.mask.into_wgpu(),
                        })
                    })
                    .collect();

                depth_stencil = raster
                    .depth_stencil
                    .as_ref()
                    .map(|ds| wgpu::DepthStencilState {
                        format: ds.format.expect_into_wgpu(),
                        depth_write_enabled: Some(ds.write_enabled),
                        depth_compare: Some(ds.compare.into_wgpu()),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    });

                fragment = raster
                    .fragment_shader
                    .as_ref()
                    .map(|fs| wgpu::FragmentState {
                        module: fs.library.wgpu(),
                        entry_point: Some(&fs.entry),
                        targets: &targets,
                        compilation_options: Default::default(),
                    });
            } else {
                targets = Vec::new();
                depth_stencil = None;
                fragment = None;
            }

            self.wgpu()
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(desc.name),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: desc.vertex_shader.library.wgpu(),
                        entry_point: Some(&desc.vertex_shader.entry),
                        buffers: &vertex_buffers,
                        compilation_options: Default::default(),
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: desc.primitive_topology.into_wgpu(),
                        strip_index_format: None,
                        front_face: desc
                            .raster
                            .as_ref()
                            .map(|r| r.front_face.into_wgpu())
                            .unwrap_or(wgpu::FrontFace::Ccw),
                        cull_mode: desc.raster.as_ref().and_then(|r| r.culling.into_wgpu()),
                        unclipped_depth: false,
                        polygon_mode: wgpu::PolygonMode::Fill,
                        conservative: false,
                    },
                    depth_stencil,
                    multisample: wgpu::MultisampleState::default(),
                    fragment,
                    multiview_mask: None,
                    cache: None,
                })
        };

        Ok(RenderPipeline::new(pipeline, bind_group_layouts.into()))
    }

    fn new_buffer(&self, desc: BufferDesc) -> Buffer {
        let buffer = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(desc.name),
            size: desc.size as u64,
            usage: desc.usage.into_wgpu(),
            mapped_at_creation: false,
        });
        Buffer::new(buffer, desc.size, desc.usage, false)
    }

    /// Creates a buffer with its full logical range mapped for writing.
    /// Requires `Features::BUFFER_MAP_AT_CREATION`; initial writes do not
    /// require `HOST_WRITE`. Drop mapped views and unmap before GPU use.
    fn new_buffer_mapped(&self, desc: BufferDesc) -> Buffer {
        let alignment = wgpu::COPY_BUFFER_ALIGNMENT as usize;
        let size = desc
            .size
            .checked_add(alignment - 1)
            .expect("Buffer size exceeds address space")
            & !(alignment - 1);
        let buffer = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(desc.name),
            size: size as u64,
            usage: desc.usage.into_wgpu(),
            mapped_at_creation: true,
        });
        Buffer::new(buffer, desc.size, desc.usage, true)
    }

    fn new_buffer_init<D: BufferInit>(&self, desc: BufferInitDesc<'_, D>) -> Buffer {
        let size = desc.data.size();
        let mut buffer = self.new_buffer_mapped(BufferDesc {
            size,
            usage: desc.usage,
            name: desc.name,
        });
        {
            let mut mapped = buffer
                .write_mapped_range(..)
                .expect("Failed to access initial buffer mapping");
            match desc.data.as_bytes() {
                Some(bytes) => mapped.write(0, bytes),
                None => {
                    // BufferInit closures need readable memory; WebGPU views are write-only.
                    let mut bytes = vec![0; size];
                    desc.data.write(&mut bytes);
                    mapped.write(0, &bytes);
                }
            }
        }
        buffer.unmap();
        buffer
    }

    fn new_image<T: ImageDims>(&self, desc: ImageDesc<T>) -> T::Image {
        desc.dims.validate();
        let extent = desc.dims.extent();
        let layers = desc.dims.layers();

        let (dimension, size) = match extent {
            ImageExtent::D1(e) => (
                wgpu::TextureDimension::D1,
                wgpu::Extent3d {
                    width: e.width(),
                    height: 1,
                    depth_or_array_layers: layers,
                },
            ),
            ImageExtent::D2(e) => (
                wgpu::TextureDimension::D2,
                wgpu::Extent3d {
                    width: e.width(),
                    height: e.height(),
                    depth_or_array_layers: layers,
                },
            ),
            ImageExtent::D3(e) => (
                wgpu::TextureDimension::D3,
                wgpu::Extent3d {
                    width: e.width(),
                    height: e.height(),
                    depth_or_array_layers: e.depth(),
                },
            ),
        };

        let texture = self.inner.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(desc.name),
            size,
            mip_level_count: desc.levels,
            sample_count: 1,
            dimension,
            format: desc.format.expect_into_wgpu(),
            usage: desc.usage.into_wgpu(),
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: None,
            dimension: None,
            usage: None,
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(desc.levels),
            base_array_layer: 0,
            array_layer_count: Some(layers),
        });

        T::wrap(Image::new(view, layers, desc.levels))
    }

    fn new_sampler(&self, desc: SamplerDesc) -> Sampler {
        let anisotropy = desc.anisotropy.map(|a| (a as u16).clamp(1, 16));
        let sampler = self.inner.device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: desc.address_mode[0].into_wgpu(),
            address_mode_v: desc.address_mode[1].into_wgpu(),
            address_mode_w: desc.address_mode[2].into_wgpu(),
            mag_filter: desc.mag_filter.into_wgpu(),
            min_filter: desc.min_filter.into_wgpu(),
            mipmap_filter: desc.mip_map_mode.into_wgpu(),
            lod_min_clamp: desc.min_lod,
            lod_max_clamp: desc.max_lod,
            compare: None,
            anisotropy_clamp: anisotropy.unwrap_or(1),
            border_color: None,
        });
        Sampler::new(sampler)
    }

    fn new_surface(
        &self,
        window: &impl HasWindowHandle,
        display: &impl HasDisplayHandle,
    ) -> Result<Surface, SurfaceError> {
        let window_handle = window
            .window_handle()
            .map_err(|_| SurfaceError::SurfaceLost)?;
        let display_handle = display
            .display_handle()
            .map_err(|_| SurfaceError::SurfaceLost)?;

        let surface = unsafe {
            self.inner
                .instance
                .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                    raw_display_handle: Some(display_handle.as_raw()),
                    raw_window_handle: window_handle.as_raw(),
                })
                .map_err(|_| SurfaceError::SurfaceLost)?
        };

        Surface::new(surface, self.clone())
    }

    fn new_fake_surface(&self, _image: Image) -> Result<Surface, SurfaceError> {
        todo!("fake surface not implemented for WebGPU backend")
    }

    fn new_blas(&self, _desc: BlasDesc) -> Blas {
        panic!("acceleration structures are not supported in the WebGPU backend")
    }

    fn new_tlas(&self, _desc: TlasDesc) -> Tlas {
        panic!("acceleration structures are not supported in the WebGPU backend")
    }
}

pub(super) fn create_bind_group_layouts(
    device: &wgpu::Device,
    argument_groups: &[ArgumentGroupLayout],
) -> Vec<wgpu::BindGroupLayout> {
    create_bind_group_layouts_with_buffer_access(device, argument_groups, &[])
}

fn create_bind_group_layouts_with_buffer_access(
    device: &wgpu::Device,
    argument_groups: &[ArgumentGroupLayout],
    storage_read_only: &[bool],
) -> Vec<wgpu::BindGroupLayout> {
    argument_groups
        .iter()
        .map(|group| {
            let entries: SmallVec<[wgpu::BindGroupLayoutEntry; 8]> = group
                .arguments
                .iter()
                .enumerate()
                .map(|(i, arg)| wgpu::BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: arg.stages.into_wgpu(),
                    ty: match arg.kind {
                        crate::ArgumentKind::StorageBuffer => wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage {
                                read_only: storage_read_only.get(i).copied().unwrap_or(false),
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        _ => arg.kind.into_wgpu(),
                    },
                    count: arg.size.map(|s| {
                        debug_assert!(
                            u32::try_from(s.get()).is_ok(),
                            "Bind group layout size exceeds u32::MAX"
                        );
                        NonZero::new(s.get() as u32).unwrap()
                    }),
                })
                .collect();

            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &entries,
            })
        })
        .collect()
}
