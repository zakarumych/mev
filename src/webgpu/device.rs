use std::{cell::RefCell, fmt, num::NonZero, rc::Rc};

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use smallvec::SmallVec;
use wgpu::util::DeviceExt;

use crate::{
    backend::from::{IntoWgpu, WgpuFrom},
    generic::{
        BlasDesc, BufferDesc, BufferInitDesc, ComputePipelineDesc, ImageDesc, ImageExtent,
        LibraryDesc, LibraryInput, PipelineError, RenderPipelineDesc, SamplerDesc, ShaderLanguage,
        ShaderLibraryError, SurfaceError, TlasDesc,
    },
    ArgumentGroupLayout,
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

const INITIAL_SCRATCH_SIZE: u64 = 1024; // 1 KB

struct ScratchBuffer {
    buffer: Option<wgpu::Buffer>,
    used: u64,
}

impl ScratchBuffer {
    fn new() -> Self {
        ScratchBuffer {
            buffer: None,
            used: 0,
        }
    }

    fn allocate(&mut self, device: &wgpu::Device, size: u64) -> (wgpu::Buffer, u64) {
        assert_ne!(size, 0, "Cannot allocate a scratch buffer of size 0");

        let mut buffer_size = self.buffer.as_ref().map_or(0, |b| b.size());

        if self.used + size > buffer_size {
            // If the requested size exceeds the current buffer size, create a new buffer
            let new_size = (self.used + size).max(buffer_size * 2);

            self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mev_scratch_buffer"),
                size: new_size,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.used = 0;

            buffer_size = new_size;
        }

        let offset = self.used;
        let buffer = self.buffer.clone().unwrap();
        self.used += size;

        (buffer, offset)
    }
}

struct DeviceInner {
    device: wgpu::Device,
    instance: Rc<wgpu::Instance>,
    adapter: wgpu::Adapter,
    scratch: RefCell<ScratchBuffer>,
}

#[derive(Clone)]
pub struct Device {
    inner: Rc<DeviceInner>,
}

impl Device {
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
                scratch: RefCell::new(ScratchBuffer::new()),
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

    pub(super) fn allocate_scratch(&self, size: u64) -> (wgpu::Buffer, u64) {
        let mut scratch = self.inner.scratch.borrow_mut();
        scratch.allocate(&self.inner.device, size)
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

#[hidden_trait::expose]
impl crate::traits::Device for Device {
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
        let bind_group_layouts = create_bind_group_layouts(&self.inner.device, desc.arguments);

        let pipeline = {
            let bgl_refs = bind_group_layouts
                .iter()
                .map(Some)
                .collect::<SmallVec<[Option<&wgpu::BindGroupLayout>; 4]>>();

            let push_constant_size = u32::try_from(desc.constants)
                .map_err(|_| PipelineError::Failure(format!("Invalid push constant size")))?;

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
                .map_err(|_| PipelineError::Failure(format!("Invalid push constant size")))?;

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

            let vertex_buffers: Vec<wgpu::VertexBufferLayout> = desc
                .vertex_layouts
                .iter()
                .enumerate()
                .map(|(i, layout)| wgpu::VertexBufferLayout {
                    array_stride: layout.stride as u64,
                    step_mode: layout.step_mode.into_wgpu(),
                    attributes: &attrs[i],
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
                            format: ct.format.into_wgpu(),
                            blend,
                            write_mask: ct
                                .blend
                                .as_ref()
                                .map(|b| b.mask.into_wgpu())
                                .unwrap_or(wgpu::ColorWrites::ALL),
                        })
                    })
                    .collect();

                depth_stencil = raster
                    .depth_stencil
                    .as_ref()
                    .map(|ds| wgpu::DepthStencilState {
                        format: ds.format.into_wgpu(),
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
        Buffer::new(buffer)
    }

    fn new_buffer_init(&self, desc: BufferInitDesc) -> Buffer {
        let buffer = self
            .inner
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(desc.name),
                contents: desc.data,
                usage: wgpu::BufferUsages::wgpu_from(desc.usage) | wgpu::BufferUsages::MAP_WRITE,
            });
        Buffer::new(buffer)
    }

    fn new_image(&self, desc: ImageDesc) -> Image {
        let (dimension, size) = match desc.extent {
            ImageExtent::D1(e) => (
                wgpu::TextureDimension::D1,
                wgpu::Extent3d {
                    width: e.width(),
                    height: 1,
                    depth_or_array_layers: desc.layers,
                },
            ),
            ImageExtent::D2(e) => (
                wgpu::TextureDimension::D2,
                wgpu::Extent3d {
                    width: e.width(),
                    height: e.height(),
                    depth_or_array_layers: desc.layers,
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
            format: desc.format.into_wgpu(),
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
            array_layer_count: Some(desc.layers),
        });

        Image::new(view, desc.layers, desc.levels)
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
                    ty: arg.kind.into_wgpu(),
                    count: NonZero::new(arg.size as u32),
                })
                .collect();

            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &entries,
            })
        })
        .collect()
}
