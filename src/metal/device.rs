use std::{fmt, sync::Arc};

use hashbrown::HashMap;
use objc2::{
    class, msg_send,
    rc::Retained,
    runtime::{AnyObject, Bool, ProtocolObject},
};
use objc2_core_foundation::{CGFloat, CGRect};
use objc2_foundation::{NSObject, NSString, NSUInteger};
use objc2_metal::{
    MTLCompileOptions, MTLComputePipelineDescriptor, MTLDevice, MTLLanguageVersion, MTLLibrary,
    MTLPipelineOption, MTLRenderPipelineDescriptor, MTLResource, MTLResourceOptions,
    MTLSamplerDescriptor, MTLStorageMode, MTLTextureDescriptor, MTLTextureType,
    MTLVertexAttributeDescriptor, MTLVertexBufferLayoutDescriptor, MTLVertexDescriptor,
    MTLVertexStepFunction,
};
use objc2_quartz_core::CAMetalLayer;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};

use crate::generic::{
    ArgumentKind, BlasDesc, BufferDesc, BufferInitDesc, ComputePipelineDesc, ImageDesc, ImageDims,
    ImageExtent, ImageUsage, LibraryDesc, LibraryInput, OutOfMemory, PipelineError, PixelFormat,
    RenderPipelineDesc, SamplerDesc, ShaderLanguage, ShaderLibraryError, SurfaceError, TlasDesc,
    VertexStepMode, parse_shader,
};

use super::{
    Blas, Buffer, ComputePipeline, Image, Library, MAX_VERTEX_BUFFERS, RenderPipeline, Sampler,
    Surface, Tlas,
    from::{IntoMetal, TryIntoMetal},
    shader::{Bindings, EntryPointData},
};

#[derive(Clone)]
pub struct Device {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
}

unsafe impl Sync for Device {}
unsafe impl Send for Device {}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Device")
            .field(&Retained::as_ptr(&self.device))
            .finish()
    }
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        Retained::as_ptr(&self.device) == Retained::as_ptr(&other.device)
    }
}

impl Eq for Device {}

impl Device {
    pub(super) fn new(device: Retained<ProtocolObject<dyn MTLDevice>>, queues: usize) -> Self {
        Device { device }
    }

    pub(super) fn set_last_cbuf(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        queues: usize,
    ) -> Self {
        Device { device }
    }
}

impl Device {
    #[inline(always)]
    pub(crate) fn metal(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }
}

impl crate::traits::Resource for Device {}

#[hidden_trait::expose]
impl crate::traits::Device for Device {
    fn new_shader_library(&self, desc: LibraryDesc) -> Result<Library, ShaderLibraryError> {
        match desc.input {
            LibraryInput::Source(source) => {
                let options = MTLCompileOptions::new();
                options.setLanguageVersion(MTLLanguageVersion::Version2_2);

                match source.language {
                    ShaderLanguage::Msl => {
                        let source = std::str::from_utf8(&*source.code)
                            .map_err(|err| ShaderLibraryError::NonUtf8(err))?;

                        let library = self
                            .device
                            .newLibraryWithSource_options_error(
                                &NSString::from_str(source),
                                Some(&options),
                            )
                            .unwrap();

                        Ok(Library::new(library))
                    }

                    src => {
                        let compiled = compile_shader(&source.code, source.filename, src)?;

                        let library = self
                            .device
                            .newLibraryWithSource_options_error(
                                &NSString::from_str(&compiled.code),
                                Some(&options),
                            )
                            .unwrap();

                        Ok(Library::with_entry_point_data(
                            library,
                            compiled.entry_point_data,
                        ))
                    }
                }
            }
        }
    }

    fn new_compute_pipeline(
        &self,
        desc: ComputePipelineDesc,
    ) -> Result<ComputePipeline, PipelineError> {
        let mdesc = MTLComputePipelineDescriptor::new();
        mdesc.setLabel(Some(&NSString::from_str(desc.name)));

        let compute_function = desc
            .shader
            .library
            .get_function(&desc.shader.entry)
            .ok_or_else(|| PipelineError::InvalidShaderEntry)?;

        mdesc.setComputeFunction(Some(&compute_function));

        let pipeline = self
            .device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &mdesc,
                MTLPipelineOption::None,
                None,
            )
            .map_err(|err| PipelineError::Failure(err.to_string()))?;

        Ok(ComputePipeline::new(
            pipeline,
            desc.shader.library.get_bindings(&desc.shader.entry),
            desc.shader.library.get_workgroup_size(&desc.shader.entry),
        ))
    }

    fn new_render_pipeline(
        &self,
        desc: RenderPipelineDesc,
    ) -> Result<RenderPipeline, PipelineError> {
        let mdesc = MTLRenderPipelineDescriptor::new();
        mdesc.setLabel(Some(&NSString::from_str(desc.name)));

        let vertex_function = desc
            .vertex_shader
            .library
            .get_function(&desc.vertex_shader.entry)
            .ok_or_else(|| PipelineError::InvalidShaderEntry)?;

        mdesc.setVertexFunction(Some(&vertex_function));

        let vertex_bindings = desc
            .vertex_shader
            .library
            .get_bindings(&desc.vertex_shader.entry);

        let vertex_buffers_count = desc
            .arguments
            .iter()
            .map(|g| {
                g.arguments
                    .iter()
                    .filter(|a| {
                        matches!(
                            a.kind,
                            ArgumentKind::UniformBuffer | ArgumentKind::StorageBuffer
                        )
                    })
                    .count()
            })
            .sum::<usize>()
            + vertex_bindings
                .as_ref()
                .map_or(0, |b| b.push_constants.is_some() as usize);

        if vertex_buffers_count > MAX_VERTEX_BUFFERS as _ {
            panic!("Too many buffer arguments and attribute buffers");
        }

        let mut fragment_bindings = None;

        let vertex_desc = MTLVertexDescriptor::new();

        let layouts = vertex_desc.layouts();
        for (idx, vertex_layout) in desc.vertex_layouts.iter().enumerate() {
            let layout_desc = MTLVertexBufferLayoutDescriptor::new();
            unsafe {
                layout_desc.setStride(vertex_layout.stride as _);
            }
            match vertex_layout.step_mode {
                VertexStepMode::Vertex => {
                    layout_desc.setStepFunction(MTLVertexStepFunction::PerVertex)
                }
                VertexStepMode::Instance { rate } => {
                    unsafe {
                        layout_desc.setStepRate(rate as _);
                    }
                    layout_desc.setStepFunction(MTLVertexStepFunction::PerInstance)
                }
                VertexStepMode::Constant => {
                    layout_desc.setStepFunction(MTLVertexStepFunction::Constant)
                }
            }
            unsafe {
                layouts.setObject_atIndexedSubscript(
                    Some(&layout_desc),
                    (vertex_buffers_count + idx) as _,
                );
            }
        }

        let attributes = vertex_desc.attributes();
        for (idx, vertex_attribute) in desc.vertex_attributes.iter().enumerate() {
            let attribute_desc = MTLVertexAttributeDescriptor::new();
            attribute_desc.setFormat(vertex_attribute.format.try_into_metal().unwrap());
            unsafe {
                attribute_desc.setOffset(vertex_attribute.offset as _);
                attribute_desc.setBufferIndex(
                    (vertex_buffers_count as u32 + vertex_attribute.buffer_index) as _,
                );
                attributes.setObject_atIndexedSubscript(Some(&attribute_desc), idx as _);
            }
        }

        mdesc.setVertexDescriptor(Some(&vertex_desc));
        unsafe {
            mdesc.setInputPrimitiveTopology(desc.primitive_topology.into_metal());
        }

        if let Some(raster) = desc.raster {
            if let Some(fragment_shader) = raster.fragment_shader {
                let fragment_function = fragment_shader
                    .library
                    .get_function(&fragment_shader.entry)
                    .ok_or_else(|| PipelineError::InvalidShaderEntry)?;

                mdesc.setFragmentFunction(Some(&fragment_function));

                fragment_bindings = fragment_shader.library.get_bindings(&fragment_shader.entry);
            }

            let color_attachments = mdesc.colorAttachments();
            for (idx, color_desc) in raster.color_targets.iter().enumerate() {
                let color_attachment =
                    unsafe { color_attachments.objectAtIndexedSubscript(idx as _) };
                color_attachment.setPixelFormat(color_desc.format.try_into_metal().unwrap());
                color_attachment.setWriteMask(color_desc.mask.into_metal());

                if let Some(blend_desc) = &color_desc.blend {
                    color_attachment.setBlendingEnabled(true);
                    color_attachment.setRgbBlendOperation(blend_desc.color.op.into_metal());
                    color_attachment.setSourceRGBBlendFactor(blend_desc.color.src.into_metal());
                    color_attachment
                        .setDestinationRGBBlendFactor(blend_desc.color.dst.into_metal());
                    color_attachment.setAlphaBlendOperation(blend_desc.alpha.op.into_metal());
                    color_attachment.setSourceAlphaBlendFactor(blend_desc.alpha.src.into_metal());
                    color_attachment
                        .setDestinationAlphaBlendFactor(blend_desc.alpha.dst.into_metal());
                } else {
                    color_attachment.setBlendingEnabled(false);
                }
                unsafe {
                    color_attachments
                        .setObject_atIndexedSubscript(Some(&color_attachment), idx as _);
                }
            }

            if let Some(depth_stencil) = raster.depth_stencil {
                let format = depth_stencil.format.try_into_metal().unwrap();
                if depth_stencil.format.is_depth() {
                    mdesc.setDepthAttachmentPixelFormat(format);
                }
                if depth_stencil.format.is_stencil() {
                    mdesc.setStencilAttachmentPixelFormat(format);
                }
            }
        }

        let pipeline = self
            .device
            .newRenderPipelineStateWithDescriptor_error(&mdesc)
            .map_err(|err| PipelineError::Failure(err.to_string()))?;

        Ok(RenderPipeline::new(
            pipeline,
            desc.primitive_topology.into_metal(),
            vertex_bindings,
            fragment_bindings,
            vertex_buffers_count as u32,
        ))
    }

    fn new_buffer(&self, desc: BufferDesc) -> Buffer {
        let mut options = MTLResourceOptions::empty();

        if desc
            .usage
            .intersects(crate::BufferUsage::HOST_READ | crate::BufferUsage::HOST_WRITE)
        {
            if desc.usage.contains(crate::BufferUsage::HOST_WRITE) {
                options |= MTLResourceOptions::CPUCacheModeWriteCombined;
            }
            options |= MTLResourceOptions::StorageModeShared;
        } else {
            options |= MTLResourceOptions::StorageModePrivate;
        }

        let buffer = self
            .device
            .newBufferWithLength_options(desc.size as NSUInteger, options)
            .expect("Failed to create buffer");
        buffer.setLabel(Some(&NSString::from_str(desc.name)));
        Buffer::new(buffer, desc.usage)
    }

    fn new_buffer_init(&self, desc: BufferInitDesc) -> Buffer {
        let Ok(len) = u64::try_from(desc.data.len()) else {
            size_exceeds_u64();
        };

        let mut options = MTLResourceOptions::empty();

        if desc
            .usage
            .intersects(crate::BufferUsage::HOST_READ | crate::BufferUsage::HOST_WRITE)
        {
            if desc.usage.contains(crate::BufferUsage::HOST_WRITE) {
                options |= MTLResourceOptions::CPUCacheModeWriteCombined;
            }
            options |= MTLResourceOptions::StorageModeShared;
        } else {
            options |= MTLResourceOptions::StorageModePrivate;
        }

        let buffer = unsafe {
            self.device.newBufferWithBytes_length_options(
                std::ptr::NonNull::from(desc.data).cast(),
                len as NSUInteger,
                options,
            )
        }
        .expect("Failed to create buffer");
        buffer.setLabel(Some(&NSString::from_str(desc.name)));
        Buffer::new(buffer, desc.usage)
    }

    fn new_image<T: ImageDims>(&self, desc: ImageDesc<T>) -> T::Image {
        desc.dims.validate();
        let extent = desc.dims.extent();
        let layers = desc.dims.layers();

        let mdesc = MTLTextureDescriptor::new();
        mdesc.setPixelFormat(desc.format.try_into_metal().unwrap());
        unsafe {
            match extent {
                ImageExtent::D1(extent) => {
                    mdesc.setTextureType(MTLTextureType::Type1D);
                    mdesc.setWidth(extent.width() as _);
                }
                ImageExtent::D2(extent) => {
                    mdesc.setTextureType(MTLTextureType::Type2D);
                    mdesc.setWidth(extent.width() as _);
                    mdesc.setHeight(extent.height() as _);
                }
                ImageExtent::D3(extent) => {
                    mdesc.setTextureType(MTLTextureType::Type3D);
                    mdesc.setWidth(extent.width() as _);
                    mdesc.setHeight(extent.height() as _);
                    mdesc.setDepth(extent.depth() as _);
                }
            }
            mdesc.setMipmapLevelCount(desc.levels as _);
            mdesc.setArrayLength(layers as _);
            mdesc.setSampleCount(1);
        }
        mdesc.setUsage(desc.usage.into_metal());
        mdesc.setStorageMode(MTLStorageMode::Private);

        let texture = self
            .device
            .newTextureWithDescriptor(&mdesc)
            .expect("Failed to create texture");
        texture.setLabel(Some(&NSString::from_str(desc.name)));
        T::wrap(Image::new(texture))
    }

    fn new_sampler(&self, desc: SamplerDesc) -> Sampler {
        let mdesc = MTLSamplerDescriptor::new();
        mdesc.setMinFilter(desc.min_filter.into_metal());
        mdesc.setMagFilter(desc.mag_filter.into_metal());
        mdesc.setMipFilter(desc.mip_map_mode.into_metal());
        mdesc.setSAddressMode(desc.address_mode[0].into_metal());
        mdesc.setTAddressMode(desc.address_mode[1].into_metal());
        mdesc.setRAddressMode(desc.address_mode[2].into_metal());
        if let Some(anisotropy) = desc.anisotropy {
            mdesc.setMaxAnisotropy((anisotropy as NSUInteger).clamp(1, 16));
        }
        mdesc.setLodMinClamp(desc.min_lod);
        mdesc.setLodMaxClamp(desc.max_lod);
        mdesc.setNormalizedCoordinates(desc.normalized);
        let state = self
            .device
            .newSamplerStateWithDescriptor(&mdesc)
            .expect("Failed to create sampler state");
        Sampler::new(state)
    }

    fn new_surface(
        &self,
        window: &impl HasWindowHandle,
        display: &impl HasDisplayHandle,
    ) -> Result<Surface, SurfaceError> {
        let window = window
            .window_handle()
            .map_err(|_| SurfaceError::SurfaceLost)?;
        let display = display
            .display_handle()
            .map_err(|_| SurfaceError::SurfaceLost)?;
        match (window.as_raw(), display.as_raw()) {
            (RawWindowHandle::UiKit(handle), RawDisplayHandle::UiKit(_)) => unsafe {
                let view_ptr = handle.ui_view.cast().as_ptr();
                let layer = layer_from_view(view_ptr);
                layer.setDevice(Some(&self.device));
                let view = Retained::retain(view_ptr.cast::<NSObject>());
                Ok(Surface::new(layer, view))
            },
            (RawWindowHandle::AppKit(handle), RawDisplayHandle::AppKit(_)) => unsafe {
                let view_ptr = handle.ns_view.cast().as_ptr();
                let layer = layer_from_view(view_ptr);
                layer.setDevice(Some(&self.device));
                let view = Retained::retain(view_ptr.cast::<NSObject>());
                Ok(Surface::new(layer, view))
            },
            (RawWindowHandle::UiKit(_), _) | (RawWindowHandle::AppKit(_), _) => {
                panic!("Mismatched window and display type")
            }
            _ => unreachable!("Unsupported window type for the metal backend"),
        }
    }

    fn new_fake_surface(&self, image: Image) -> Result<Surface, SurfaceError> {
        Ok(Surface::fake(image))
    }

    fn new_blas(&self, desc: BlasDesc) -> Blas {
        let Ok(size) = u64::try_from(desc.size) else {
            size_exceeds_u64();
        };
        let blas = self
            .device
            .newAccelerationStructureWithSize(size as NSUInteger)
            .expect("Failed to create acceleration structure");
        // blas.setLabel(desc.name);
        Blas::new(blas)
    }

    fn new_tlas(&self, desc: TlasDesc) -> Tlas {
        let Ok(size) = u64::try_from(desc.size) else {
            size_exceeds_u64();
        };
        let tlas = self
            .device
            .newAccelerationStructureWithSize(size as NSUInteger)
            .expect("Failed to create acceleration structure");
        // tlas.setLabel(desc.name);
        Tlas::new(tlas)
    }
}

unsafe fn layer_from_view(view: *mut AnyObject) -> Retained<CAMetalLayer> {
    unsafe {
        let main_layer: *mut AnyObject = msg_send![view, layer];
        let class = class!(CAMetalLayer);
        let is_valid_layer: Bool = msg_send![main_layer, isKindOfClass: class];

        if is_valid_layer == Bool::YES {
            Retained::retain(main_layer.cast::<CAMetalLayer>())
                .expect("view's layer must not be null")
        } else {
            let new_layer = CAMetalLayer::new();
            let frame: CGRect = msg_send![main_layer, bounds];
            new_layer.setFrame(frame);
            #[cfg(target_os = "macos")]
            {
                let () = msg_send![view, setLayer: &*new_layer];
                let () = msg_send![view, setWantsLayer: Bool::YES];
                let () = msg_send![&*new_layer, setContentsGravity: kCAGravityTopLeft];
                let window: *mut AnyObject = msg_send![view, window];
                if !window.is_null() {
                    let scale_factor: CGFloat = msg_send![window, backingScaleFactor];
                    let () = msg_send![&*new_layer, setContentsScale: scale_factor];
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                let () = msg_send![main_layer, addSublayer: &*new_layer];
                let () = msg_send![main_layer, setAutoresizingMask: 0x1Fu64];
                let screen: *mut AnyObject = msg_send![class!(UIScreen), mainScreen];
                let scale_factor: CGFloat = msg_send![screen, nativeScale];
                let () = msg_send![view, setContentScaleFactor: scale_factor];
            }
            new_layer
        }
    }
}

#[cfg(target_os = "macos")]
#[link(name = "QuartzCore", kind = "framework")]
unsafe extern "C" {
    #[allow(non_upper_case_globals)]
    static kCAGravityTopLeft: *mut AnyObject;
}

struct CompiledMetalShader {
    code: String,
    entry_point_data: HashMap<String, EntryPointData>,
}

fn compile_shader(
    code: &[u8],
    filename: Option<&str>,
    lang: ShaderLanguage,
) -> Result<CompiledMetalShader, ShaderLibraryError> {
    let (module, info, _source_code) = parse_shader(code, filename, lang)?;

    let mut options = naga::back::msl::Options {
        lang_version: (2, 4),
        per_entry_point_map: Default::default(),
        inline_samplers: Vec::new(),
        spirv_cross_compatibility: false,
        fake_missing_bindings: true,
        bounds_check_policies: Default::default(),
        zero_initialize_workgroup_memory: false,
        force_loop_bounding: true,
    };

    let mut entry_point_data = HashMap::new();

    for (i, entry) in module.entry_points.iter().enumerate() {
        let mut bindings = Bindings::new();
        let mut map = naga::back::msl::EntryPointResources::default();
        map.sizes_buffer = Some(30);
        let mut next_buffer_slot = 0u8;
        let mut next_texture_slot = 0u8;
        let mut next_sampler_slot = 0u8;

        for (global_handle, global_variable) in module.global_variables.iter() {
            let function_info = info.get_entry_point(i);
            let usage = function_info[global_handle];
            if usage.is_empty() {
                continue;
            }

            if let naga::AddressSpace::Immediate = global_variable.space {
                map.immediates_buffer = Some(next_buffer_slot);
                bindings.set_push_constants(next_buffer_slot);
                next_buffer_slot += 1;
                continue;
            }

            if let Some(binding) = global_variable.binding.clone() {
                match global_variable.space {
                    naga::AddressSpace::Immediate => unreachable!(),
                    naga::AddressSpace::Uniform => {
                        map.resources.insert(
                            binding.clone(),
                            naga::back::msl::BindTarget {
                                buffer: Some(next_buffer_slot),
                                texture: None,
                                sampler: None,
                                external_texture: None,
                                mutable: false,
                            },
                        );
                        bindings.insert(binding, next_buffer_slot);
                        next_buffer_slot += 1;
                    }
                    naga::AddressSpace::Storage { access } => {
                        map.resources.insert(
                            binding.clone(),
                            naga::back::msl::BindTarget {
                                buffer: Some(next_buffer_slot),
                                texture: None,
                                sampler: None,
                                external_texture: None,
                                mutable: access.contains(naga::StorageAccess::STORE),
                            },
                        );
                        bindings.insert(binding, next_buffer_slot);
                        next_buffer_slot += 1;
                    }
                    naga::AddressSpace::Handle => {
                        let ty = &module.types[global_variable.ty];
                        match ty.inner {
                            naga::TypeInner::Sampler { .. } => {
                                map.resources.insert(
                                    binding.clone(),
                                    naga::back::msl::BindTarget {
                                        buffer: None,
                                        texture: None,
                                        sampler: Some(
                                            naga::back::msl::BindSamplerTarget::Resource(
                                                next_sampler_slot,
                                            ),
                                        ),
                                        external_texture: None,
                                        mutable: false,
                                    },
                                );
                                bindings.insert(binding, next_sampler_slot);
                                next_sampler_slot += 1;
                            }
                            naga::TypeInner::Image { class, .. } => {
                                map.resources.insert(
                                    binding.clone(),
                                    naga::back::msl::BindTarget {
                                        buffer: None,
                                        texture: Some(next_texture_slot),
                                        sampler: None,
                                        external_texture: None,
                                        mutable: matches!(class, naga::ImageClass::Storage { .. }),
                                    },
                                );
                                bindings.insert(binding, next_texture_slot);
                                next_texture_slot += 1;
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
        }

        options.per_entry_point_map.insert(entry.name.clone(), map);

        entry_point_data.insert(
            entry.name.clone(),
            EntryPointData {
                bindings: Arc::new(bindings),
                workgroup_size: entry.workgroup_size,
                name: Ok(String::new()),
            },
        );
    }

    let (code, translation) = naga::back::msl::write_string(
        &module,
        &info,
        &options,
        &naga::back::msl::PipelineOptions {
            entry_point: None,
            allow_and_force_point_size: false,
            vertex_pulling_transform: true,
            vertex_buffer_mappings: vec![],
        },
    )
    .map_err(ShaderLibraryError::GenMsl)?;

    // eprintln!("{}", code);
    // eprintln!("{:?}", translation.entry_point_names);

    for (i, entry) in module.entry_points.iter().enumerate() {
        entry_point_data.get_mut(&entry.name).unwrap().name =
            translation.entry_point_names[i].clone();
    }

    Ok(CompiledMetalShader {
        code,
        entry_point_data,
    })
}

fn size_exceeds_u64() -> ! {
    panic!("Buffer data length exceeds u64::MAX");
}
