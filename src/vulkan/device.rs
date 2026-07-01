use std::{
    any::TypeId,
    ffi, fmt,
    hash::{Hash, Hasher},
    ptr::NonNull,
    sync::{
        Arc, Weak,
        atomic::{AtomicU16, Ordering},
    },
};

use ash::vk::{self, Handle};
use gpu_alloc::{AllocationFlags, MemoryBlock};
use hashbrown::HashMap;
use parking_lot::Mutex;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use slab::Slab;
use smallvec::SmallVec;

use crate::{
    BufferUsage,
    backend::new_semaphore,
    generic::{
        BlasDesc, BufferDesc, BufferInitDesc, ComputePipelineDesc, DeviceError, Features,
        ImageDesc, LibraryDesc, LibraryInput, OutOfMemory, PipelineError, PrimitiveTopology,
        RenderPipelineDesc, SamplerDesc, ShaderLanguage, ShaderLibraryError, SurfaceError, Swizzle,
        TlasDesc, VertexStepMode, ViewDesc, parse_shader,
    },
};

use super::{
    Blas,
    ComputePipeline,
    Sampler,
    Tlas,
    Version,
    arguments::descriptor_type,
    buffer::Buffer,
    format_aspect,
    from::{IntoAsh, TryIntoAsh},
    handle_host_oom,
    image::Image,
    instance::InstanceGuard,
    layout::{
        DescriptorSetLayout, DescriptorSetLayoutDesc, PipelineLayout, PipelineLayoutDesc,
        WeakDescriptorSetLayout, WeakPipelineLayout,
    },
    // queue::PendingEpochs,
    render::RenderPipeline,
    sampler::WeakSampler,
    shader::Library,
    surface::Surface,
    unexpected_error,
};

#[derive(Debug)]
pub(super) struct DeviceMemory {
    handle: vk::DeviceMemory,
    idx: usize,
}

impl gpu_alloc::MemoryDevice<DeviceMemory> for DeviceInner {
    #[inline]
    unsafe fn allocate_memory(
        &self,
        size: u64,
        memory_type: u32,
        flags: gpu_alloc::AllocationFlags,
    ) -> Result<DeviceMemory, gpu_alloc::OutOfMemory> {
        assert!((flags & !(gpu_alloc::AllocationFlags::DEVICE_ADDRESS)).is_empty());

        let mut info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(memory_type);

        let mut info_flags;

        if flags.contains(AllocationFlags::DEVICE_ADDRESS) {
            info_flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            info = info.push_next(&mut info_flags);
        }

        let result = unsafe { self.device.allocate_memory(&info, None) };
        let handle = match result {
            Ok(handle) => handle,
            Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                return Err(gpu_alloc::OutOfMemory::OutOfDeviceMemory);
            }
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY) => {
                return Err(gpu_alloc::OutOfMemory::OutOfHostMemory);
            }
            Err(vk::Result::ERROR_TOO_MANY_OBJECTS) => panic!("Too many objects"),
            Err(err) => unexpected_error(err),
        };

        let idx = self.memory.lock().insert(handle);
        Ok(DeviceMemory { handle, idx })
    }

    #[inline]
    unsafe fn deallocate_memory(&self, memory: DeviceMemory) {
        unsafe {
            self.device.free_memory(memory.handle, None);
        }

        self.memory.lock().remove(memory.idx);
    }

    #[inline]
    unsafe fn map_memory(
        &self,
        memory: &mut DeviceMemory,
        offset: u64,
        size: u64,
    ) -> Result<NonNull<u8>, gpu_alloc::DeviceMapError> {
        let result = unsafe {
            self.device
                .map_memory(memory.handle, offset, size, vk::MemoryMapFlags::empty())
        };
        match result {
            Ok(ptr) => {
                Ok(NonNull::new(ptr as *mut u8)
                    .expect("Pointer to memory mapping must not be null"))
            }
            Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                Err(gpu_alloc::DeviceMapError::OutOfDeviceMemory)
            }
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY) => {
                Err(gpu_alloc::DeviceMapError::OutOfHostMemory)
            }
            Err(vk::Result::ERROR_MEMORY_MAP_FAILED) => Err(gpu_alloc::DeviceMapError::MapFailed),
            Err(err) => panic!("Unexpected Vulkan error: `{}`", err),
        }
    }

    #[inline]
    unsafe fn unmap_memory(&self, memory: &mut DeviceMemory) {
        unsafe {
            self.device.unmap_memory(memory.handle);
        }
    }

    #[inline]
    unsafe fn invalidate_memory_ranges(
        &self,
        ranges: &[gpu_alloc::MappedMemoryRange<'_, DeviceMemory>],
    ) -> Result<(), gpu_alloc::OutOfMemory> {
        let result = unsafe {
            self.device.invalidate_mapped_memory_ranges(
                &ranges
                    .iter()
                    .map(|range| {
                        vk::MappedMemoryRange::default()
                            .memory(range.memory.handle)
                            .offset(range.offset)
                            .size(range.size)
                    })
                    .collect::<SmallVec<[_; 4]>>(),
            )
        };

        result.map_err(|err| match err {
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => gpu_alloc::OutOfMemory::OutOfDeviceMemory,
            vk::Result::ERROR_OUT_OF_HOST_MEMORY => gpu_alloc::OutOfMemory::OutOfHostMemory,
            err => panic!("Unexpected Vulkan error: `{}`", err),
        })
    }

    #[inline]
    unsafe fn flush_memory_ranges(
        &self,
        ranges: &[gpu_alloc::MappedMemoryRange<'_, DeviceMemory>],
    ) -> Result<(), gpu_alloc::OutOfMemory> {
        let result = unsafe {
            self.device.flush_mapped_memory_ranges(
                &ranges
                    .iter()
                    .map(|range| {
                        vk::MappedMemoryRange::default()
                            .memory(range.memory.handle)
                            .offset(range.offset)
                            .size(range.size)
                    })
                    .collect::<SmallVec<[_; 4]>>(),
            )
        };

        result.map_err(|err| match err {
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => gpu_alloc::OutOfMemory::OutOfDeviceMemory,
            vk::Result::ERROR_OUT_OF_HOST_MEMORY => gpu_alloc::OutOfMemory::OutOfHostMemory,
            err => panic!("Unexpected Vulkan error: `{}`", err),
        })
    }
}

struct DescriptorUpdateTemplateEntries {
    entries: Vec<ash::vk::DescriptorUpdateTemplateEntry>,
}

impl PartialEq for DescriptorUpdateTemplateEntries {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.entries.iter().zip(other.entries.iter()).all(|(a, b)| {
            a.dst_binding == b.dst_binding
                && a.dst_array_element == b.dst_array_element
                && a.descriptor_count == b.descriptor_count
                && a.descriptor_type == b.descriptor_type
                && a.offset == b.offset
                && a.stride == b.stride
        })
    }

    #[inline]
    fn ne(&self, other: &Self) -> bool {
        self.entries.iter().zip(other.entries.iter()).any(|(a, b)| {
            a.dst_binding != b.dst_binding
                && a.dst_array_element != b.dst_array_element
                && a.descriptor_count != b.descriptor_count
                && a.descriptor_type != b.descriptor_type
                && a.offset != b.offset
                && a.stride != b.stride
        })
    }
}

impl Eq for DescriptorUpdateTemplateEntries {}

impl Hash for DescriptorUpdateTemplateEntries {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        for entry in &self.entries {
            entry.dst_binding.hash(state);
            entry.dst_array_element.hash(state);
            entry.descriptor_count.hash(state);
            entry.descriptor_type.hash(state);
            entry.offset.hash(state);
            entry.stride.hash(state);
        }
    }
}

struct ErrorState {
    state_flags: AtomicU16,
}

impl ErrorState {
    const DEVICE_OUT_OF_MEMORY_FLAG: u16 = 0x01;
    const DEVICE_LOST_FLAG: u16 = 0x02;

    #[inline]
    fn new() -> Self {
        ErrorState {
            state_flags: AtomicU16::new(0),
        }
    }

    /// Whenever the device runs out of memory, this function should be called to set the sticky OOM flag.
    #[cold]
    #[inline]
    fn set_oom(&self) {
        self.state_flags
            .fetch_or(Self::DEVICE_OUT_OF_MEMORY_FLAG, Ordering::Relaxed);
    }

    /// Whenever the device is lost, this function should be called to set the sticky device lost flag.
    #[cold]
    #[inline]
    fn set_lost(&self) {
        self.state_flags
            .fetch_or(Self::DEVICE_LOST_FLAG, Ordering::Relaxed);
    }

    #[inline]
    fn is_oom(&self) -> bool {
        (self.state_flags.load(Ordering::Relaxed) & Self::DEVICE_OUT_OF_MEMORY_FLAG)
            == Self::DEVICE_OUT_OF_MEMORY_FLAG
    }

    #[inline]
    fn is_lost(&self) -> bool {
        (self.state_flags.load(Ordering::Relaxed) & Self::DEVICE_LOST_FLAG)
            == Self::DEVICE_LOST_FLAG
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.state_flags.load(Ordering::Relaxed) == 0
    }

    #[inline]
    fn is_some(&self) -> bool {
        self.state_flags.load(Ordering::Relaxed) != 0
    }

    #[inline]
    fn get_error(&self) -> Result<(), DeviceError> {
        let flags = self.state_flags.load(Ordering::Relaxed);
        if (flags & Self::DEVICE_LOST_FLAG) == Self::DEVICE_LOST_FLAG {
            Err(DeviceError::DeviceLost)
        } else if (flags & Self::DEVICE_OUT_OF_MEMORY_FLAG) == Self::DEVICE_OUT_OF_MEMORY_FLAG {
            Err(DeviceError::OutOfMemory)
        } else {
            Ok(())
        }
    }
}

pub(super) struct DeviceInner {
    _guard: Arc<InstanceGuard>,
    device: ash::Device,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    version: Version,
    families: Vec<u32>,
    features: Features,
    properties: ash::vk::PhysicalDeviceProperties,

    error_state: ErrorState,

    memory: Mutex<Slab<vk::DeviceMemory>>,
    buffers: Mutex<Slab<vk::Buffer>>,
    images: Mutex<Slab<vk::Image>>,
    image_views: Mutex<Slab<vk::ImageView>>,
    samplers: Mutex<HashMap<SamplerDesc, WeakSampler>>,

    libraries: Mutex<Slab<vk::ShaderModule>>,
    set_layouts: Mutex<HashMap<DescriptorSetLayoutDesc, WeakDescriptorSetLayout>>,
    pipeline_layouts: Mutex<HashMap<PipelineLayoutDesc, WeakPipelineLayout>>,
    pipelines: Mutex<Slab<vk::Pipeline>>,

    allocator: Mutex<gpu_alloc::GpuAllocator<DeviceMemory>>,

    // # Extensions
    push_descriptor: ash::khr::push_descriptor::Device,
    surface: Option<ash::khr::surface::Instance>,
    swapchain: Option<ash::khr::swapchain::Device>,
    swapchain_maintenance1: Option<ash::ext::swapchain_maintenance1::Device>,

    #[cfg(target_os = "windows")]
    win32_surface: Option<ash::khr::win32_surface::Instance>,

    #[cfg(any(debug_assertions, feature = "debug"))]
    debug_utils: Option<ash::ext::debug_utils::Device>,
}

impl Drop for DeviceInner {
    fn drop(&mut self) {
        if let Err(err) = unsafe { self.device.device_wait_idle() } {
            tracing::error!("Failed to wait for device idle: {}", err);
        }

        for buffer in self.buffers.get_mut().drain() {
            unsafe {
                self.device.destroy_buffer(buffer, None);
            }
        }

        for image_view in self.image_views.get_mut().drain() {
            unsafe {
                self.device.destroy_image_view(image_view, None);
            }
        }

        for image in self.images.get_mut().drain() {
            unsafe {
                self.device.destroy_image(image, None);
            }
        }

        for memory in self.memory.get_mut().drain() {
            unsafe {
                self.device.free_memory(memory, None);
            }
        }

        for sampler in self.samplers.get_mut().values_mut() {
            if !sampler.unused() {
                unsafe {
                    self.device.destroy_sampler(sampler.handle(), None);
                }
            }
        }

        for pipeline in self.pipelines.get_mut().drain() {
            unsafe {
                self.device.destroy_pipeline(pipeline, None);
            }
        }

        for pipeline_layout in self.pipeline_layouts.get_mut().values_mut() {
            if let Some(pipeline_layout) = pipeline_layout.upgrade() {
                unsafe {
                    self.device
                        .destroy_pipeline_layout(pipeline_layout.handle(), None);
                }
                unsafe {
                    for template in pipeline_layout.templates().lock().values().copied() {
                        self.device
                            .destroy_descriptor_update_template(template, None);
                    }
                }
            }
        }

        for set_layout in self.set_layouts.get_mut().values_mut() {
            if !set_layout.unused() {
                unsafe {
                    self.device
                        .destroy_descriptor_set_layout(set_layout.handle(), None);
                }
            }
        }

        for library in self.libraries.get_mut().drain() {
            unsafe {
                self.device.destroy_shader_module(library, None);
            }
        }

        unsafe {
            self.device.destroy_device(None);
        }
    }
}

#[derive(Clone)]
pub(super) struct WeakDevice {
    inner: std::sync::Weak<DeviceInner>,
}

impl WeakDevice {
    /// Creates a WeakDevice that will never upgrade (for null/OOM resources).
    pub(super) fn null() -> Self {
        WeakDevice { inner: Weak::new() }
    }

    #[inline]
    pub fn upgrade(&self) -> Option<Device> {
        self.inner.upgrade().map(|inner| Device { inner })
    }

    #[inline]
    pub fn drop_buffer(&self, idx: usize, block: MemoryBlock<DeviceMemory>) {
        if let Some(inner) = self.inner.upgrade() {
            unsafe { inner.allocator.lock().dealloc(&*inner, block) }

            let mut buffers = inner.buffers.lock();
            let buffer = buffers.remove(idx);
            unsafe {
                inner.device.destroy_buffer(buffer, None);
            }
        }
    }

    #[inline]
    pub fn drop_image(&self, idx: usize, block: MemoryBlock<DeviceMemory>) {
        if let Some(inner) = self.inner.upgrade() {
            unsafe { inner.allocator.lock().dealloc(&*inner, block) }

            let mut images = inner.images.lock();
            let image = images.remove(idx);
            unsafe {
                inner.device.destroy_image(image, None);
            }
        }
    }

    #[inline]
    pub fn drop_sampler(&self, desc: SamplerDesc) {
        if let Some(inner) = self.inner.upgrade() {
            let mut samplers = inner.samplers.lock();
            match samplers.entry(desc) {
                hashbrown::hash_map::Entry::Occupied(entry) => {
                    let weak = entry.get();
                    // It is only safe to drop when no strong refs exist.
                    // While this function is called when last strong reference is dropped
                    // the weak could have been upgraded to strong reference before lock was acquired.
                    if weak.unused() {
                        // No strong references exists.
                        unsafe {
                            inner.device.destroy_sampler(weak.handle(), None);
                        }
                    }
                }
                _ => {
                    // Entry was removed.
                    // It could have been reused and dropped again before lock was acquired.
                    // No action is needed in this case.
                }
            }
        }
    }

    #[inline]
    pub fn drop_descriptor_set_layout(&self, desc: DescriptorSetLayoutDesc) {
        if let Some(inner) = self.inner.upgrade() {
            let mut samplers = inner.set_layouts.lock();
            match samplers.entry(desc) {
                hashbrown::hash_map::Entry::Occupied(entry) => {
                    let weak = entry.get();
                    // It is only safe to drop when no strong refs exist.
                    // While this function is called when last strong reference is dropped
                    // the entry could be replaced by new layout before lock was acquired.
                    if weak.unused() {
                        // No strong references exists.
                        unsafe {
                            inner
                                .device
                                .destroy_descriptor_set_layout(weak.handle(), None);
                        }
                    }
                }
                _ => {
                    // Entry was removed.
                    // It could have been reused and dropped again before lock was acquired.
                    // No action is needed in this case.
                }
            }
        }
    }

    #[inline]
    pub fn drop_pipeline_layout(
        &self,
        desc: PipelineLayoutDesc,
        templates: impl Iterator<Item = ash::vk::DescriptorUpdateTemplate>,
    ) {
        if let Some(inner) = self.inner.upgrade() {
            unsafe {
                for template in templates {
                    inner
                        .device
                        .destroy_descriptor_update_template(template, None);
                }
            }

            let mut pipeline_layouts = inner.pipeline_layouts.lock();
            match pipeline_layouts.entry(desc) {
                hashbrown::hash_map::Entry::Occupied(entry) => {
                    let weak = entry.get();
                    // It is only safe to drop when no strong refs exist.
                    // While this function is called when last strong reference is dropped
                    // the entry could be replaced by new layout before lock was acquired.
                    if weak.unused() {
                        // No strong references exists.
                        unsafe {
                            inner.device.destroy_pipeline_layout(weak.handle(), None);
                        }
                    }
                }
                _ => {
                    // Entry was removed.
                    // It could have been reused and dropped again before lock was acquired.
                    // No action is needed in this case.
                }
            }
        }
    }

    #[inline]
    pub fn drop_pipeline(&self, idx: usize) {
        if let Some(inner) = self.inner.upgrade() {
            let pipeline = inner.pipelines.lock().remove(idx);
            unsafe {
                inner.device.destroy_pipeline(pipeline, None);
            }
        }
    }

    #[inline]
    pub fn drop_image_view(&self, idx: usize) {
        if let Some(inner) = self.inner.upgrade() {
            let mut image_views = inner.image_views.lock();
            let view = image_views.remove(idx);
            unsafe {
                inner.device.destroy_image_view(view, None);
            }
        }
    }

    #[inline]
    pub fn drop_image_views(&self, iter: impl Iterator<Item = usize>) {
        if let Some(inner) = self.inner.upgrade() {
            let mut image_views = inner.image_views.lock();
            for idx in iter {
                let image_view = image_views.remove(idx);
                unsafe {
                    inner.device.destroy_image_view(image_view, None);
                }
            }
        }
    }

    #[inline]
    pub fn drop_library(&self, idx: usize) {
        if let Some(inner) = self.inner.upgrade() {
            let library = inner.libraries.lock().remove(idx);
            unsafe {
                inner.device.destroy_shader_module(library, None);
            }
        }
    }
}

pub(super) trait DeviceOwned {
    fn owner(&self) -> &WeakDevice;
}

#[derive(Clone)]
pub struct Device {
    inner: Arc<DeviceInner>,
}

impl PartialEq for Device {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for Device {}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Device({:p}@{:p})",
            self.inner.device.handle(),
            self.inner.instance.handle()
        )
    }
}

impl Device {
    pub(super) fn new(
        guard: Arc<InstanceGuard>,
        version: Version,
        instance: ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: ash::Device,
        families: Vec<u32>,
        features: Features,
        properties: ash::vk::PhysicalDeviceProperties,
        allocator: gpu_alloc::GpuAllocator<DeviceMemory>,
        // epochs: Vec<Arc<PendingEpochs>>,
        push_descriptor: ash::khr::push_descriptor::Device,
        surface: Option<ash::khr::surface::Instance>,
        #[cfg(target_os = "windows")] win32_surface: Option<ash::khr::win32_surface::Instance>,
        swapchain: Option<ash::khr::swapchain::Device>,
        swapchain_maintenance1: Option<ash::ext::swapchain_maintenance1::Device>,
        #[cfg(any(debug_assertions, feature = "debug"))] debug_utils: Option<
            ash::ext::debug_utils::Device,
        >,
    ) -> Self {
        Device {
            inner: Arc::new(DeviceInner {
                _guard: guard,
                device,
                instance,
                physical_device,
                version,
                families,
                features,
                properties,
                memory: Mutex::new(Slab::with_capacity(64)),
                error_state: ErrorState::new(),
                buffers: Mutex::new(Slab::with_capacity(1024)),
                images: Mutex::new(Slab::with_capacity(1024)),
                image_views: Mutex::new(Slab::with_capacity(1024)),
                samplers: Mutex::new(HashMap::with_capacity(64)),
                libraries: Mutex::new(Slab::with_capacity(64)),
                set_layouts: Mutex::new(HashMap::with_capacity(256)),
                pipeline_layouts: Mutex::new(HashMap::with_capacity(64)),
                pipelines: Mutex::new(Slab::with_capacity(128)),
                allocator: Mutex::new(allocator),
                push_descriptor,
                surface,
                #[cfg(target_os = "windows")]
                win32_surface,
                swapchain,
                swapchain_maintenance1,
                // epochs,
                #[cfg(any(debug_assertions, feature = "debug"))]
                debug_utils,
            }),
        }
    }

    #[inline]
    pub(super) fn inner(&self) -> &DeviceInner {
        &self.inner
    }

    #[inline]
    pub(super) fn ash(&self) -> &ash::Device {
        &self.inner.device
    }

    #[inline]
    pub(super) fn ash_instance(&self) -> &ash::Instance {
        &self.inner.instance
    }

    #[inline]
    pub(super) fn is(&self, weak: &WeakDevice) -> bool {
        Arc::as_ptr(&self.inner) == Weak::as_ptr(&weak.inner)
    }

    #[inline]
    pub(super) fn is_owner(&self, owned: &impl DeviceOwned) -> bool {
        self.is(owned.owner())
    }

    /// Sets the sticky out-of-device-memory flag.
    pub(super) fn set_oom(&self) {
        self.inner.error_state.set_oom();
    }

    /// Returns true if the OOM flag is set.
    pub(super) fn is_oom(&self) -> bool {
        self.inner.error_state.is_oom()
    }

    /// Sets the sticky device lost flag.
    pub(super) fn set_lost(&self) {
        self.inner.error_state.set_lost();
    }

    /// Returns true if the device lost flag is set.
    pub(super) fn is_lost(&self) -> bool {
        self.inner.error_state.is_lost()
    }

    /// Returns true if the device is in an error state (lost or out of memory).
    pub(super) fn get_error(&self) -> Result<(), DeviceError> {
        self.inner.error_state.get_error()
    }

    #[inline]
    pub(super) fn weak(&self) -> WeakDevice {
        WeakDevice {
            inner: Arc::downgrade(&self.inner),
        }
    }

    #[inline]
    pub fn push_descriptor(&self) -> &ash::khr::push_descriptor::Device {
        &self.inner.push_descriptor
    }

    #[inline]
    pub(super) fn surface(&self) -> &ash::khr::surface::Instance {
        self.inner.surface.as_ref().unwrap()
    }

    #[inline]
    pub(super) fn swapchain(&self) -> &ash::khr::swapchain::Device {
        self.inner.swapchain.as_ref().unwrap()
    }

    #[inline]
    pub(super) fn swapchain_maintenance1(
        &self,
    ) -> Option<&ash::ext::swapchain_maintenance1::Device> {
        self.inner.swapchain_maintenance1.as_ref()
    }

    #[inline]
    pub(super) fn physical_device(&self) -> vk::PhysicalDevice {
        self.inner.physical_device
    }

    #[inline]
    pub(super) fn queue_families(&self) -> &[u32] {
        &self.inner.families
    }

    #[inline]
    #[cfg(any(debug_assertions, feature = "debug"))]
    fn set_object_name<T: Handle>(&self, handle: T, name: &str) {
        if !name.is_empty() {
            if let Some(debug_utils) = &self.inner.debug_utils {
                let name_cstr = ffi::CString::new(name).unwrap();
                let _ = unsafe {
                    debug_utils.set_debug_utils_object_name(
                        &vk::DebugUtilsObjectNameInfoEXT::default()
                            .object_handle(handle)
                            .object_name(&name_cstr),
                    )
                };
            }
        }
    }

    #[inline]
    fn new_sampler_slow(&self, count: usize, desc: SamplerDesc) -> Sampler {
        if self.inner.properties.limits.max_sampler_allocation_count as usize <= count {
            self.set_oom();
            return Sampler::null();
        }

        let result = unsafe {
            self.ash().create_sampler(
                &ash::vk::SamplerCreateInfo::default()
                    .min_filter(desc.min_filter.into_ash())
                    .mag_filter(desc.mag_filter.into_ash())
                    .mipmap_mode(desc.mip_map_mode.into_ash())
                    .address_mode_u(desc.address_mode[0].into_ash())
                    .address_mode_v(desc.address_mode[1].into_ash())
                    .address_mode_w(desc.address_mode[2].into_ash())
                    .anisotropy_enable(desc.anisotropy.is_some())
                    .max_anisotropy(desc.anisotropy.unwrap_or(0.0))
                    .unnormalized_coordinates(!desc.normalized),
                None,
            )
        };

        let handle = match result {
            Ok(handle) => handle,
            Err(err) => match err {
                ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY => handle_host_oom(),
                ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
                    self.set_oom();
                    return Sampler::null();
                }
                _ => unexpected_error(err),
            },
        };

        Sampler::new(self.weak(), handle, desc)
    }

    fn new_set_layout_slow(
        &self,
        desc: DescriptorSetLayoutDesc,
    ) -> Result<DescriptorSetLayout, OutOfMemory> {
        let bindings = desc
            .arguments
            .iter()
            .enumerate()
            .map(|(idx, arg)| {
                ash::vk::DescriptorSetLayoutBinding::default()
                    .binding(u32::try_from(idx).expect("Too many descriptor bindings"))
                    .descriptor_count(
                        u32::try_from(arg.size).expect("Too many descriptors in array"),
                    )
                    .descriptor_type(descriptor_type(arg.kind))
                    .stage_flags(arg.stages.into_ash())
            })
            .collect::<Vec<_>>();

        let result = unsafe {
            self.ash().create_descriptor_set_layout(
                &ash::vk::DescriptorSetLayoutCreateInfo::default()
                    .flags(ash::vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
                    .bindings(&bindings),
                None,
            )
        };

        let handle = result.map_err(|err| match err {
            ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY => handle_host_oom(),
            ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
                self.set_oom();
                OutOfMemory
            }
            _ => unexpected_error(err),
        })?;
        Ok(DescriptorSetLayout::new(self.weak(), handle, desc))
    }

    fn new_set_layout(
        &self,
        desc: DescriptorSetLayoutDesc,
    ) -> Result<DescriptorSetLayout, OutOfMemory> {
        let mut set_layouts = self.inner.set_layouts.lock();

        match set_layouts.entry(desc) {
            hashbrown::hash_map::Entry::Occupied(mut entry) => match entry.get().upgrade() {
                Some(set_layout) => Ok(set_layout.clone()),
                None => {
                    let set_layout = self.new_set_layout_slow(entry.key().clone())?;
                    entry.insert(set_layout.downgrade());
                    Ok(set_layout)
                }
            },
            hashbrown::hash_map::Entry::Vacant(entry) => {
                let set_layout = self.new_set_layout_slow(entry.key().clone())?;
                entry.insert(set_layout.downgrade());
                Ok(set_layout)
            }
        }
    }

    fn new_pipeline_layout_slow(
        &self,
        desc: PipelineLayoutDesc,
    ) -> Result<PipelineLayout, OutOfMemory> {
        let set_layouts = desc
            .groups
            .iter()
            .map(|group| {
                self.new_set_layout(DescriptorSetLayoutDesc {
                    arguments: group.clone(),
                })
            })
            .collect::<Result<Vec<_>, OutOfMemory>>()?;

        let handles = set_layouts
            .iter()
            .map(|set_layout| set_layout.handle())
            .collect::<Vec<_>>();

        let mut info = ash::vk::PipelineLayoutCreateInfo::default().set_layouts(&handles);

        let push_constant_ranges;

        if desc.constants > 0 {
            push_constant_ranges = ash::vk::PushConstantRange::default()
                .stage_flags(ash::vk::ShaderStageFlags::ALL)
                .size((desc.constants as u32 + 3) & !3);

            info = info.push_constant_ranges(std::slice::from_ref(&push_constant_ranges));
        }

        let result = unsafe { self.ash().create_pipeline_layout(&info, None) };
        let handle = result.map_err(|err| match err {
            ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY => handle_host_oom(),
            ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
                self.set_oom();
                OutOfMemory
            }
            _ => unexpected_error(err),
        })?;
        Ok(PipelineLayout::new(self.weak(), handle, desc, set_layouts))
    }

    fn new_pipeline_layout(&self, desc: PipelineLayoutDesc) -> Result<PipelineLayout, OutOfMemory> {
        let mut pipeline_layouts = self.inner.pipeline_layouts.lock();

        match pipeline_layouts.entry(desc) {
            hashbrown::hash_map::Entry::Occupied(mut entry) => match entry.get().upgrade() {
                Some(pipeline_layout) => Ok(pipeline_layout.clone()),
                None => {
                    let pipeline_layout = self.new_pipeline_layout_slow(entry.key().clone())?;
                    entry.insert(pipeline_layout.downgrade());
                    Ok(pipeline_layout)
                }
            },
            hashbrown::hash_map::Entry::Vacant(entry) => {
                let pipeline_layout = self.new_pipeline_layout_slow(entry.key().clone())?;
                entry.insert(pipeline_layout.downgrade());
                Ok(pipeline_layout)
            }
        }
    }

    #[doc(hidden)]
    pub(super) fn get_descriptor_update_template<T: 'static>(
        &self,
        entries: &[ash::vk::DescriptorUpdateTemplateEntry],
        bind: ash::vk::PipelineBindPoint,
        layout: &PipelineLayout,
        set: u32,
    ) -> Result<ash::vk::DescriptorUpdateTemplate, OutOfMemory> {
        match layout
            .templates()
            .lock()
            .entry((TypeId::of::<T>(), bind, set))
        {
            hashbrown::hash_map::Entry::Occupied(entry) => Ok(*entry.get()),
            hashbrown::hash_map::Entry::Vacant(entry) => {
                let result = unsafe {
                    self.ash().create_descriptor_update_template(
                        &ash::vk::DescriptorUpdateTemplateCreateInfo::default()
                            .template_type(
                                ash::vk::DescriptorUpdateTemplateType::PUSH_DESCRIPTORS_KHR,
                            )
                            .pipeline_bind_point(bind)
                            .pipeline_layout(layout.handle())
                            .descriptor_update_entries(entries)
                            .set(set),
                        None,
                    )
                };

                let template = result.map_err(|err| match err {
                    ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY => handle_host_oom(),
                    ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
                        self.set_oom();
                        OutOfMemory
                    }
                    _ => unexpected_error(err),
                })?;

                entry.insert(template);
                Ok(template)
            }
        }
    }

    #[inline]
    #[cold]
    pub(super) fn new_image_view(
        &self,
        image: vk::Image,
        view_type: vk::ImageViewType,
        desc: ViewDesc,
    ) -> Result<(ash::vk::ImageView, usize), OutOfMemory> {
        let result = unsafe {
            self.inner.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(view_type)
                    .format(desc.format.try_into_ash().unwrap())
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(format_aspect(desc.format))
                            .base_mip_level(desc.base_level)
                            .level_count(desc.levels)
                            .base_array_layer(desc.base_layer)
                            .layer_count(desc.layers),
                    )
                    .components(desc.swizzle.into_ash()),
                None,
            )
        };

        let view = result.map_err(|err| match err {
            vk::Result::ERROR_OUT_OF_HOST_MEMORY => handle_host_oom(),
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
                self.set_oom();
                OutOfMemory
            }
            _ => unexpected_error(err),
        })?;

        let idx = self.inner.image_views.lock().insert(view);

        Ok((view, idx))
    }

    pub(super) fn new_fence(&self) -> Result<vk::Fence, OutOfMemory> {
        let result = unsafe {
            self.ash()
                .create_fence(&vk::FenceCreateInfo::default(), None)
        };

        result.map_err(|err| match err {
            vk::Result::ERROR_OUT_OF_HOST_MEMORY => handle_host_oom(),
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
                self.set_oom();
                OutOfMemory
            }
            _ => unexpected_error(err),
        })
    }

    pub(super) fn get_fence_status(&self, fence: vk::Fence) -> Result<bool, OutOfMemory> {
        match unsafe { self.ash().get_fence_status(fence) } {
            Ok(true) => Ok(true),
            Ok(false) => Ok(false),
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY) => handle_host_oom(),
            Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                self.set_oom();
                Err(OutOfMemory)
            }
            Err(err) => unexpected_error(err),
        }
    }

    pub(super) fn reset_fences(&self, fences: &[vk::Fence]) -> Result<(), OutOfMemory> {
        match unsafe { self.ash().reset_fences(fences) } {
            Ok(()) => Ok(()),
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY) => handle_host_oom(),
            Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                self.set_oom();
                Err(OutOfMemory)
            }
            Err(err) => unexpected_error(err),
        }
    }

    /// Wait for all operations on the device to complete.
    pub(super) fn wait_idle(&self) -> Result<(), DeviceError> {
        let result = unsafe { self.inner.device.device_wait_idle() };

        result.map_err(|err| match err {
            ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY => handle_host_oom(),
            ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
                self.set_oom();
                DeviceError::OutOfMemory
            }
            ash::vk::Result::ERROR_DEVICE_LOST => {
                self.set_lost();
                DeviceError::DeviceLost
            }
            _ => unexpected_error(err),
        })
    }
}

impl crate::traits::Resource for Device {}

#[hidden_trait::expose]
impl crate::traits::Device for Device {
    fn new_shader_library(&self, desc: LibraryDesc) -> Result<Library, ShaderLibraryError> {
        let me = &*self.inner;
        match desc.input {
            LibraryInput::Source(source) => {
                let compiled: Box<[u32]>;
                let code = match source.language {
                    ShaderLanguage::SpirV => unsafe {
                        let (left, words, right) = source.code.align_to::<u32>();

                        if left.is_empty() && right.is_empty() {
                            words
                        } else {
                            let (chunks, remainder) = source.code.as_chunks::<4>();
                            assert!(remainder.is_empty());

                            compiled = chunks
                                .iter()
                                .map(|c| u32::from_ne_bytes(*c))
                                .collect::<Vec<u32>>()
                                .into_boxed_slice();

                            &*compiled
                        }
                    },
                    _ => {
                        compiled = compile_shader(&source.code, source.filename, source.language)?;
                        &*compiled
                    }
                };
                let result = unsafe {
                    me.device.create_shader_module(
                        &vk::ShaderModuleCreateInfo::default().code(code),
                        None,
                    )
                };

                match result {
                    Ok(module) => {
                        let idx = self.inner.libraries.lock().insert(module);

                        #[cfg(any(debug_assertions, feature = "debug"))]
                        self.set_object_name(module, desc.name);

                        Ok(Library::new(self.weak(), module, idx))
                    }
                    Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY) => handle_host_oom(),
                    Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                        self.set_oom();
                        Ok(Library::null())
                    }
                    Err(result) => unexpected_error(result),
                }
            }
        }
    }

    /// Create a new render pipeline.
    fn new_compute_pipeline(
        &self,
        desc: ComputePipelineDesc,
    ) -> Result<ComputePipeline, PipelineError> {
        if self.inner.error_state.is_some() {
            return Ok(ComputePipeline::null());
        }

        let layout_desc = PipelineLayoutDesc {
            groups: desc
                .arguments
                .iter()
                .map(|group| group.arguments.to_vec())
                .collect(),
            constants: desc.constants,
        };

        let Ok(layout) = self.new_pipeline_layout(layout_desc) else {
            self.set_oom();
            return Ok(ComputePipeline::null());
        };

        let shader_name;

        let create_info = vk::ComputePipelineCreateInfo::default()
            .stage(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(desc.shader.library.module())
                    .name({
                        shader_name = ffi::CString::new(&*desc.shader.entry).unwrap();
                        &*shader_name
                    }),
            )
            .layout(layout.handle());

        let result = unsafe {
            self.ash().create_compute_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&create_info),
                None,
            )
        };

        let pipelines = match result {
            Ok(pipelines) => pipelines,
            Err((_, vk::Result::ERROR_OUT_OF_HOST_MEMORY)) => handle_host_oom(),
            Err((_, vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)) => {
                self.set_oom();
                return Ok(ComputePipeline::null());
            }
            Err((_, err)) => unexpected_error(err),
        };

        let pipeline = pipelines[0];

        #[cfg(any(debug_assertions, feature = "debug"))]
        self.set_object_name(pipeline, desc.name);

        let idx = self.inner.pipelines.lock().insert(pipeline);

        Ok(ComputePipeline::new(
            self.weak(),
            pipeline,
            idx,
            layout,
            desc.shader.library,
        ))
    }

    fn new_render_pipeline(
        &self,
        desc: RenderPipelineDesc,
    ) -> Result<RenderPipeline, PipelineError> {
        if self.inner.error_state.is_some() {
            return Ok(RenderPipeline::null());
        }

        let layout_desc = PipelineLayoutDesc {
            groups: desc
                .arguments
                .iter()
                .map(|group| group.arguments.to_vec())
                .collect(),
            constants: desc.constants,
        };

        let Ok(layout) = self.new_pipeline_layout(layout_desc) else {
            self.set_oom();
            return Ok(RenderPipeline::null());
        };

        let vertex_attributes = desc
            .vertex_attributes
            .iter()
            .enumerate()
            .map(|(idx, attr)| vk::VertexInputAttributeDescription {
                location: idx as u32,
                binding: attr.buffer_index,
                format: attr.format.try_into_ash().expect("Unsupported on Vulkan"),
                offset: attr.offset,
            })
            .collect::<Vec<_>>();

        let vertex_bindings = desc
            .vertex_layouts
            .iter()
            .enumerate()
            .map(|(idx, attr)| vk::VertexInputBindingDescription {
                binding: idx as u32,
                stride: attr.stride,
                input_rate: match attr.step_mode {
                    VertexStepMode::Vertex => vk::VertexInputRate::VERTEX,
                    VertexStepMode::Instance { rate: 1 } => vk::VertexInputRate::INSTANCE,
                    VertexStepMode::Instance { rate } => {
                        panic!(
                            "Instance vertex step mode with rate {rate} is not supported on Vulkan"
                        )
                    }
                    VertexStepMode::Constant => {
                        panic!("Constant vertex step mode is not supported on Vulkan")
                    }
                },
            })
            .collect::<Vec<_>>();

        let vertex_shader_name;
        let fragment_shader_name;

        let mut stages = vec![
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(desc.vertex_shader.library.module())
                .name({
                    vertex_shader_name = ffi::CString::new(&*desc.vertex_shader.entry).unwrap();
                    &*vertex_shader_name
                }),
        ];

        let mut raster_state = vk::PipelineRasterizationStateCreateInfo::default();
        let mut depth_state = vk::PipelineDepthStencilStateCreateInfo::default();
        let mut attachments = Vec::new();
        let mut color_attachment_formats = Vec::new();
        let mut rendering = vk::PipelineRenderingCreateInfo::default();

        let vertex_library = desc.vertex_shader.library;
        let mut fragment_library = None;

        if let Some(raster) = desc.raster {
            if let Some(fragment_shader) = raster.fragment_shader {
                stages.push(
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::FRAGMENT)
                        .module(fragment_shader.library.module())
                        .name({
                            fragment_shader_name =
                                ffi::CString::new(&*fragment_shader.entry).unwrap();
                            &*fragment_shader_name
                        }),
                );

                fragment_library = Some(fragment_shader.library);
            }

            raster_state = raster_state
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(raster.culling.into_ash())
                .front_face(raster.front_face.into_ash())
                .line_width(1.0);

            if let Some(depth) = &raster.depth_stencil {
                depth_state = depth_state
                    .depth_test_enable(depth.format.is_depth())
                    .depth_compare_op(depth.compare.into_ash())
                    .depth_write_enable(depth.write_enabled)
                    .stencil_test_enable(depth.format.is_stencil());

                if depth.format.is_depth() {
                    rendering.depth_attachment_format = depth.format.try_into_ash().unwrap();
                }
                if depth.format.is_stencil() {
                    rendering.stencil_attachment_format = depth.format.try_into_ash().unwrap();
                }
            }

            for color in &raster.color_targets {
                let mut blend_state = vk::PipelineColorBlendAttachmentState::default();
                if let Some(blend) = color.blend {
                    blend_state = blend_state
                        .blend_enable(true)
                        .src_color_blend_factor(blend.color.src.into_ash())
                        .dst_color_blend_factor(blend.color.dst.into_ash())
                        .color_blend_op(blend.color.op.into_ash())
                        .src_alpha_blend_factor(blend.alpha.src.into_ash())
                        .dst_alpha_blend_factor(blend.alpha.dst.into_ash())
                        .alpha_blend_op(blend.alpha.op.into_ash())
                        .color_write_mask(blend.mask.into_ash());
                }
                attachments.push(blend_state);
                color_attachment_formats.push(color.format.try_into_ash().unwrap());
            }
        } else {
            raster_state = raster_state.rasterizer_discard_enable(true);
        }

        rendering = rendering
            .view_mask(0)
            .color_attachment_formats(&color_attachment_formats);
        let create_info = vk::GraphicsPipelineCreateInfo::default().push_next(&mut rendering);

        let result = unsafe {
            self.inner.device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(
                    &create_info
                        .stages(&stages)
                        .vertex_input_state(
                            &vk::PipelineVertexInputStateCreateInfo::default()
                                .vertex_attribute_descriptions(&vertex_attributes)
                                .vertex_binding_descriptions(&vertex_bindings),
                        )
                        .input_assembly_state(
                            &vk::PipelineInputAssemblyStateCreateInfo::default().topology(
                                match desc.primitive_topology {
                                    PrimitiveTopology::Point => vk::PrimitiveTopology::POINT_LIST,
                                    PrimitiveTopology::Line => vk::PrimitiveTopology::LINE_LIST,
                                    PrimitiveTopology::Triangle => {
                                        vk::PrimitiveTopology::TRIANGLE_LIST
                                    }
                                },
                            ),
                        )
                        .rasterization_state(&raster_state)
                        .multisample_state(
                            &vk::PipelineMultisampleStateCreateInfo::default()
                                .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                        )
                        .depth_stencil_state(&depth_state)
                        .color_blend_state(
                            &vk::PipelineColorBlendStateCreateInfo::default()
                                .attachments(&attachments)
                                .blend_constants([1.0; 4]),
                        )
                        .viewport_state(
                            &ash::vk::PipelineViewportStateCreateInfo::default()
                                .scissors(&[vk::Rect2D {
                                    offset: vk::Offset2D { x: 0, y: 0 },
                                    extent: vk::Extent2D {
                                        width: 0,
                                        height: 0,
                                    },
                                }])
                                .viewports(&[vk::Viewport {
                                    x: 0.0,
                                    y: 0.0,
                                    width: 0.0,
                                    height: 0.0,
                                    min_depth: 0.0,
                                    max_depth: 1.0,
                                }]),
                        )
                        .dynamic_state(
                            &vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&[
                                vk::DynamicState::VIEWPORT,
                                vk::DynamicState::SCISSOR,
                            ]),
                        )
                        .layout(layout.handle()),
                ),
                None,
            )
        };

        let pipelines = match result {
            Ok(pipelines) => pipelines,
            Err((_, vk::Result::ERROR_OUT_OF_HOST_MEMORY)) => handle_host_oom(),
            Err((_, vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)) => {
                self.set_oom();
                return Ok(RenderPipeline::null());
            }
            Err((_, err)) => unexpected_error(err),
        };
        let pipeline = pipelines[0];

        #[cfg(any(debug_assertions, feature = "debug"))]
        self.set_object_name(pipeline, desc.name);

        let idx = self.inner.pipelines.lock().insert(pipeline);

        Ok(RenderPipeline::new(
            self.weak(),
            pipeline,
            idx,
            layout,
            vertex_library,
            fragment_library,
        ))
    }

    fn new_buffer(&self, desc: BufferDesc) -> Buffer {
        if self.inner.error_state.is_some() {
            return Buffer::null(desc.size, desc.usage);
        }

        let Ok(size) = u64::try_from(desc.size) else {
            self.set_oom();
            return Buffer::null(desc.size, desc.usage);
        };

        let result = unsafe {
            self.ash().create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .usage(desc.usage.into_ash()),
                None,
            )
        };

        let buffer = match result {
            Ok(buffer) => buffer,
            Err(err) => match err {
                vk::Result::ERROR_OUT_OF_HOST_MEMORY => handle_host_oom(),
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
                    self.set_oom();
                    return Buffer::null(desc.size, desc.usage);
                }
                _ => unexpected_error(err),
            },
        };

        let requirements = unsafe { self.ash().get_buffer_memory_requirements(buffer) };
        let align_mask = requirements.alignment - 1;

        let result = unsafe {
            self.inner.allocator.lock().alloc(
                &*self.inner,
                gpu_alloc::Request {
                    size: requirements.size,
                    align_mask,
                    usage: map_usage_flags(desc.usage),
                    memory_types: requirements.memory_type_bits,
                },
            )
        };

        let block = match result {
            Ok(block) => block,
            Err(gpu_alloc::AllocationError::OutOfHostMemory) => handle_host_oom(),
            _ => {
                self.set_oom();
                unsafe {
                    self.ash().destroy_buffer(buffer, None);
                }
                return Buffer::null(desc.size, desc.usage);
            }
        };

        let result = unsafe {
            self.ash()
                .bind_buffer_memory(buffer, block.memory().handle, block.offset())
        };

        match result {
            Ok(()) => {
                #[cfg(any(debug_assertions, feature = "debug"))]
                self.set_object_name(buffer, desc.name);

                let idx = self.inner.buffers.lock().insert(buffer);

                Buffer::new(self.weak(), buffer, desc.size, desc.usage, block, idx)
            }
            Err(err) => {
                unsafe {
                    self.inner.allocator.lock().dealloc(&*self.inner, block);

                    self.ash().destroy_buffer(buffer, None);
                }

                match err {
                    vk::Result::ERROR_OUT_OF_HOST_MEMORY => handle_host_oom(),
                    vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
                        self.set_oom();
                        Buffer::null(desc.size, desc.usage)
                    }
                    _ => unexpected_error(err),
                }
            }
        }
    }

    fn new_buffer_init(&self, desc: BufferInitDesc<'_>) -> Buffer {
        if self.inner.error_state.is_some() {
            return Buffer::null(desc.data.len(), desc.usage);
        }

        let mut buffer = self.new_buffer(BufferDesc {
            size: desc.data.len(),
            usage: desc.usage | BufferUsage::HOST_WRITE,
            name: desc.name,
        });

        let _ = buffer.write(0, desc.data);

        buffer
    }

    fn new_image(&self, desc: ImageDesc) -> Image {
        if self.inner.error_state.is_some() {
            return Image::null(
                desc.extent,
                desc.format,
                desc.usage,
                desc.layers,
                desc.levels,
            );
        }

        let result = unsafe {
            self.inner.device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(desc.extent.into_ash())
                    .format(desc.format.try_into_ash().expect("Unsupported format"))
                    .extent(desc.extent.into_ash())
                    .array_layers(desc.layers)
                    .mip_levels(desc.levels)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage((desc.usage, desc.format).into_ash())
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
        };

        let image = match result {
            Ok(image) => image,
            Err(err) => match err {
                vk::Result::ERROR_OUT_OF_HOST_MEMORY => handle_host_oom(),
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
                    self.set_oom();
                    return Image::null(
                        desc.extent,
                        desc.format,
                        desc.usage,
                        desc.layers,
                        desc.levels,
                    );
                }
                _ => unexpected_error(err),
            },
        };

        let requirements = unsafe { self.inner.device.get_image_memory_requirements(image) };
        let align_mask = requirements.alignment - 1;

        let result = unsafe {
            self.inner.allocator.lock().alloc(
                &*self.inner,
                gpu_alloc::Request {
                    size: requirements.size,
                    align_mask,
                    usage: gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
                    memory_types: requirements.memory_type_bits,
                },
            )
        };

        let block = match result {
            Ok(block) => block,
            Err(err) => {
                unsafe {
                    self.inner.device.destroy_image(image, None);
                }
                match err {
                    gpu_alloc::AllocationError::OutOfHostMemory => handle_host_oom(),
                    _ => {
                        self.set_oom();
                        return Image::null(
                            desc.extent,
                            desc.format,
                            desc.usage,
                            desc.layers,
                            desc.levels,
                        );
                    }
                }
            }
        };

        let result = unsafe {
            self.inner
                .device
                .bind_image_memory(image, block.memory().handle, block.offset())
        };

        if let Err(err) = result {
            unsafe {
                self.inner.device.destroy_image(image, None);
                self.inner.allocator.lock().dealloc(&*self.inner, block);
            }

            match err {
                vk::Result::ERROR_OUT_OF_HOST_MEMORY => handle_host_oom(),
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
                    self.set_oom();
                    return Image::null(
                        desc.extent,
                        desc.format,
                        desc.usage,
                        desc.layers,
                        desc.levels,
                    );
                }
                _ => unexpected_error(err),
            }
        }

        let result = self.new_image_view(
            image,
            desc.extent.into_ash(),
            ViewDesc {
                format: desc.format,
                base_layer: 0,
                layers: desc.layers,
                base_level: 0,
                levels: desc.levels,
                swizzle: Swizzle::IDENTITY,
            },
        );

        let (view, view_idx) = match result {
            Ok((view, idx)) => (view, idx),
            Err(OutOfMemory) => {
                unsafe {
                    self.inner.device.destroy_image(image, None);
                    self.inner.allocator.lock().dealloc(&*self.inner, block);
                }

                self.set_oom();
                return Image::null(
                    desc.extent,
                    desc.format,
                    desc.usage,
                    desc.layers,
                    desc.levels,
                );
            }
        };

        #[cfg(any(debug_assertions, feature = "debug"))]
        self.set_object_name(image, desc.name);

        let idx = self.inner.images.lock().insert(image);

        Image::new(
            self.weak(),
            image,
            view,
            view_idx,
            desc.extent,
            desc.format,
            desc.usage,
            desc.layers,
            desc.levels,
            block,
            idx,
        )
    }

    fn new_sampler(&self, desc: SamplerDesc) -> Sampler {
        if self.inner.error_state.is_some() {
            return Sampler::null();
        }

        let mut samplers = self.inner.samplers.lock();
        let len = samplers.len();
        match samplers.entry(desc) {
            hashbrown::hash_map::Entry::Occupied(mut entry) => match entry.get().upgrade() {
                Some(sampler) => sampler,
                None => {
                    let sampler = self.new_sampler_slow(len, desc);
                    entry.insert(sampler.downgrade());
                    sampler
                }
            },
            hashbrown::hash_map::Entry::Vacant(entry) => {
                let sampler = self.new_sampler_slow(len, desc);
                entry.insert(sampler.downgrade());
                sampler
            }
        }
    }

    fn new_surface(
        &self,
        window: &impl HasWindowHandle,
        display: &impl HasDisplayHandle,
    ) -> Result<Surface, SurfaceError> {
        let me = &*self.inner;

        me.error_state.get_error()?;

        let me = &*self.inner;
        assert!(
            me.features.contains(Features::SURFACE),
            "Surface feature is not enabled"
        );

        let window = window
            .window_handle()
            .map_err(|_| SurfaceError::SurfaceLost)?;
        let display = display
            .display_handle()
            .map_err(|_| SurfaceError::SurfaceLost)?;

        match (window.as_raw(), display.as_raw()) {
            #[cfg(target_os = "windows")]
            (RawWindowHandle::Win32(window), RawDisplayHandle::Windows(_)) => {
                let win32_surface = me.win32_surface.as_ref().unwrap();
                let result = unsafe {
                    win32_surface.create_win32_surface(
                        &ash::vk::Win32SurfaceCreateInfoKHR::default()
                            // .hinstance(hinstance)
                            .hwnd(window.hwnd.get() as _),
                        None,
                    )
                };
                let surface = match result {
                    Ok(surface) => surface,
                    Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY) => handle_host_oom(),
                    Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                        self.set_oom();
                        return Err(SurfaceError::OutOfMemory);
                    }
                    Err(err) => unexpected_error(err),
                };

                let result = unsafe {
                    self.surface()
                        .get_physical_device_surface_formats(self.physical_device(), surface)
                };
                let formats = match result {
                    Ok(formats) => formats,
                    Err(ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY) => handle_host_oom(),
                    Err(ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                        self.set_oom();
                        return Err(SurfaceError::OutOfMemory);
                    }
                    Err(ash::vk::Result::ERROR_SURFACE_LOST_KHR) => {
                        return Err(SurfaceError::SurfaceLost);
                    }
                    Err(err) => unexpected_error(err),
                };

                let result = unsafe {
                    self.surface()
                        .get_physical_device_surface_present_modes(self.physical_device(), surface)
                };
                let modes = match result {
                    Ok(modes) => modes,
                    Err(ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY) => handle_host_oom(),
                    Err(ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                        self.set_oom();
                        return Err(SurfaceError::OutOfMemory);
                    }
                    Err(ash::vk::Result::ERROR_SURFACE_LOST_KHR) => {
                        return Err(SurfaceError::SurfaceLost);
                    }
                    Err(err) => unexpected_error(err),
                };

                let family_supports =
                    self.queue_families()
                        .iter()
                        .try_fold(Vec::new(), |mut supports, &idx| {
                            let result = unsafe {
                                self.surface().get_physical_device_surface_support(
                                    self.physical_device(),
                                    idx,
                                    surface,
                                )
                            };

                            let support = match result {
                                Ok(support) => support,
                                Err(ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY) => handle_host_oom(),
                                Err(ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                                    self.set_oom();
                                    return Err(SurfaceError::OutOfMemory);
                                }
                                Err(ash::vk::Result::ERROR_SURFACE_LOST_KHR) => {
                                    return Err(SurfaceError::SurfaceLost);
                                }
                                Err(err) => unexpected_error(err),
                            };

                            supports.push(support);
                            Ok::<_, SurfaceError>(supports)
                        })?;

                Ok(Surface::new(
                    self.clone(),
                    surface,
                    formats,
                    modes,
                    family_supports,
                ))
            }
            (RawWindowHandle::Win32(_), _) => {
                panic!("Mismatched window and display type")
            }
            _ => {
                unreachable!("Unsupported window type for this platform")
            }
        }
    }

    fn new_fake_surface(&self, image: Image) -> Result<Surface, SurfaceError> {
        self.inner.error_state.get_error()?;

        let semaphore = new_semaphore(self.ash())?;
        Ok(Surface::fake(self.clone(), image, semaphore))
    }

    /// Create a new bottom-level acceleration structure.
    fn new_blas(&self, desc: BlasDesc) -> Blas {
        todo!()
    }

    /// Create a new top-level acceleration structure.
    fn new_tlas(&self, desc: TlasDesc) -> Tlas {
        todo!()
    }
}

fn map_usage_flags(usage: BufferUsage) -> gpu_alloc::UsageFlags {
    let mut flags = gpu_alloc::UsageFlags::empty();

    if usage.intersects(BufferUsage::HOST_READ | BufferUsage::HOST_WRITE) {
        flags |= gpu_alloc::UsageFlags::HOST_ACCESS;
    }

    if usage.contains(BufferUsage::HOST_READ) {
        flags |= gpu_alloc::UsageFlags::DOWNLOAD;
    }

    if usage.contains(BufferUsage::HOST_WRITE) {
        flags |= gpu_alloc::UsageFlags::UPLOAD;
    }

    if usage.contains(BufferUsage::TRANSIENT) {
        flags |= gpu_alloc::UsageFlags::TRANSIENT;
    }

    flags
}

pub(crate) fn compile_shader(
    code: &[u8],
    filename: Option<&str>,
    lang: ShaderLanguage,
) -> Result<Box<[u32]>, ShaderLibraryError> {
    let (module, info, source_code) = parse_shader(code, filename, lang)?;

    let options = naga::back::spv::Options {
        lang_version: (1, 3),
        flags: naga::back::spv::WriterFlags::ADJUST_COORDINATE_SPACE
            | naga::back::spv::WriterFlags::LABEL_VARYINGS
            | naga::back::spv::WriterFlags::CLAMP_FRAG_DEPTH
            | if cfg!(debug_assertions) {
                naga::back::spv::WriterFlags::DEBUG
            } else {
                naga::back::spv::WriterFlags::empty()
            },
        fake_missing_bindings: true,
        binding_map: naga::back::spv::BindingMap::default(),
        capabilities: None,
        bounds_check_policies: naga::proc::BoundsCheckPolicies::default(),
        zero_initialize_workgroup_memory: naga::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
        force_loop_bounding: true,
        ray_query_initialization_tracking: true,
        use_storage_input_output_16: true,
        debug_info: match source_code {
            None => None,
            Some(source_code) => Some(naga::back::spv::DebugInfo {
                source_code,
                file_name: filename.unwrap_or("<nofile>").as_ref(),
                language: match lang {
                    ShaderLanguage::Wgsl => naga::back::spv::SourceLanguage::WGSL,
                    ShaderLanguage::Glsl { .. } => naga::back::spv::SourceLanguage::GLSL,
                    ShaderLanguage::Msl { .. } => naga::back::spv::SourceLanguage::Unknown,
                    ShaderLanguage::SpirV => naga::back::spv::SourceLanguage::Unknown,
                },
            }),
        },
        task_dispatch_limits: None,
        mesh_shader_primitive_indices_clamp: false,
    };

    let words = naga::back::spv::write_vec(&module, &info, &options, None)
        .map(|vec| vec.into())
        .map_err(ShaderLibraryError::GenSpirV)?;

    Ok(words)
}
