use std::{collections::VecDeque, ops::Deref};

use ash::vk::{self, Handle};
use bitflags::Flags;
use smallvec::SmallVec;

use crate::{
    ColorSpace, DeviceError, ImageDesc, ImageUsage, PixelFormat, PresentMode,
    backend::{
        from::{AshFrom, FromAsh, IntoAsh, TryFromAsh, TryIntoAsh},
        new_semaphore, unknown_error,
    },
    generic::{Extent2, Image2D, ImageExtent, ImageN, SurfaceError, Swizzle, ViewDesc},
};

use super::{
    Device, Image,
    from::{AshInto, TryAshInto},
    handle_host_oom, unexpected_error,
};

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
    struct SurfaceStatus: u32 {
        const LOST = 0x01;
        const OUT_OF_DATE = 0x02;
        const SUBOPTIMAL = 0x04;

        const NEEDS_RECONFIGURE = 0x06; // OUT_OF_DATE | SUBOPTIMAL
    }
}

const RECONFIGURE_COOLDOWN: u64 = 10;

struct SwapchainFences {
    array: SmallVec<[vk::Fence; 4]>,
    next: usize,
}

struct Swapchain {
    handle: vk::SwapchainKHR,
    extent: vk::Extent2D,
    format: vk::SurfaceFormatKHR,
    mode: vk::PresentModeKHR,
    usage: vk::ImageUsageFlags,
    images: SmallVec<[(Image, [vk::Semaphore; 2]); 4]>,
    next: vk::Semaphore,
    fences: Option<SwapchainFences>,
}

struct FakeSwapchain {
    image: Image,
    semaphore: vk::Semaphore,
    frame_idx: u64,
}

enum MaybeFakeSwapchain {
    Real(Swapchain),
    Fake(FakeSwapchain),
}

pub struct Surface {
    device: Device,
    surface: vk::SurfaceKHR,
    current: Option<MaybeFakeSwapchain>,
    retired: VecDeque<MaybeFakeSwapchain>,
    caps: vk::SurfaceCapabilitiesKHR,
    ash_formats: Vec<vk::SurfaceFormatKHR>,
    mev_formats: Vec<(PixelFormat, ColorSpace)>,
    ash_modes: Vec<vk::PresentModeKHR>,
    mev_modes: Vec<PresentMode>,
    family_supports: Vec<bool>,

    preferred_extent: Option<Extent2>,
    preferred_format: Option<PixelFormat>,
    preferred_color_space: Option<ColorSpace>,
    preferred_mode: Option<PresentMode>,
    preferred_usage: Option<ImageUsage>,
    preferred_image_count: Option<u32>,

    bound_queue_family: Option<u32>,

    cooldown: u64,
    status: SurfaceStatus,
}

impl Drop for Surface {
    fn drop(&mut self) {
        self.force_clear_retired().unwrap();

        let device = self.device.ash();

        match self.current.take() {
            None => {}
            Some(MaybeFakeSwapchain::Real(mut swapchain)) => {
                let can_destroy = swapchain
                    .images
                    .iter_mut()
                    .all(|(image, _)| image.detached());
                assert!(can_destroy);

                for (_, [acquire, present]) in swapchain.images {
                    unsafe {
                        device.destroy_semaphore(acquire, None);
                        device.destroy_semaphore(present, None);
                    }
                }

                unsafe {
                    device.destroy_semaphore(swapchain.next, None);
                }

                if let Some(fences) = swapchain.fences {
                    for fence in fences.array {
                        unsafe {
                            device.destroy_fence(fence, None);
                        }
                    }
                }

                unsafe {
                    self.device
                        .swapchain()
                        .destroy_swapchain(swapchain.handle, None);
                }
            }
            Some(MaybeFakeSwapchain::Fake(fake)) => unsafe {
                device.destroy_semaphore(fake.semaphore, None);
            },
        }

        unsafe {
            self.device.surface().destroy_surface(self.surface, None);
        }
    }
}

impl Surface {
    pub(super) fn new(device: Device, surface: vk::SurfaceKHR) -> Self {
        Surface {
            device,
            surface,
            current: None,
            retired: VecDeque::new(),
            caps: vk::SurfaceCapabilitiesKHR::default(),
            ash_formats: Vec::new(),
            mev_formats: Vec::new(),
            ash_modes: Vec::new(),
            mev_modes: Vec::new(),
            family_supports: Vec::new(),

            preferred_extent: None,
            preferred_format: None,
            preferred_color_space: None,
            preferred_mode: None,
            preferred_usage: None,
            preferred_image_count: None,

            bound_queue_family: None,

            cooldown: 0,
            status: SurfaceStatus::empty(),
        }
    }

    pub(super) fn fake(device: Device, image: Image, semaphore: vk::Semaphore) -> Self {
        let surface_format = vk::SurfaceFormatKHR {
            format: image
                .format()
                .try_into_ash()
                .expect("Existing image format should be convertible to ash format"),
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };

        Surface {
            device,
            surface: vk::SurfaceKHR::null(),
            retired: VecDeque::new(),
            caps: vk::SurfaceCapabilitiesKHR::default(),
            ash_formats: vec![surface_format],
            mev_formats: vec![(image.format(), ColorSpace::SrgbNonLinear)],
            ash_modes: Vec::new(),
            mev_modes: Vec::new(),
            family_supports: Vec::new(),

            preferred_extent: None,
            preferred_format: None,
            preferred_color_space: None,
            preferred_mode: None,
            preferred_usage: None,
            preferred_image_count: None,

            bound_queue_family: None,

            cooldown: 0,

            status: SurfaceStatus::empty(),

            current: Some(MaybeFakeSwapchain::Fake(FakeSwapchain {
                image,
                semaphore,
                frame_idx: 0,
            })),
        }
    }

    fn update_surface_parameters(&mut self) -> Result<(), SurfaceError> {
        if self.surface.is_null() {
            // Fake surface does not need to update parameters.
            return Ok(());
        }

        if self.status.contains(SurfaceStatus::LOST) {
            return Err(SurfaceError::SurfaceLost);
        }

        self.family_supports.clear();

        for &fidx in self.device.queue_families() {
            let mut supported = 0;

            let result = unsafe {
                (self
                    .device
                    .surface()
                    .fp()
                    .get_physical_device_surface_support_khr)(
                    self.device.physical_device(),
                    fidx,
                    self.surface,
                    &mut supported,
                )
            };

            match result {
                vk::Result::SUCCESS => {
                    if self.family_supports.len() <= fidx as usize {
                        self.family_supports.resize(fidx as usize + 1, false);
                    }
                    self.family_supports[fidx as usize] = supported != 0;
                }
                vk::Result::ERROR_OUT_OF_HOST_MEMORY => handle_host_oom(),
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
                    self.device.set_oom();
                    return Err(SurfaceError::OutOfMemory);
                }
                vk::Result::ERROR_SURFACE_LOST_KHR => {
                    self.status.insert(SurfaceStatus::LOST);
                    return Err(SurfaceError::SurfaceLost);
                }
                vk::Result::ERROR_VALIDATION_FAILED_EXT => {
                    unimplemented!(
                        "Validation failed when querying surface formats. This should not happen."
                    )
                }
                vk::Result::ERROR_UNKNOWN => unknown_error(),
                err => unexpected_error(err),
            }
        }

        loop {
            let mut count = self.ash_formats.capacity() as u32;

            let result = unsafe {
                (self
                    .device
                    .surface()
                    .fp()
                    .get_physical_device_surface_formats_khr)(
                    self.device.physical_device(),
                    self.surface,
                    &mut count,
                    self.ash_formats.as_mut_ptr(),
                )
            };

            match result {
                vk::Result::SUCCESS => {
                    unsafe {
                        self.ash_formats.set_len(count as usize);
                    }
                    break;
                }
                vk::Result::INCOMPLETE => {
                    unsafe {
                        (self
                            .device
                            .surface()
                            .fp()
                            .get_physical_device_surface_formats_khr)(
                            self.device.physical_device(),
                            self.surface,
                            &mut count,
                            std::ptr::null_mut(),
                        )
                    };

                    self.ash_formats
                        .resize(count as usize, vk::SurfaceFormatKHR::default());
                    continue;
                }
                vk::Result::ERROR_OUT_OF_HOST_MEMORY => handle_host_oom(),
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
                    self.device.set_oom();
                    return Err(SurfaceError::OutOfMemory);
                }
                vk::Result::ERROR_SURFACE_LOST_KHR => {
                    self.status.insert(SurfaceStatus::LOST);
                    return Err(SurfaceError::SurfaceLost);
                }
                vk::Result::ERROR_VALIDATION_FAILED_EXT => {
                    unimplemented!(
                        "Validation failed when querying surface formats. This should not happen."
                    )
                }
                vk::Result::ERROR_UNKNOWN => unknown_error(),
                err => unexpected_error(err),
            }
        }

        dedup_vec(&mut self.ash_formats);

        self.mev_formats.clear();
        self.mev_formats
            .extend(self.ash_formats.iter().filter_map(|sf| {
                let f = sf.format.try_ash_into()?;
                let cs = sf.color_space.try_ash_into()?;
                Some((f, cs))
            }));

        loop {
            let mut count = self.ash_modes.capacity() as u32;

            let result = unsafe {
                (self
                    .device
                    .surface()
                    .fp()
                    .get_physical_device_surface_present_modes_khr)(
                    self.device.physical_device(),
                    self.surface,
                    &mut count,
                    self.ash_modes.as_mut_ptr(),
                )
            };

            match result {
                vk::Result::SUCCESS => {
                    unsafe {
                        self.ash_modes.set_len(count as usize);
                    }
                    break;
                }
                vk::Result::INCOMPLETE => {
                    unsafe {
                        (self
                            .device
                            .surface()
                            .fp()
                            .get_physical_device_surface_present_modes_khr)(
                            self.device.physical_device(),
                            self.surface,
                            &mut count,
                            std::ptr::null_mut(),
                        )
                    };

                    self.ash_modes
                        .resize(count as usize, vk::PresentModeKHR::default());
                    continue;
                }
                vk::Result::ERROR_OUT_OF_HOST_MEMORY => handle_host_oom(),
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
                    self.device.set_oom();
                    return Err(SurfaceError::OutOfMemory);
                }
                vk::Result::ERROR_SURFACE_LOST_KHR => {
                    self.status.insert(SurfaceStatus::LOST);
                    return Err(SurfaceError::SurfaceLost);
                }
                vk::Result::ERROR_VALIDATION_FAILED_EXT => {
                    unimplemented!(
                        "Validation failed when querying surface formats. This should not happen."
                    )
                }
                vk::Result::ERROR_UNKNOWN => unknown_error(),
                err => unexpected_error(err),
            }
        }

        let result = unsafe {
            self.device
                .surface()
                .get_physical_device_surface_capabilities(
                    self.device.physical_device(),
                    self.surface,
                )
        };

        self.caps = match result {
            Ok(caps) => caps,
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY) => handle_host_oom(),
            Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                self.device.set_oom();
                return Err(SurfaceError::OutOfMemory);
            }
            Err(vk::Result::ERROR_SURFACE_LOST_KHR) => {
                self.status.insert(SurfaceStatus::LOST);
                self.device.set_lost();
                return Err(SurfaceError::SurfaceLost);
            }
            Err(err) => unexpected_error(err),
        };

        tracing::debug!(
            "Surface '{:?}' capabilities: '{:?}'",
            self.surface,
            self.caps
        );

        Ok(())
    }

    fn preprocess(&mut self) -> Result<(), SurfaceError> {
        self.cooldown = self.cooldown.saturating_sub(1);
        if self.cooldown > 0 {
            return Ok(());
        }

        self.update_surface_parameters()?;
        self.configure()?;
        Ok(())
    }

    // Initialize the swapchain.
    // Retires any old swapchain.
    fn configure(&mut self) -> Result<(), SurfaceError> {
        if self.surface.is_null() {
            // Fake swapchain does not need to be reconfigured.
            return Ok(());
        }

        tracing::debug!("Surface '{:?}' configure", self.surface);

        self.handle_retired()?;

        if self.status.contains(SurfaceStatus::LOST) {
            return Err(SurfaceError::SurfaceLost);
        }

        let use_extent = pick_extent(&self.caps, self.preferred_extent);
        let use_mode = pick_mode(&self.ash_modes, self.preferred_mode);
        let use_format = pick_format(
            &self.ash_formats,
            self.preferred_format,
            self.preferred_color_space,
        );
        let use_mev_format = PixelFormat::try_from_ash(use_format.format).unwrap();
        let use_usage = pick_usage(&self.caps, self.preferred_usage, use_mev_format);
        let use_mev_usage = ImageUsage::from_ash(use_usage);

        if !self.status.intersects(SurfaceStatus::NEEDS_RECONFIGURE) {
            if let Some(MaybeFakeSwapchain::Real(swapchain)) = &self.current {
                if swapchain.extent == use_extent
                    && swapchain.format == use_format
                    && swapchain.mode == use_mode
                    && swapchain.usage == use_usage
                {
                    return Ok(());
                }
            }
        }

        if use_extent.width == 0 || use_extent.height == 0 {
            if let Some(MaybeFakeSwapchain::Fake(swapchain)) = &self.current {
                // Faked swapchain on real surface is always suboptimal.
                self.cooldown = RECONFIGURE_COOLDOWN;
                return Ok(());
            }

            match self.current.take() {
                None => {}
                Some(MaybeFakeSwapchain::Real(swapchain)) => {
                    self.retired.push_back(MaybeFakeSwapchain::Real(swapchain));
                }
                _ => unreachable!(),
            }

            let image = self.device.new_image(ImageDesc {
                dims: ImageExtent::D2(Extent2::new(1, 1)),
                format: use_format.format.try_ash_into().unwrap(),
                usage: self.preferred_usage.unwrap_or(ImageUsage::empty()),
                levels: 1,
                name: "fake-swapchain-image",
            });

            let semaphore = new_semaphore(self.device.ash())?;

            self.current = Some(MaybeFakeSwapchain::Fake(FakeSwapchain {
                image,
                semaphore,
                frame_idx: 0,
            }));

            self.status.remove(SurfaceStatus::NEEDS_RECONFIGURE);
            self.cooldown = RECONFIGURE_COOLDOWN;

            return Ok(());
        }

        let use_image_count = match (
            self.caps.min_image_count,
            self.caps.max_image_count,
            self.preferred_image_count,
        ) {
            (min, 0, Some(preferred)) => preferred.max(min),
            (min, max, Some(preferred)) => preferred.clamp(min, max),
            (min, _, None) => min,
        };

        let old = self.current.take();

        let result = unsafe {
            let mut info = vk::SwapchainCreateInfoKHR::default();
            let mut present_modes_info = vk::SwapchainPresentModesCreateInfoEXT::default();
            let modes = [use_mode];

            if self.device.swapchain_maintenance1().is_some() {
                present_modes_info = present_modes_info.present_modes(&modes);
                info = info.push_next(&mut present_modes_info)
            }

            info = info
                .surface(self.surface)
                .min_image_count(use_image_count)
                .image_format(use_format.format)
                .image_color_space(use_format.color_space)
                .image_extent(use_extent)
                .image_array_layers(1)
                .image_usage(use_usage)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .present_mode(use_mode)
                .clipped(true)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .old_swapchain(match old {
                    Some(MaybeFakeSwapchain::Real(ref swapchain)) => swapchain.handle,
                    _ => vk::SwapchainKHR::null(),
                });

            self.device.swapchain().create_swapchain(&info, None)
        };

        // Old swapchain is retired even if the creation of the new one fails.
        if let Some(old) = old {
            self.retired.push_back(old);
        }

        let handle = match result {
            Ok(handle) => handle,
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY) => handle_host_oom(),
            Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                self.device.set_oom();
                return Err(SurfaceError::OutOfMemory);
            }
            Err(vk::Result::ERROR_DEVICE_LOST) => {
                self.status.insert(SurfaceStatus::LOST);
                self.device.set_lost();
                return Err(SurfaceError::SurfaceLost);
            }
            Err(vk::Result::ERROR_SURFACE_LOST_KHR) => {
                self.status.insert(SurfaceStatus::LOST);
                return Err(SurfaceError::SurfaceLost);
            }
            Err(vk::Result::ERROR_NATIVE_WINDOW_IN_USE_KHR) => {
                panic!("Native window is already in use.");
            }
            Err(vk::Result::ERROR_INITIALIZATION_FAILED) => {
                panic!("Failed to create swapchain due to some implementation-specific reasons");
            }
            Err(err) => unexpected_error(err),
        };

        let next = new_semaphore(self.device.ash())?;

        let result = unsafe { self.device.swapchain().get_swapchain_images(handle) };
        let images = match result {
            Ok(images) => images,
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY) => handle_host_oom(),
            Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                self.device.set_oom();
                return Err(SurfaceError::OutOfMemory);
            }
            Err(err) => unexpected_error(err),
        };

        let mut swapchain_images = SmallVec::new();
        for &handle in &images {
            let (view, view_idx) = self
                .device
                .new_image_view(
                    handle,
                    vk::ImageViewType::TYPE_2D,
                    ViewDesc {
                        format: use_mev_format,
                        layers: (0..1).into(),
                        levels: (0..1).into(),
                        swizzle: Swizzle::IDENTITY,
                    },
                )
                .unwrap();

            let acquire = new_semaphore(self.device.ash())?;
            let present = new_semaphore(self.device.ash())?;

            let image = Image::from_swapchain_image(
                self.device.weak(),
                handle,
                view,
                view_idx,
                Extent2::new(use_extent.width, use_extent.height),
                use_mev_format,
                use_mev_usage,
            );

            swapchain_images.push((image, [acquire, present]));
        }

        self.current = Some(MaybeFakeSwapchain::Real(Swapchain {
            handle,
            extent: use_extent,
            format: use_format,
            mode: use_mode,
            usage: use_usage,
            images: swapchain_images,
            next,
            fences: self
                .device
                .swapchain_maintenance1()
                .is_some()
                .then(|| SwapchainFences {
                    array: SmallVec::new(),
                    next: 0,
                }),
        }));

        self.status.remove(SurfaceStatus::NEEDS_RECONFIGURE);
        self.cooldown = RECONFIGURE_COOLDOWN;

        Ok(())
    }

    fn handle_retired(&mut self) -> Result<(), DeviceError> {
        self.clear_retired(true)?;

        if self.retired.len() >= 8 {
            self.force_clear_retired()?;
        }

        Ok(())
    }

    fn force_clear_retired(&mut self) -> Result<(), DeviceError> {
        self.device.wait_idle()?;

        self.clear_retired(false)?;
        assert_eq!(
            self.retired.len(),
            0,
            "User-code should not hold on to swapchain images."
        );

        Ok(())
    }

    fn clear_retired(&mut self, mut do_wait: bool) -> Result<(), DeviceError> {
        let device = self.device.ash();

        while let Some(swapchain) = self.retired.pop_front() {
            match swapchain {
                MaybeFakeSwapchain::Real(swapchain) => {
                    let images_detached =
                        swapchain.images.iter().all(|(image, _)| image.detached());
                    let mut can_destroy = false;

                    if images_detached {
                        if do_wait {
                            match swapchain.fences {
                                None => {
                                    self.device.wait_idle()?;
                                    do_wait = false;
                                    can_destroy = true;
                                }
                                Some(ref fences) => {
                                    let mut all_signaled = true;
                                    for &fence in &fences.array {
                                        all_signaled &= self.device.get_fence_status(fence)?;
                                    }
                                    if all_signaled {
                                        can_destroy = true;
                                    }
                                }
                            }
                        } else {
                            can_destroy = true;
                        }
                    }

                    if can_destroy {
                        for (_, [acquire, present]) in swapchain.images {
                            unsafe {
                                device.destroy_semaphore(acquire, None);
                                device.destroy_semaphore(present, None);
                            }
                        }

                        unsafe {
                            device.destroy_semaphore(swapchain.next, None);
                        }

                        if let Some(fences) = swapchain.fences {
                            for fence in fences.array {
                                unsafe {
                                    device.destroy_fence(fence, None);
                                }
                            }
                        }

                        unsafe {
                            self.device
                                .swapchain()
                                .destroy_swapchain(swapchain.handle, None);
                        }
                    } else {
                        // Do this later.
                        self.retired.push_front(MaybeFakeSwapchain::Real(swapchain));
                        break;
                    }
                }
                MaybeFakeSwapchain::Fake(fake) => unsafe {
                    if fake.image.detached() {
                        device.destroy_semaphore(fake.semaphore, None);
                    } else {
                        // Do this later.
                        self.retired.push_front(MaybeFakeSwapchain::Fake(fake));
                        break;
                    }
                },
            }
        }

        Ok(())
    }
}

impl crate::traits::Resource for Surface {}

#[hidden_trait::expose]
impl crate::traits::Surface for Surface {
    fn available_formats(&self) -> &[(PixelFormat, ColorSpace)] {
        &self.mev_formats
    }

    fn preferred_format(&mut self, format: PixelFormat) {
        self.preferred_format = Some(format);
    }

    fn preferred_color_space(&mut self, color_space: ColorSpace) {
        self.preferred_color_space = Some(color_space);
    }

    fn preferred_extent(&mut self, extent: Extent2) {
        self.preferred_extent = Some(extent);
    }

    fn preferred_present_mode(&mut self, mode: PresentMode) {
        self.preferred_mode = Some(mode);
    }

    fn preferred_usage(&mut self, usage: ImageUsage) {
        self.preferred_usage = Some(usage);
    }

    fn preferred_image_count(&mut self, count: u32) {
        self.preferred_image_count = Some(count);
    }

    fn next_frame(&mut self) -> Result<Frame, SurfaceError> {
        self.preprocess()?;

        loop {
            let current = self.current.as_mut().unwrap();

            match current {
                MaybeFakeSwapchain::Real(swapchain) => {
                    let result = unsafe {
                        self.device.swapchain().acquire_next_image(
                            swapchain.handle,
                            u64::MAX,
                            swapchain.next,
                            vk::Fence::null(),
                        )
                    };
                    let idx = match result {
                        Ok((idx, false)) => idx,
                        Ok((idx, true)) => {
                            tracing::debug!("Surface '{:?}' is suboptimal", self.surface);
                            self.status.insert(SurfaceStatus::SUBOPTIMAL);
                            idx
                        }
                        Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY) => handle_host_oom(),
                        Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                            self.device.set_oom();
                            return Err(SurfaceError::OutOfMemory);
                        }
                        Err(vk::Result::ERROR_DEVICE_LOST) => {
                            self.status.insert(SurfaceStatus::LOST);
                            self.device.set_lost();
                            return Err(SurfaceError::DeviceLost);
                        }
                        Err(
                            vk::Result::ERROR_SURFACE_LOST_KHR
                            | vk::Result::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT,
                        ) => {
                            self.status.insert(SurfaceStatus::LOST);
                            return Err(SurfaceError::SurfaceLost);
                        }
                        Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                            tracing::debug!("Surface '{:?}' is out of date", self.surface);
                            self.status.insert(SurfaceStatus::OUT_OF_DATE);
                            self.cooldown = 0;
                            self.update_surface_parameters()?;
                            self.configure()?;
                            continue;
                        }
                        Err(err) => unexpected_error(err),
                    };

                    let (ref image, [ref mut acquire, present]) = swapchain.images[idx as usize];
                    std::mem::swap(&mut swapchain.next, acquire);

                    let fence = match &mut swapchain.fences {
                        None => vk::Fence::null(),
                        Some(fences) if fences.array.is_empty() => {
                            let fence = self.device.new_fence()?;
                            fences.array.push(fence);
                            fences.next = 0;
                            fence
                        }
                        Some(fences) => {
                            let fence = fences.array[fences.next];
                            if self.device.get_fence_status(fence)? {
                                self.device.reset_fences(&[fence])?;
                                fences.next = (fences.next + 1) % fences.array.len();
                                fence
                            } else {
                                let fence = self.device.new_fence()?;
                                fences.array.insert(fences.next, fence);
                                fences.next = (fences.next + 1) % fences.array.len();
                                fence
                            }
                        }
                    };

                    return Ok(Frame {
                        swapchain: swapchain.handle,
                        image: ImageN::new_unchecked(image.clone()),
                        idx,
                        acquire: *acquire,
                        present,
                        synced: false,
                        fence,
                    });
                }
                MaybeFakeSwapchain::Fake(fake) => {
                    if !self.surface.is_null() {
                        // Reconfigure each frame the fake swapchain in real surface.
                        // Fake is only used here when maxExtent is {0,0}
                        // As soon as it is not, we will reconfigure to real swapchain.
                        self.status.insert(SurfaceStatus::SUBOPTIMAL);
                    }

                    let frame = Frame {
                        swapchain: vk::SwapchainKHR::null(),
                        image: ImageN::new_unchecked(fake.image.clone()),
                        idx: 0,
                        acquire: if fake.frame_idx > 0 {
                            fake.semaphore
                        } else {
                            vk::Semaphore::null()
                        },
                        present: fake.semaphore,
                        synced: false,
                        fence: vk::Fence::null(),
                    };
                    fake.frame_idx += 1;
                    return Ok(frame);
                }
            }
        }
    }
}

pub struct Frame {
    pub(super) image: Image2D,
    pub(super) swapchain: vk::SwapchainKHR,
    pub(super) idx: u32,
    pub(super) acquire: vk::Semaphore,
    pub(super) present: vk::Semaphore,
    pub(super) synced: bool,
    pub(super) fence: vk::Fence,
}

impl Frame {
    #[inline(always)]
    pub(super) fn is_real(&self) -> bool {
        self.swapchain != vk::SwapchainKHR::null()
    }

    #[inline(always)]
    pub(super) fn present_layout(&self) -> vk::ImageLayout {
        if self.is_real() {
            vk::ImageLayout::PRESENT_SRC_KHR
        } else {
            vk::ImageLayout::GENERAL
        }
    }
}

impl crate::traits::Resource for Frame {}

#[hidden_trait::expose]
impl crate::traits::Frame for Frame {
    #[inline(always)]
    fn image(&self) -> &Image2D {
        &self.image
    }
}

fn pick_format(
    formats: &[vk::SurfaceFormatKHR],
    preferred_format: Option<PixelFormat>,
    preferred_color_space: Option<ColorSpace>,
) -> vk::SurfaceFormatKHR {
    match formats.len() {
        0 => panic!("No surface formats available"),
        _ => {
            let format_priority = [
                Some(vk::Format::B8G8R8A8_UNORM),
                Some(vk::Format::R8G8B8A8_UNORM),
                Some(vk::Format::B8G8R8A8_SRGB),
                Some(vk::Format::R8G8B8A8_SRGB),
                preferred_format.and_then(TryIntoAsh::try_into_ash),
            ];

            let color_space_priority = [
                Some(vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT),
                Some(vk::ColorSpaceKHR::SRGB_NONLINEAR),
                preferred_color_space.map(IntoAsh::into_ash),
            ];

            let mut best_candidate = formats[0];
            let mut best_score = 0usize;

            for &candidate in formats {
                if PixelFormat::try_from_ash(candidate.format).is_none() {
                    continue;
                }

                if ColorSpace::try_from_ash(candidate.color_space).is_none() {
                    continue;
                }

                let mut score: usize = 1;

                for (p, f) in format_priority.iter().enumerate() {
                    if Some(candidate.format) == *f {
                        score += (p + 1) * 256;
                    }
                }

                for (p, cs) in color_space_priority.iter().enumerate() {
                    if Some(candidate.color_space) == *cs {
                        score += (p + 1);
                    }
                }

                if score > best_score {
                    best_candidate = candidate;
                    best_score = score;
                }
            }

            if best_score > 0 {
                return best_candidate;
            }

            panic!(
                "No usable format found for surface. Available formats: {:?}",
                formats
            );
        }
    }
}

fn pick_mode(
    modes: &[vk::PresentModeKHR],
    preferred_mode: Option<PresentMode>,
) -> vk::PresentModeKHR {
    match modes.len() {
        0 => panic!("No surface present modes available"),
        _ => {
            let mode_priority = [
                Some(vk::PresentModeKHR::FIFO),
                Some(vk::PresentModeKHR::IMMEDIATE),
                Some(vk::PresentModeKHR::MAILBOX),
                preferred_mode.map(IntoAsh::into_ash),
            ];

            let mut best_candidate = modes[0];
            let mut best_score = 0usize;

            for &candidate in modes {
                if PresentMode::try_from_ash(candidate).is_none() {
                    continue;
                }

                let mut score: usize = 1;

                for (p, mode) in mode_priority.iter().enumerate() {
                    if Some(candidate) == *mode {
                        score += (p + 1) * 256;
                    }
                }

                if score > best_score {
                    best_candidate = candidate;
                    best_score = score;
                }
            }

            if best_score > 0 {
                return best_candidate;
            }

            panic!(
                "No usable present modes found for surface. Available present modes: {:?}",
                modes
            );
        }
    }
}

fn pick_extent(
    caps: &vk::SurfaceCapabilitiesKHR,
    preferred_extent: Option<Extent2>,
) -> vk::Extent2D {
    if caps.current_extent.width == u32::MAX && caps.current_extent.height == u32::MAX {
        match preferred_extent {
            None => caps.min_image_extent,
            Some(preferred) => {
                let width = preferred
                    .width()
                    .min(caps.max_image_extent.width)
                    .max(caps.min_image_extent.width);
                let height = preferred
                    .height()
                    .min(caps.max_image_extent.height)
                    .max(caps.min_image_extent.height);
                vk::Extent2D { width, height }
            }
        }
    } else {
        caps.current_extent
    }
}

fn pick_usage(
    caps: &vk::SurfaceCapabilitiesKHR,
    preferred_usage: Option<ImageUsage>,
    pixel_format: PixelFormat,
) -> vk::ImageUsageFlags {
    let usage =
        vk::ImageUsageFlags::ash_from((preferred_usage.unwrap_or(ImageUsage::all()), pixel_format));
    usage & caps.supported_usage_flags
}

fn dedup_vec<T>(vec: &mut Vec<T>)
where
    T: Copy + Eq,
{
    let mut i = 0;
    while i < vec.len() {
        let elem = vec[i];
        let mut j = i + 1;
        while j < vec.len() {
            if vec[j] == elem {
                vec.remove(j);
            } else {
                j += 1;
            }
        }
        i += 1;
    }
}
