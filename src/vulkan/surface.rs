use std::{collections::VecDeque, ops::Deref};

use ash::vk::{self, Handle};
use bitflags::Flags;
use smallvec::SmallVec;

use crate::{
    DeviceError, ImageDesc, PixelFormat,
    backend::{
        from::{IntoAsh, TryFromAsh, TryIntoAsh},
        new_semaphore,
    },
    generic::{Extent2, ImageExtent, SurfaceError, Swizzle, ViewDesc},
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
        const NOT_PREFERRED_EXTENT = 0x08;
        const NOT_PREFERRED_FORMAT = 0x10;

        const NEEDS_RECONFIGURE = 0x1E; // OUT_OF_DATE | SUBOPTIMAL | NOT_PREFERRED_EXTENT | NOT_PREFERRED_FORMAT
    }
}

const RECONFIGURE_COOLDOWN: u64 = 10;

struct SwapchainFences {
    array: SmallVec<[vk::Fence; 4]>,
    next: usize,
}

struct Swapchain {
    handle: vk::SwapchainKHR,
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
    formats: Vec<vk::SurfaceFormatKHR>,
    pixel_formats: Vec<PixelFormat>,
    modes: Vec<vk::PresentModeKHR>,
    family_supports: Vec<bool>,

    preferred_format: vk::SurfaceFormatKHR,
    preferred_mode: vk::PresentModeKHR,
    preferred_usage: vk::ImageUsageFlags,
    preferred_extent: Option<Extent2>,

    bound_queue_family: Option<u32>,

    reconfigure_cooldown: u64,
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
    pub(super) fn new(
        device: Device,
        surface: vk::SurfaceKHR,
        formats: Vec<vk::SurfaceFormatKHR>,
        modes: Vec<vk::PresentModeKHR>,
        family_supports: Vec<bool>,
    ) -> Self {
        let preferred_format = pick_format(&formats, None);
        let preferred_mode = pick_mode(&modes);

        let mut pixel_formats = formats
            .iter()
            .filter_map(|f| match f.format.try_ash_into() {
                Some(pixel_format) => Some(pixel_format),
                None => {
                    tracing::debug!("Surface format '{:?}' is not supported by mev", f.format);
                    None
                }
            })
            .collect::<Vec<_>>();

        dedup_vec(&mut pixel_formats);

        tracing::debug!(
            "New surface preferred format: '{:?}' and mode: '{:?}'",
            preferred_format,
            preferred_mode
        );

        Surface {
            device,
            surface,
            current: None,
            retired: VecDeque::new(),
            caps: vk::SurfaceCapabilitiesKHR::default(),
            formats,
            pixel_formats,
            modes,
            family_supports,

            preferred_format,
            preferred_mode,
            preferred_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            preferred_extent: None,

            bound_queue_family: None,

            reconfigure_cooldown: 0,
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

        let preferred_usage = (image.usage(), image.format()).into_ash();

        Surface {
            device,
            surface: vk::SurfaceKHR::null(),
            retired: VecDeque::new(),
            caps: vk::SurfaceCapabilitiesKHR::default(),
            formats: vec![surface_format],
            pixel_formats: vec![image.format()],
            modes: Vec::new(),
            family_supports: Vec::new(),

            preferred_format: surface_format,
            preferred_mode: vk::PresentModeKHR::IMMEDIATE,
            preferred_usage,
            preferred_extent: Some(image.extent().expect_2d()),

            bound_queue_family: None,

            reconfigure_cooldown: 0,

            status: SurfaceStatus::empty(),

            current: Some(MaybeFakeSwapchain::Fake(FakeSwapchain {
                image,
                semaphore,
                frame_idx: 0,
            })),
        }
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

        let use_extent = if self.caps.current_extent.width == u32::MAX
            && self.caps.current_extent.height == u32::MAX
        {
            match self.preferred_extent {
                None => self.caps.min_image_extent,
                Some(preferred) => {
                    let width = preferred
                        .width()
                        .min(self.caps.max_image_extent.width)
                        .max(self.caps.min_image_extent.width);
                    let height = preferred
                        .height()
                        .min(self.caps.max_image_extent.height)
                        .max(self.caps.min_image_extent.height);
                    vk::Extent2D { width, height }
                }
            }
        } else {
            self.caps.current_extent
        };

        if self.status == SurfaceStatus::NOT_PREFERRED_EXTENT {
            // Reconfiguration is only requested because the preferred extent is different from the current one.
            // If the extent new surface would be created is the same as the current one, we can skip reconfiguration.
            if let Some(MaybeFakeSwapchain::Real(swapchain)) = &self.current {
                let current_extent = swapchain.images[0].0.extent().expect_2d();
                if current_extent.width() == use_extent.width
                    && current_extent.height() == use_extent.height
                {
                    self.reconfigure_cooldown = RECONFIGURE_COOLDOWN;
                    return Ok(());
                }
            }
        }

        if use_extent.width == 0 || use_extent.height == 0 {
            if let Some(MaybeFakeSwapchain::Fake(swapchain)) = &self.current {
                self.status.insert(SurfaceStatus::SUBOPTIMAL);
                self.reconfigure_cooldown = RECONFIGURE_COOLDOWN;
                return Ok(());
            }

            match self.current.take() {
                None => {}
                Some(MaybeFakeSwapchain::Real(swapchain)) => {
                    self.retired.push_back(MaybeFakeSwapchain::Real(swapchain));
                }
                _ => unreachable!(),
            }

            let pixel_format = self.preferred_format.format.try_ash_into().unwrap();

            let image = self.device.new_image(ImageDesc {
                extent: ImageExtent::D2(Extent2::new(1, 1)),
                format: pixel_format,
                usage: self.preferred_usage.ash_into(),
                layers: 1,
                levels: 1,
                name: "fake-swapchain-image",
            });

            let semaphore = new_semaphore(self.device.ash())?;

            self.current = Some(MaybeFakeSwapchain::Fake(FakeSwapchain {
                image,
                semaphore,
                frame_idx: 0,
            }));

            if self.preferred_extent == Some(Extent2::new(1, 1)) {
                self.status.remove(SurfaceStatus::NOT_PREFERRED_EXTENT);
            }
            self.status.remove(SurfaceStatus::NOT_PREFERRED_FORMAT);
            self.status.remove(SurfaceStatus::OUT_OF_DATE);
            self.status.insert(SurfaceStatus::SUBOPTIMAL);
            self.reconfigure_cooldown = RECONFIGURE_COOLDOWN;

            return Ok(());
        }

        let image_count = match (self.caps.min_image_count, self.caps.max_image_count) {
            (min, 0) => 3.max(min),
            (min, max) => 3.clamp(min, max),
        };

        let old = self.current.take();

        let result = unsafe {
            let mut info = vk::SwapchainCreateInfoKHR::default();
            let mut present_modes_info = vk::SwapchainPresentModesCreateInfoEXT::default();
            let modes = [self.preferred_mode];

            if self.device.swapchain_maintenance1().is_some() {
                present_modes_info = present_modes_info.present_modes(&modes);
                info = info.push_next(&mut present_modes_info)
            }

            info = info
                .surface(self.surface)
                .min_image_count(image_count)
                .image_format(self.preferred_format.format)
                .image_color_space(self.preferred_format.color_space)
                .image_extent(use_extent)
                .image_array_layers(1)
                .image_usage(self.caps.supported_usage_flags & self.preferred_usage)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .present_mode(self.preferred_mode)
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

        let pixel_format = self
            .preferred_format
            .format
            .try_ash_into()
            .expect("Only covertible format can be picked");
        let usage = self.preferred_usage.ash_into();

        let mut swapchain_images = SmallVec::new();
        for &handle in &images {
            let (view, view_idx) = self
                .device
                .new_image_view(
                    handle,
                    vk::ImageViewType::TYPE_2D,
                    ViewDesc {
                        format: pixel_format,
                        base_layer: 0,
                        layers: 1,
                        base_level: 0,
                        levels: 1,
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
                pixel_format,
                usage,
            );

            swapchain_images.push((image, [acquire, present]));
        }

        self.current = Some(MaybeFakeSwapchain::Real(Swapchain {
            handle,
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

        if self.preferred_extent == Some(Extent2::new(use_extent.width, use_extent.height)) {
            self.status.remove(SurfaceStatus::NOT_PREFERRED_EXTENT);
        }
        self.status.remove(SurfaceStatus::NOT_PREFERRED_FORMAT);
        self.status
            .remove(SurfaceStatus::OUT_OF_DATE | SurfaceStatus::SUBOPTIMAL);
        self.reconfigure_cooldown = RECONFIGURE_COOLDOWN;

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
    fn available_formats(&self) -> &[PixelFormat] {
        &self.pixel_formats
    }

    fn preferred_format(&mut self, format: PixelFormat) {
        let format = pick_format(&self.formats, Some(format));
        if self.preferred_format != format {
            self.preferred_format = format;
        }

        if let Some(MaybeFakeSwapchain::Real(swapchain)) = &self.current {
            let current_format = swapchain.images[0].0.format();
            if current_format != format.format.try_ash_into().unwrap() {
                self.status.insert(SurfaceStatus::NOT_PREFERRED_FORMAT);
            }
        }
    }

    fn preferred_extent(&mut self, extent: Extent2) {
        if self.preferred_extent != Some(extent) {
            self.preferred_extent = Some(extent);
        }

        if let Some(MaybeFakeSwapchain::Real(swapchain)) = &self.current {
            let current_format = swapchain.images[0].0.extent();
            if current_format.expect_2d() != extent {
                self.status.insert(SurfaceStatus::NOT_PREFERRED_EXTENT);
            }
        }
    }

    fn next_frame(&mut self) -> Result<Frame, SurfaceError> {
        self.clear_retired(true)?;

        self.reconfigure_cooldown = self.reconfigure_cooldown.saturating_sub(1);

        if self.status.intersects(SurfaceStatus::NEEDS_RECONFIGURE)
            && self.reconfigure_cooldown == 0
        {
            self.configure()?;
        }

        if self.current.is_none() {
            self.configure()?;
        }

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
                            self.reconfigure_cooldown = 0;
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
                                fence
                            }
                        }
                    };

                    return Ok(Frame {
                        swapchain: swapchain.handle,
                        image: image.clone(),
                        idx,
                        acquire: *acquire,
                        present,
                        synced: false,
                        fence,
                    });
                }
                MaybeFakeSwapchain::Fake(fake) => {
                    if self.surface.is_null() {
                        // Reconfigure each frame the fake swapchain in real surface.
                        // Fake is only used here when maxExtent is {0,0}
                        // As soon as it is not, we will reconfigure to real swapchain.
                        self.status.insert(SurfaceStatus::SUBOPTIMAL);
                    }

                    let frame = Frame {
                        swapchain: vk::SwapchainKHR::null(),
                        image: fake.image.clone(),
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
    pub(super) image: Image,
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
    #[inline]
    fn image(&self) -> &Image {
        &self.image
    }
}

fn pick_format(
    formats: &[vk::SurfaceFormatKHR],
    pixel_format: Option<PixelFormat>,
) -> vk::SurfaceFormatKHR {
    if let Some(pixel_format) = pixel_format.and_then(|f| f.try_into_ash()) {
        for &format in formats {
            if format.format == pixel_format {
                return format;
            }
        }
    }

    for &format in formats {
        if format.format == vk::Format::R8G8B8A8_UNORM {
            return format;
        }
    }

    for &format in formats {
        if format.format == vk::Format::B8G8R8A8_UNORM {
            return format;
        }
    }

    for &format in formats {
        if format.format == vk::Format::B8G8R8A8_SRGB {
            return format;
        }
    }

    for &format in formats {
        if format.format == vk::Format::R8G8B8A8_SRGB {
            return format;
        }
    }

    for &format in formats {
        if PixelFormat::try_from_ash(format.format).is_some() {
            return format;
        }
    }

    panic!(
        "No usable format found for surface. Available formats: {:?}",
        formats
    );
}

fn pick_mode(modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    for &mode in modes {
        if mode == vk::PresentModeKHR::MAILBOX {
            return mode;
        }
    }
    for &mode in modes {
        if mode == vk::PresentModeKHR::FIFO {
            return mode;
        }
    }
    for &mode in modes {
        if mode == vk::PresentModeKHR::IMMEDIATE {
            return mode;
        }
    }
    panic!("Can't pick present mode");
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
