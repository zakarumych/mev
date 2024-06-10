use std::{
    collections::VecDeque,
    fmt,
    ops::Deref,
    time::{Duration, Instant},
};

use ash::vk;

use crate::generic::{
    Extent2, ImageExtent, OutOfMemory, PipelineStages, SurfaceError, Swizzle, ViewDesc,
};

use super::{
    from::{AshInto, TryAshInto},
    handle_host_oom, unexpected_error, Device, Image, Queue,
};

const SUBOPTIMAL_RETIRE_COOLDOWN: u64 = 10;

struct Swapchain {
    handle: vk::SwapchainKHR,
    images: Vec<(Image, [vk::Semaphore; 2])>,
    next: vk::Semaphore,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SuboptimalRetire {
    Cooldown(u64),
    Retire,
}

pub struct Surface {
    device: Device,
    surface: vk::SurfaceKHR,
    current: Option<Swapchain>,
    retired: VecDeque<Swapchain>,
    caps: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    modes: Vec<vk::PresentModeKHR>,
    family_supports: Vec<bool>,

    preferred_format: vk::SurfaceFormatKHR,
    preferred_mode: vk::PresentModeKHR,
    preferred_usage: vk::ImageUsageFlags,
    bound_queue_family: Option<u32>,

    /// Number of frames to wait before retiring a suboptimal swapchain.
    suboptimal_retire: SuboptimalRetire,

    /// Signals that surface or device was lost.
    lost: bool,
}

impl Drop for Surface {
    fn drop(&mut self) {
        let _ = self.device.wait_idle();
        self.clear_retired();

        let device = self.device.ash();

        if let Some(mut swapchain) = self.current.take() {
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

            unsafe {
                self.device
                    .swapchain()
                    .destroy_swapchain(swapchain.handle, None);
            }
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
        let preferred_format = pick_format(&formats);
        let preferred_mode = pick_mode(&modes);

        tracing::info!(
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
            modes,
            family_supports,

            preferred_format,
            preferred_mode,
            preferred_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            bound_queue_family: None,

            suboptimal_retire: SuboptimalRetire::Cooldown(SUBOPTIMAL_RETIRE_COOLDOWN),
            lost: false,
        }
    }

    // Initialize the swapchain.
    // Retires any old swapchain.
    fn init(&mut self) -> Result<(), SurfaceError> {
        self.handle_retired()?;

        if self.lost {
            return Err(SurfaceError::SurfaceLost);
        }
        self.suboptimal_retire = SuboptimalRetire::Cooldown(SUBOPTIMAL_RETIRE_COOLDOWN);

        let old = self.current.take();

        let result = unsafe {
            self.device
                .surface()
                .get_physical_device_surface_capabilities(
                    self.device.physical_device(),
                    self.surface,
                )
        };
        self.caps = result.map_err(|err| match err {
            vk::Result::ERROR_OUT_OF_HOST_MEMORY => handle_host_oom(),
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => SurfaceError::OutOfMemory,
            vk::Result::ERROR_SURFACE_LOST_KHR => {
                self.lost = true;
                SurfaceError::SurfaceLost
            }
            _ => unexpected_error(err),
        })?;

        let result = unsafe {
            self.device.swapchain().create_swapchain(
                &vk::SwapchainCreateInfoKHR::default()
                    .surface(self.surface)
                    .min_image_count(3.clamp(self.caps.min_image_count, self.caps.max_image_count))
                    .image_format(self.preferred_format.format)
                    .image_color_space(self.preferred_format.color_space)
                    .image_extent(self.caps.current_extent)
                    .image_array_layers(1)
                    .image_usage(self.caps.supported_usage_flags & self.preferred_usage)
                    .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .present_mode(self.preferred_mode)
                    .clipped(true)
                    .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                    .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                    .old_swapchain(old.as_ref().map_or(vk::SwapchainKHR::null(), |s| s.handle)),
                None,
            )
        };

        // Old swapchain is retired even if the creation of the new one fails.
        if let Some(old) = old {
            self.retired.push_back(old);
        }

        let handle = result.map_err(|err| match err {
            vk::Result::ERROR_OUT_OF_HOST_MEMORY => handle_host_oom(),
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => SurfaceError::OutOfMemory,
            vk::Result::ERROR_DEVICE_LOST | vk::Result::ERROR_SURFACE_LOST_KHR => {
                self.lost = true;
                SurfaceError::SurfaceLost
            }
            vk::Result::ERROR_NATIVE_WINDOW_IN_USE_KHR => {
                panic!("Native window is already in use.");
            }
            vk::Result::ERROR_INITIALIZATION_FAILED => {
                panic!("Failed to create swapchain due to some implementation-specific reasons");
            }
            _ => unexpected_error(err),
        })?;

        let semaphore = |device: &ash::Device| {
            let result =
                unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) };
            result.map_err(|err| match err {
                vk::Result::ERROR_OUT_OF_HOST_MEMORY => handle_host_oom(),
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => SurfaceError::OutOfMemory,
                _ => unexpected_error(err),
            })
        };

        let next = semaphore(self.device.ash())?;

        let result = unsafe { self.device.swapchain().get_swapchain_images(handle) };
        let images = result.map_err(|err| match err {
            vk::Result::ERROR_OUT_OF_HOST_MEMORY => handle_host_oom(),
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => SurfaceError::OutOfMemory,
            _ => unexpected_error(err),
        })?;

        let pixel_format = self.preferred_format.format.try_ash_into().unwrap();
        let usage = self.preferred_usage.ash_into();

        let mut swapchain_images = Vec::new();
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

            let acquire = semaphore(self.device.ash())?;
            let present = semaphore(self.device.ash())?;

            let image = Image::from_swapchain_image(
                self.device.weak(),
                handle,
                view,
                view_idx,
                Extent2::new(
                    self.caps.current_extent.width,
                    self.caps.current_extent.height,
                ),
                pixel_format,
                usage,
            );

            swapchain_images.push((image, [acquire, present]));
        }

        self.current = Some(Swapchain {
            handle,
            images: swapchain_images,
            next,
        });
        Ok(())
    }

    fn handle_retired(&mut self) -> Result<(), SurfaceError> {
        self.clear_retired();

        if self.retired.len() >= 8 {
            self.device
                .wait_idle()
                .map_err(|OutOfMemory| SurfaceError::OutOfMemory)?;

            self.clear_retired();
            assert_eq!(
                self.retired.len(),
                0,
                "User-code should not hold on to swapchain images."
            );
        }

        Ok(())
    }

    fn clear_retired(&mut self) {
        let device = self.device.ash();

        while let Some(mut swapchain) = self.retired.pop_front() {
            let can_destroy = swapchain
                .images
                .iter_mut()
                .all(|(image, _)| image.detached());
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

                unsafe {
                    self.device
                        .swapchain()
                        .destroy_swapchain(swapchain.handle, None);
                }
            } else {
                // Do this later.
                self.retired.push_front(swapchain);
                break;
            }
        }
    }
}

#[hidden_trait::expose]
impl crate::traits::Surface for Surface {
    fn next_frame(&mut self) -> Result<Frame, SurfaceError> {
        self.clear_retired();

        match self.suboptimal_retire {
            SuboptimalRetire::Cooldown(0) => {}
            SuboptimalRetire::Cooldown(ref mut n) => {
                *n -= 1;
            }
            SuboptimalRetire::Retire => {
                self.init()?;
            }
        }

        if self.current.is_none() {
            self.init()?;
        }

        loop {
            let current = self.current.as_mut().unwrap();

            let result = unsafe {
                self.device.swapchain().acquire_next_image(
                    current.handle,
                    u64::MAX,
                    current.next,
                    vk::Fence::null(),
                )
            };
            let idx = match result {
                Ok((idx, false)) => idx,
                Ok((idx, true)) => {
                    if self.suboptimal_retire == SuboptimalRetire::Cooldown(0) {
                        self.suboptimal_retire = SuboptimalRetire::Retire;
                    }
                    idx
                }
                Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY) => handle_host_oom(),
                Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                    return Err(SurfaceError::OutOfMemory)
                }
                Err(
                    vk::Result::ERROR_DEVICE_LOST
                    | vk::Result::ERROR_SURFACE_LOST_KHR
                    | vk::Result::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT,
                ) => {
                    self.lost = true;
                    return Err(SurfaceError::SurfaceLost);
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.init()?;
                    continue;
                }
                Err(err) => unexpected_error(err),
            };

            let (ref image, [ref mut acquire, present]) = current.images[idx as usize];
            std::mem::swap(&mut current.next, acquire);

            return Ok(Frame {
                swapchain: current.handle,
                image: image.clone(),
                idx,
                acquire: *acquire,
                present,
                synced: false,
            });
        }
    }
}

pub struct Frame {
    pub(super) swapchain: vk::SwapchainKHR,
    pub(super) image: Image,
    pub(super) idx: u32,
    pub(super) acquire: vk::Semaphore,
    pub(super) present: vk::Semaphore,
    pub(super) synced: bool,
}

impl Deref for Frame {
    type Target = Image;

    fn deref(&self) -> &Self::Target {
        &self.image
    }
}

#[hidden_trait::expose]
impl crate::traits::Frame for Frame {
    fn image(&self) -> &Image {
        &self.image
    }
}

fn pick_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
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
    panic!("Can't pick present mode");
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
