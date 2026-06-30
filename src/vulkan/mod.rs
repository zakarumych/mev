use std::{alloc::Layout, fmt};

use ash::vk;

mod access;
mod acst;
mod arguments;
mod buffer;
mod command;
mod compute;
mod device;
mod from;
mod image;
mod instance;
mod layout;
mod queue;
mod refs;
mod render;
mod sampler;
mod shader;
mod surface;

use crate::generic::{DeviceError, OutOfMemory, PixelFormat};

pub use self::{
    acst::{Blas, Tlas},
    buffer::Buffer,
    command::{
        AccelerationStructureCommandEncoder, CommandBuffer, CommandEncoder, ComputeCommandEncoder,
        CopyCommandEncoder, RenderCommandEncoder,
    },
    compute::ComputePipeline,
    device::Device,
    image::Image,
    instance::Instance,
    queue::Queue,
    render::RenderPipeline,
    sampler::Sampler,
    shader::Library,
    surface::{Frame, Surface},
};

#[track_caller]
fn handle_host_oom() -> ! {
    std::alloc::handle_alloc_error(Layout::new::<()>())
}

#[track_caller]
fn unexpected_error(err: vk::Result) -> ! {
    panic!("unexpected error: {err:?}")
}

/// Version of the API.
/// For internal use only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl Version {
    const V1_0: Self = Self {
        major: 1,
        minor: 0,
        patch: 0,
    };

    const V1_1: Self = Self {
        major: 1,
        minor: 1,
        patch: 0,
    };

    const V1_2: Self = Self {
        major: 1,
        minor: 2,
        patch: 0,
    };

    const V1_3: Self = Self {
        major: 1,
        minor: 3,
        patch: 0,
    };

    fn api_version(&self) -> u32 {
        vk::make_api_version(0, self.major, self.minor, self.patch)
    }
}

impl fmt::Display for Version {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

#[inline]
fn format_aspect(format: PixelFormat) -> vk::ImageAspectFlags {
    let mut aspect = vk::ImageAspectFlags::empty();
    if format.is_color() {
        aspect |= vk::ImageAspectFlags::COLOR;
    }
    if format.is_depth() {
        aspect |= vk::ImageAspectFlags::DEPTH;
    }
    if format.is_stencil() {
        aspect |= vk::ImageAspectFlags::STENCIL;
    }
    aspect
}

#[track_caller]
fn map_oom(err: vk::Result) -> OutOfMemory {
    match err {
        ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY => handle_host_oom(),
        ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => OutOfMemory,
        _ => unexpected_error(err),
    }
}

fn map_device_error(err: vk::Result) -> DeviceError {
    match err {
        ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY => handle_host_oom(),
        ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => DeviceError::OutOfMemory,
        ash::vk::Result::ERROR_DEVICE_LOST => DeviceError::DeviceLost,
        _ => unexpected_error(err),
    }
}

fn new_semaphore(device: &ash::Device) -> Result<vk::Semaphore, OutOfMemory> {
    let result = unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) };
    result.map_err(|err| match err {
        vk::Result::ERROR_OUT_OF_HOST_MEMORY => handle_host_oom(),
        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => OutOfMemory,
        _ => unexpected_error(err),
    })
}

pub mod for_macro {
    pub use super::{
        arguments::{descriptor_type, Arguments, ArgumentsField},
        refs::Refs,
    };
    pub use ash::vk::DescriptorUpdateTemplateEntry;
}
