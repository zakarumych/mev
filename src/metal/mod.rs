mod acst;
mod arguments;
mod buffer;
mod command;
mod compute_pipeline;
mod device;
mod from;
mod image;
mod instance;
mod queue;
mod render_pipeline;
mod sampler;
mod shader;
mod surface;

pub use self::{
    acst::{Blas, Tlas},
    buffer::Buffer,
    command::{
        AccelerationStructureCommandEncoder, CommandBuffer, CommandEncoder, ComputeCommandEncoder,
        CopyCommandEncoder, RenderCommandEncoder,
    },
    compute_pipeline::ComputePipeline,
    device::Device,
    image::Image,
    instance::Instance,
    queue::Queue,
    render_pipeline::RenderPipeline,
    sampler::Sampler,
    shader::Library,
    surface::{Frame, Surface},
};

pub(crate) use self::{
    instance::{CreateErrorKind, LoadErrorKind},
    render_pipeline::CreatePipelineErrorKind,
};

// Minimize functions size by offloading panic to a separate function.
#[cold]
#[cfg_attr(feature = "inline-more", inline(always))]
#[track_caller]
fn out_of_bounds() -> ! {
    panic!("offset + data.len() > buffer.length()");
}

const MAX_VERTEX_BUFFERS: u32 = 31;

pub mod for_macro {
    pub use crate::generic::DeviceRepr;

    pub use super::arguments::{Arguments, ArgumentsField};
    pub use bytemuck::{Pod, Zeroable};
    pub use std::{
        mem::{align_of, size_of, MaybeUninit},
        ptr::addr_of,
    };

    pub const fn align_end(end: usize, align: usize) -> usize {
        ((end + (align - 1)) & !(align - 1))
    }

    pub const fn repr_pad_for<T: DeviceRepr>(end: usize) -> usize {
        let align = T::ALIGN;
        pad_align(end, align)
    }

    pub const fn pad_align(end: usize, align: usize) -> usize {
        align_end(end, align) - end
    }

    pub const fn repr_append_field<T: DeviceRepr>(end: usize) -> usize {
        align_end(end, T::ALIGN) + T::SIZE
    }

    pub const fn repr_align_of<T: DeviceRepr>() -> usize {
        T::ALIGN
    }
}
