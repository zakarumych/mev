mod acst;
mod arguments;
mod buffer;
mod mapped;
mod command;
mod compute;
// mod convert;
mod device;
mod event;
mod from;
mod image;
mod instance;
#[path = "../unsupported_io.rs"]
mod io;
mod queue;
mod render;
mod sampler;
mod shader;
mod surface;

pub use self::{
    acst::{Blas, Tlas},
    buffer::Buffer,
    mapped::{BufferMappedRange, BufferMappedRangeMut},
    command::{
        AccelerationStructureCommandEncoder, CommandBuffer, CommandEncoder, ComputeCommandEncoder,
        CopyCommandEncoder, RenderCommandEncoder,
    },
    compute::ComputePipeline,
    device::Device,
    event::Event,
    image::Image,
    instance::Instance,
    io::{IoBatch, IoStatus},
    queue::Queue,
    render::RenderPipeline,
    sampler::Sampler,
    shader::Library,
    surface::{Frame, Surface},
};

pub mod for_macro {
    pub use super::arguments::{Arguments, ArgumentsField};
    pub use wgpu;
}

pub(crate) trait Resource: Sized + 'static {}
