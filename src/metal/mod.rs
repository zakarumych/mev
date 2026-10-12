mod acst;
mod arguments;
mod buffer;
mod mapped;
mod command;
mod compute;
mod device;
mod event;
mod from;
mod image;
mod instance;
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

const MAX_VERTEX_BUFFERS: u32 = 31;

pub mod for_macro {
    pub use super::arguments::{Arguments, ArgumentsField};
}

pub(crate) trait Resource: Send + Sync + Sized + 'static {}
