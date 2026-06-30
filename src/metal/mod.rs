mod acst;
mod arguments;
mod buffer;
mod command;
mod compute;
mod device;
mod from;
mod image;
mod instance;
mod queue;
mod render;
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

const MAX_VERTEX_BUFFERS: u32 = 31;

pub mod for_macro {
    pub use super::arguments::{Arguments, ArgumentsField};
}
