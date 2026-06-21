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

// Minimize functions size by offloading panic to a separate function.
#[cold]
#[inline]
#[track_caller]
fn out_of_bounds() -> ! {
    panic!("offset + data.len() > buffer.length()");
}

const MAX_VERTEX_BUFFERS: u32 = 31;

pub mod for_macro {
    pub use super::arguments::{Arguments, ArgumentsField};
}
