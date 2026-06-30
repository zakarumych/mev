mod arguments;
mod buffer;
mod command;
mod device;
mod image;
mod instance;
mod queue;
mod render;
mod sampler;
mod shader;
mod surface;

pub struct Tlas;
pub struct Blas;
pub struct ComputePipeline;

pub use self::{
    buffer::Buffer,
    command::{
        AccelerationStructureCommandEncoder, CommandBuffer, CommandEncoder, ComputeCommandEncoder,
        CopyCommandEncoder, RenderCommandEncoder,
    },
    device::Device,
    image::Image,
    instance::Instance,
    queue::Queue,
    render::RenderPipeline,
    sampler::Sampler,
    shader::Library,
    surface::{Frame, Surface},
};

#[doc(hidden)]
pub mod for_macro {
    pub use super::arguments::{Arguments, ArgumentsField};
}

/// A helper function to hash a `JsValue` by its underlying handle.
///
/// It relies on the fact that `JsValue` is represented as a 32-bit integer handle in WebAssembly,
/// and uses `transmute_copy` to convert it to a `u32` for hashing.
///
/// Size and alignment assertions are included to ensure that the assumptions about `JsValue` representation hold true.
/// However it is possible that JsValue representation may change in future versions of wasm-bindgen or the WebAssembly specification,
/// in a way that will not break assertions but will still break the hashing.
/// I hope this won't happen ^^
unsafe fn impudent_hash(val: &wasm_bindgen::JsValue, hasher: &mut impl std::hash::Hasher) {
    unsafe {
        const {
            assert!(std::mem::size_of::<wasm_bindgen::JsValue>() == std::mem::size_of::<u32>());
            assert!(std::mem::align_of::<wasm_bindgen::JsValue>() >= std::mem::align_of::<u32>());
        }

        let handle: u32 = std::mem::transmute_copy(val);
        hasher.write_u32(handle);
    }
}
