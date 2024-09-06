use crate::backend::{ComputeCommandEncoder, RenderCommandEncoder};

use super::ShaderStages;


/// Kind of the shader argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArgumentKind {
    // Constant,

    /// Argument is uniform buffer.
    /// Read-only data that is shared between all invocations of a shader.
    UniformBuffer,

    /// Argument is storage buffer.
    /// Read-write memory that is shared between all invocations of a shader.
    StorageBuffer,

    /// Argument is sampled image.
    /// Read-only image that must be sampled using a sampler.
    SampledImage,

    /// Argument is storage image.
    /// Read-write image that can be used as a texture or a pixel buffer.
    StorageImage,

    /// Argument is sampler.
    /// Read-only object that describes how to sample an image.
    Sampler,
}

/// Layout of the shader argument slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ArgumentLayout {
    /// Kind of the argument.
    pub kind: ArgumentKind,

    /// Number of resources in the argument.
    pub size: usize,

    /// Stages that the argument may be used in.
    pub stages: ShaderStages,
}

/// Layout of the argument group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ArgumentGroupLayout<'a> {
    /// Arguments in the group.
    pub arguments: &'a [ArgumentLayout],
}

/// This is not a part of public API.
/// It is only public because it is used in the `mev` macro.
#[doc(hidden)]
pub trait ArgumentsSealed {}

/// Shader arguments trait.
/// Implemented by types that serve as shader arguments.
/// 
/// Derive this trait for structures where all fields are `ArgumentsField` implementations.
/// It can be buffers, buffer slices, images, samplers, etc.
/// Use attributes to override default argument kind and specify stages.
pub trait Arguments: ArgumentsSealed + 'static {
    /// Layout of the argument group defined by the type.
    const LAYOUT: ArgumentGroupLayout<'static>;

    /// Bind arguments to the command encoder.
    fn bind_render(&self, group: u32, encoder: &mut RenderCommandEncoder);

    /// Bind arguments to the command encoder.
    fn bind_compute(&self, group: u32, encoder: &mut ComputeCommandEncoder);
}

/// Marker type for `Argument` trait.
pub enum Uniform {}

impl ArgumentsSealed for Uniform {}

/// Marker type for `Argument` trait.
pub enum Sampled {}

impl ArgumentsSealed for Sampled {}

/// Marker type for `Argument` trait.
pub enum Storage {}

impl ArgumentsSealed for Storage {}

/// Marker type for `Argument` trait.
pub enum Automatic {}

impl ArgumentsSealed for Automatic {}

/// Trait implemented by types that can be fields in type that derive `Arguments`.
/// This cannot be implemented outside of the crate.
pub trait ArgumentsField<T: ArgumentsSealed>: ArgumentsSealed {
    const KIND: ArgumentKind;
    const SIZE: usize;
}
