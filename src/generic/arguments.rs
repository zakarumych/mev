use crate::backend::{ComputeCommandEncoder, RenderCommandEncoder};

use super::ShaderStages;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArgumentKind {
    // Constant,
    UniformBuffer,
    StorageBuffer,
    SampledImage,
    StorageImage,
    Sampler,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ArgumentLayout {
    pub kind: ArgumentKind,
    pub size: usize,
    pub stages: ShaderStages,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ArgumentGroupLayout<'a> {
    pub arguments: &'a [ArgumentLayout],
}

/// This is not a part of public API.
/// It is only public because it is used in the `mev` macro.
#[doc(hidden)]
pub trait ArgumentsSealed {}

pub trait Arguments: ArgumentsSealed + 'static {
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

pub trait ArgumentsField<T: ArgumentsSealed>: ArgumentsSealed {
    const KIND: ArgumentKind;
    const SIZE: usize;
}
