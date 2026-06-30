use web_sys::{WebGl2RenderingContext as GL, WebGlBuffer};

use crate::{
    generic::{ArgumentGroupLayout, ArgumentKind, ArgumentsSealed},
    ComputeCommandEncoder,
};

use super::command::RenderCommandEncoder;

pub trait Arguments: 'static {
    const LAYOUT: ArgumentGroupLayout<'static>;

    fn bind_render(&self, group: u32, encoder: &mut RenderCommandEncoder);
}

impl<T> ArgumentsSealed for T where T: Arguments {}
impl<T> crate::generic::Arguments for T
where
    T: Arguments,
{
    const LAYOUT: ArgumentGroupLayout<'static> = T::LAYOUT;

    #[inline(always)]
    fn bind_render(&self, group: u32, encoder: &mut RenderCommandEncoder) {
        Arguments::bind_render(self, group, encoder)
    }

    #[inline(always)]
    fn bind_compute(&self, group: u32, encoder: &mut ComputeCommandEncoder) {
        unimplemented!("WebGL does not support compute shaders");
    }
}

fn non_zero_group_no_bindings() -> ! {
    panic!(
        "Attempt to bind non-zero group to a pipeline stage with shader compiled from Metal Shading Language.
        This use-case is not supported right now."
    );
}

#[doc(hidden)]
pub trait ArgumentsField<T>: 'static {
    const KIND: ArgumentKind;
    const SIZE: usize;

    fn bind_vertex(&self, slot: u32, encoder: &mut RenderCommandEncoder);
    fn bind_fragment(&self, slot: u32, encoder: &mut RenderCommandEncoder);
}

impl<T, F> crate::generic::ArgumentsField<T> for F
where
    T: ArgumentsSealed,
    F: ArgumentsField<T> + ArgumentsSealed,
{
    const KIND: ArgumentKind = F::KIND;
    const SIZE: usize = F::SIZE;
}
