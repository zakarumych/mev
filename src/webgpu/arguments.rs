use crate::generic::{ArgumentGroupLayout, ArgumentKind, ArgumentsSealed};

use super::{ComputeCommandEncoder, RenderCommandEncoder};

pub trait Arguments: 'static {
    const LAYOUT: ArgumentGroupLayout<'static>;

    fn bind_render(&self, group: u32, encoder: &mut RenderCommandEncoder);
    fn bind_compute(&self, group: u32, encoder: &mut ComputeCommandEncoder);
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
        Arguments::bind_compute(self, group, encoder)
    }
}

#[doc(hidden)]
pub trait ArgumentsField<T>: 'static {
    const KIND: ArgumentKind;
    const SIZE: usize;

    fn as_binding_resource(&self) -> wgpu::BindingResource<'_>;
}

impl<T, F> crate::generic::ArgumentsField<T> for F
where
    T: ArgumentsSealed,
    F: ArgumentsField<T> + ArgumentsSealed,
{
    const KIND: ArgumentKind = F::KIND;
    const SIZE: usize = F::SIZE;
}
