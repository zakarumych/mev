use crate::generic::{ArgumentGroupLayout, ArgumentKind, ArgumentsSealed};

use super::{shader::Bindings, ComputeCommandEncoder, RenderCommandEncoder};

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

    fn bind_vertex(&self, slot: u32, encoder: &metal::RenderCommandEncoderRef);
    fn bind_fragment(&self, slot: u32, encoder: &metal::RenderCommandEncoderRef);
    fn bind_compute(&self, slot: u32, encoder: &metal::ComputeCommandEncoderRef);

    #[inline]
    fn bind_vertex_argument(
        &self,
        group: u32,
        index: u32,
        bindings: Option<&Bindings>,
        encoder: &metal::RenderCommandEncoderRef,
    ) {
        match bindings {
            Some(bindings) => {
                let slot = bindings.groups[group as usize].bindings[index as usize];
                self.bind_vertex(slot.into(), encoder);
            }
            None if group == 0 => self.bind_vertex(index, encoder),
            None => non_zero_group_no_bindings(),
        }
    }

    #[inline]
    fn bind_fragment_argument(
        &self,
        group: u32,
        index: u32,
        bindings: Option<&Bindings>,
        encoder: &metal::RenderCommandEncoderRef,
    ) {
        match bindings {
            Some(bindings) => {
                let slot = bindings.groups[group as usize].bindings[index as usize];
                self.bind_fragment(slot.into(), encoder);
            }
            None if group == 0 => self.bind_fragment(index, encoder),
            None => non_zero_group_no_bindings(),
        }
    }

    #[inline]
    fn bind_compute_argument(
        &self,
        group: u32,
        index: u32,
        bindings: Option<&Bindings>,
        encoder: &metal::ComputeCommandEncoderRef,
    ) {
        match bindings {
            Some(bindings) => {
                let slot = bindings.groups[group as usize].bindings[index as usize];
                self.bind_compute(slot.into(), encoder);
            }
            None if group == 0 => self.bind_compute(index, encoder),
            None => non_zero_group_no_bindings(),
        }
    }
}

impl<T, F> crate::generic::ArgumentsField<T> for F
where
    T: ArgumentsSealed,
    F: ArgumentsField<T> + ArgumentsSealed,
{
    const KIND: ArgumentKind = F::KIND;
    const SIZE: usize = F::SIZE;
}
