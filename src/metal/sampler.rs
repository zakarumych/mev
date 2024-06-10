use crate::generic::{ArgumentKind, Automatic};

use super::arguments::ArgumentsField;

#[derive(Clone)]
pub struct Sampler {
    sampler: metal::SamplerState,
}

impl Sampler {
    pub(super) fn new(sampler: metal::SamplerState) -> Self {
        Self { sampler }
    }
}

impl ArgumentsField<Automatic> for Sampler {
    const KIND: ArgumentKind = ArgumentKind::Sampler;
    const SIZE: usize = 1;

    fn bind_vertex(&self, slot: u32, encoder: &metal::RenderCommandEncoderRef) {
        encoder.set_vertex_sampler_state(slot.into(), Some(&self.sampler));
    }
    #[inline(always)]
    fn bind_fragment(&self, slot: u32, encoder: &metal::RenderCommandEncoderRef) {
        encoder.set_fragment_sampler_state(slot.into(), Some(&self.sampler));
    }

    #[inline(always)]
    fn bind_compute(&self, slot: u32, encoder: &metal::ComputeCommandEncoderRef) {
        encoder.set_sampler_state(slot.into(), Some(&self.sampler));
    }
}
