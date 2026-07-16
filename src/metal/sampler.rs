use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSUInteger;
use objc2_metal::{MTLComputeCommandEncoder, MTLRenderCommandEncoder, MTLSamplerState};

use crate::generic::{ArgumentKind, Automatic};

use super::arguments::ArgumentsField;

#[derive(Clone)]
pub struct Sampler {
    sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
}

impl Sampler {
    pub(super) fn new(sampler: Retained<ProtocolObject<dyn MTLSamplerState>>) -> Self {
        Self { sampler }
    }
}

impl ArgumentsField<Automatic> for Sampler {
    const KIND: ArgumentKind = ArgumentKind::Sampler;
    const SIZE: usize = 1;

    fn bind_vertex(&self, slot: u32, encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>) {
        unsafe {
            encoder.setVertexSamplerState_atIndex(Some(&self.sampler), slot as NSUInteger);
        }
    }
    #[inline(always)]
    fn bind_fragment(&self, slot: u32, encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>) {
        unsafe {
            encoder.setFragmentSamplerState_atIndex(Some(&self.sampler), slot as NSUInteger);
        }
    }

    #[inline(always)]
    fn bind_compute(&self, slot: u32, encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
        unsafe {
            encoder.setSamplerState_atIndex(Some(&self.sampler), slot as NSUInteger);
        }
    }
}
