use std::{fmt, num::NonZero, rc::Rc};

use crate::{ArgumentKind, Automatic, generic::SamplerDesc};

use super::arguments::ArgumentsField;

#[derive(Clone)]
pub struct Sampler {
    inner: wgpu::Sampler,
}

impl Sampler {
    pub(super) fn new(sampler: wgpu::Sampler) -> Self {
        Sampler { inner: sampler }
    }

    pub(super) fn wgpu(&self) -> &wgpu::Sampler {
        &self.inner
    }
}

impl fmt::Debug for Sampler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Sampler").finish()
    }
}

impl crate::traits::Resource for Sampler {}

impl ArgumentsField<Automatic> for Sampler {
    const KIND: ArgumentKind = ArgumentKind::Sampler;
    const SIZE: Option<NonZero<usize>> = None;

    fn as_binding_resource(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Sampler(&self.inner)
    }
}
