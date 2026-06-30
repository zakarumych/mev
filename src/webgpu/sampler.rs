use std::{fmt, rc::Rc};

use crate::generic::SamplerDesc;

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
