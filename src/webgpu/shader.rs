use std::{fmt, rc::Rc};

use crate::generic::{Shader, ShaderLanguage, ShaderLibraryError};

#[derive(Clone)]
pub struct Library {
    module: Rc<wgpu::ShaderModule>,
}

impl Library {
    pub(super) fn new(module: wgpu::ShaderModule) -> Self {
        Library {
            module: Rc::new(module),
        }
    }

    pub(super) fn wgpu(&self) -> &wgpu::ShaderModule {
        &self.module
    }
}

impl fmt::Debug for Library {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Library").finish()
    }
}

impl crate::traits::Resource for Library {}

#[hidden_trait::expose]
impl crate::traits::Library for Library {
    fn entry<'a>(&self, entry: &'a str) -> Shader<'a> {
        Shader {
            library: self.clone(),
            entry: entry.into(),
        }
    }
}
