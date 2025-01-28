use std::{borrow::Cow, fmt, hash::{Hash, Hasher}, sync::Arc};

use ash::vk;

use crate::generic::Shader;

use super::device::WeakDevice;

struct LibraryInner {
    owner: WeakDevice,
    idx: usize,
}

impl Drop for LibraryInner {
    fn drop(&mut self) {
        self.owner.drop_library(self.idx);
    }
}

#[derive(Clone)]
pub struct Library {
    module: vk::ShaderModule,
    inner: Arc<LibraryInner>,
}

impl Library {
    pub(super) fn new(owner: WeakDevice, module: vk::ShaderModule, idx: usize) -> Self {
        Library {
            module,
            inner: Arc::new(LibraryInner { idx, owner }),
        }
    }

    pub(super) fn module(&self) -> vk::ShaderModule {
        self.module
    }
}

impl PartialEq for Library {
    fn eq(&self, other: &Self) -> bool {
        self.module == other.module && Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for Library {}

impl Hash for Library {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.module.hash(state);
        Arc::as_ptr(&self.inner).hash(state);
    }
}

impl fmt::Debug for Library {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Library")
            .field("module", &self.module)
            .finish()
    }
}

impl crate::traits::Resource for Library {}

#[hidden_trait::expose]
impl crate::traits::Library for Library {
    fn entry<'a>(&self, entry: &'a str) -> Shader<'a> {
        Shader {
            library: self.clone(),
            entry: Cow::Borrowed(entry),
        }
    }
}
