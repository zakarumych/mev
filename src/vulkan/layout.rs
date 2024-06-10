use std::{
    any::TypeId,
    hash::{Hash, Hasher},
    sync::{Arc, Weak},
};

use hashbrown::HashMap;
use parking_lot::Mutex;

use crate::generic::ArgumentLayout;

use super::device::WeakDevice;

#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) struct DescriptorSetLayoutDesc {
    pub arguments: Vec<ArgumentLayout>,
}

struct DescriptorSetLayoutInner {
    owner: WeakDevice,
    desc: DescriptorSetLayoutDesc,
}

impl Drop for DescriptorSetLayoutInner {
    fn drop(&mut self) {
        self.owner.drop_descriptor_set_layout(std::mem::replace(
            &mut self.desc,
            DescriptorSetLayoutDesc {
                arguments: Vec::new(),
            },
        ));
    }
}

#[derive(Clone)]
pub(super) struct WeakDescriptorSetLayout {
    handle: ash::vk::DescriptorSetLayout,
    inner: Weak<DescriptorSetLayoutInner>,
}

impl WeakDescriptorSetLayout {
    pub fn unused(&self) -> bool {
        self.inner.strong_count() == 0
    }

    pub fn upgrade(&self) -> Option<DescriptorSetLayout> {
        let inner = self.inner.upgrade()?;
        Some(DescriptorSetLayout {
            handle: self.handle,
            inner,
        })
    }

    pub fn handle(&self) -> ash::vk::DescriptorSetLayout {
        self.handle
    }
}

#[derive(Clone)]
pub(super) struct DescriptorSetLayout {
    handle: ash::vk::DescriptorSetLayout,
    inner: Arc<DescriptorSetLayoutInner>,
}

impl DescriptorSetLayout {
    pub fn new(
        owner: WeakDevice,
        handle: ash::vk::DescriptorSetLayout,
        desc: DescriptorSetLayoutDesc,
    ) -> Self {
        DescriptorSetLayout {
            handle,
            inner: Arc::new(DescriptorSetLayoutInner { owner, desc }),
        }
    }

    pub fn downgrade(&self) -> WeakDescriptorSetLayout {
        WeakDescriptorSetLayout {
            handle: self.handle,
            inner: Arc::downgrade(&self.inner),
        }
    }

    pub fn handle(&self) -> ash::vk::DescriptorSetLayout {
        self.handle
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) struct PipelineLayoutDesc {
    pub groups: Vec<Vec<ArgumentLayout>>,
    pub constants: usize,
}

struct PipelineLayoutInner {
    set_layouts: Vec<DescriptorSetLayout>,
    owner: WeakDevice,
    desc: PipelineLayoutDesc,
    templates: Mutex<
        HashMap<(TypeId, ash::vk::PipelineBindPoint, u32), ash::vk::DescriptorUpdateTemplate>,
    >,
}

impl Drop for PipelineLayoutInner {
    fn drop(&mut self) {
        let desc = std::mem::replace(
            &mut self.desc,
            PipelineLayoutDesc {
                groups: Vec::new(),
                constants: 0,
            },
        );
        self.owner
            .drop_pipeline_layout(desc, self.templates.get_mut().values().copied());
    }
}

#[derive(Clone)]
pub(super) struct WeakPipelineLayout {
    handle: ash::vk::PipelineLayout,
    inner: Weak<PipelineLayoutInner>,
}

impl PartialEq for WeakPipelineLayout {
    fn eq(&self, other: &Self) -> bool {
        self.inner.ptr_eq(&other.inner)
    }
}

impl Eq for WeakPipelineLayout {}

impl Hash for WeakPipelineLayout {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.as_ptr().hash(state)
    }
}

impl WeakPipelineLayout {
    pub fn unused(&self) -> bool {
        self.inner.strong_count() == 0
    }

    pub fn upgrade(&self) -> Option<PipelineLayout> {
        let inner = self.inner.upgrade()?;
        Some(PipelineLayout {
            handle: self.handle,
            inner,
        })
    }

    pub fn handle(&self) -> ash::vk::PipelineLayout {
        self.handle
    }
}

#[derive(Clone)]
pub(super) struct PipelineLayout {
    handle: ash::vk::PipelineLayout,
    inner: Arc<PipelineLayoutInner>,
}

impl PipelineLayout {
    pub fn new(
        owner: WeakDevice,
        handle: ash::vk::PipelineLayout,
        desc: PipelineLayoutDesc,
        set_layouts: Vec<DescriptorSetLayout>,
    ) -> Self {
        PipelineLayout {
            handle,
            inner: Arc::new(PipelineLayoutInner {
                owner,
                desc,
                templates: Mutex::new(HashMap::new()),
                set_layouts,
            }),
        }
    }

    pub fn downgrade(&self) -> WeakPipelineLayout {
        WeakPipelineLayout {
            handle: self.handle,
            inner: Arc::downgrade(&self.inner),
        }
    }

    pub fn handle(&self) -> ash::vk::PipelineLayout {
        self.handle
    }

    pub fn group_layout(&self, idx: usize) -> &[ArgumentLayout] {
        &self.inner.desc.groups[idx]
    }

    pub fn templates(
        &self,
    ) -> &Mutex<HashMap<(TypeId, ash::vk::PipelineBindPoint, u32), ash::vk::DescriptorUpdateTemplate>>
    {
        &self.inner.templates
    }
}
