use std::{borrow::Cow, fmt, sync::Arc};

use hashbrown::HashMap;

use crate::generic::{Shader, ShaderCompileError};

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub struct GroupBindings {
    pub bindings: [u8; 64],
}

impl GroupBindings {
    const INVALID: Self = GroupBindings {
        bindings: [0xff; 64],
    };
}

pub struct Bindings {
    pub groups: [GroupBindings; 16],
    pub push_constants: Option<u8>,
}

impl Bindings {
    #[inline(always)]
    pub fn new() -> Self {
        Bindings {
            groups: [GroupBindings::INVALID; 16],
            push_constants: None,
        }
    }

    #[inline(always)]
    pub fn insert(&mut self, binding: naga::ResourceBinding, slot: u8) {
        self.groups[binding.group as usize].bindings[binding.binding as usize] = slot;
    }

    #[inline(always)]
    pub fn set_push_constants(&mut self, slot: u8) {
        self.push_constants = Some(slot);
    }
}

#[derive(Clone)]
pub(super) struct EntryPointData {
    pub bindings: Arc<Bindings>,
    pub workgroup_size: [u32; 3],
    pub name: Result<String, naga::back::msl::EntryPointError>,
}

#[derive(Clone)]
pub struct Library {
    library: metal::Library,
    entry_point_data: HashMap<String, EntryPointData>,
}

impl Library {
    #[cfg_attr(feature = "inline-more", inline)]
    pub(super) fn new(library: metal::Library) -> Self {
        Library {
            library,
            entry_point_data: HashMap::new(),
        }
    }

    #[cfg_attr(feature = "inline-more", inline)]
    pub(super) fn with_entry_point_data(
        library: metal::Library,
        entry_point_data: HashMap<String, EntryPointData>,
    ) -> Self {
        Library {
            library,
            entry_point_data,
        }
    }

    #[cfg_attr(feature = "inline-more", inline)]
    pub(super) fn get_function(&self, entry: &str) -> Option<metal::Function> {
        match self.entry_point_data.get(entry) {
            Some(ep) => match &ep.name {
                Ok(name) => self.library.get_function(name, None).ok(),
                Err(_) => None,
            },
            None => self.library.get_function(entry, None).ok(),
        }
    }

    #[cfg_attr(feature = "inline-more", inline)]
    pub(super) fn get_bindings(&self, entry: &str) -> Option<Arc<Bindings>> {
        let ep = self.entry_point_data.get(entry)?;
        Some(ep.bindings.clone())
    }

    #[cfg_attr(feature = "inline-more", inline)]
    pub(super) fn get_workgroup_size(&self, entry: &str) -> Option<[u32; 3]> {
        let ep = self.entry_point_data.get(entry)?;
        Some(ep.workgroup_size)
    }
}

#[hidden_trait::expose]
impl crate::traits::Library for Library {
    fn entry<'a>(&self, entry: &'a str) -> Shader<'a> {
        Shader {
            library: self.clone(),
            entry: Cow::Borrowed(entry),
        }
    }
}
