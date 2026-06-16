use core::fmt;
use std::{
    hash::{Hash, Hasher},
    mem::{size_of, ManuallyDrop},
    sync::Arc,
};

use ash::vk::{self, Handle};
use gpu_alloc::{MemoryBlock, MemoryPropertyFlags};

use crate::generic::{ArgumentKind, Automatic, BufferUsage, Storage, Uniform};

use super::{
    arguments::ArgumentsField,
    device::{DeviceOwned, WeakDevice},
    refs::Refs,
};

struct Inner {
    owner: WeakDevice,
    size: usize,
    usage: BufferUsage,
    block: Option<MemoryBlock<(vk::DeviceMemory, usize)>>,
    idx: usize,
}

#[derive(Clone)]
pub struct Buffer {
    handle: vk::Buffer,
    inner: Arc<Inner>,
}

impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for Buffer {}

impl Hash for Buffer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        Arc::as_ptr(&self.inner).hash(state);
    }
}

impl fmt::Debug for Buffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Image")
            .field("handle", &self.handle)
            .finish()
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        if let Some(block) = self.block.take() {
            self.owner.drop_buffer(self.idx, block);
        }
    }
}

impl DeviceOwned for Buffer {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn owner(&self) -> &WeakDevice {
        &self.inner.owner
    }
}

impl Buffer {
    pub(super) fn new(
        owner: WeakDevice,
        handle: vk::Buffer,
        size: usize,
        usage: BufferUsage,
        block: MemoryBlock<(vk::DeviceMemory, usize)>,
        idx: usize,
    ) -> Self {
        Buffer {
            handle,
            inner: Arc::new(Inner {
                owner,
                size,
                usage,
                block: Some(block),
                idx,
            }),
        }
    }

    /// Creates a null/invalid Buffer for use when device OOM occurs.
    pub(super) fn null(size: usize, usage: BufferUsage) -> Self {
        Buffer {
            handle: vk::Buffer::null(),
            inner: Arc::new(Inner {
                owner: WeakDevice::null(),
                size,
                usage,
                block: None,
                idx: 0,
            }),
        }
    }

    pub fn is_null(&self) -> bool {
        self.handle.is_null()
    }

    #[cfg_attr(feature = "inline-more", inline(always))]
    pub fn handle(&self) -> vk::Buffer {
        self.handle
    }
}

impl crate::traits::Resource for Buffer {}

#[hidden_trait::expose]
impl crate::traits::Buffer for Buffer {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn size(&self) -> usize {
        self.inner.size
    }

    #[cfg_attr(feature = "inline-more", inline(always))]
    fn detached(&self) -> bool {
        debug_assert_eq!(Arc::weak_count(&self.inner), 0, "No weak refs allowed");
        Arc::strong_count(&self.inner) == 1
    }

    #[cfg_attr(feature = "inline-more", inline(always))]
    unsafe fn write_unchecked(&mut self, offset: usize, data: &[u8]) {
        let inner = Arc::get_mut(&mut self.inner).unwrap();

        let Some(block) = &mut inner.block else {
            return;
        };

        debug_assert!(block.props().contains(MemoryPropertyFlags::HOST_VISIBLE));

        let Some(device) = inner.owner.upgrade() else {
            return;
        };

        unsafe {
            let ptr = block
                .map(device.inner(), offset as u64, data.len())
                .unwrap();
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.as_ptr(), data.len());
        }
    }
}

impl ArgumentsField<Automatic> for Buffer {
    const KIND: ArgumentKind = <Self as ArgumentsField<Uniform>>::KIND;
    const SIZE: usize = <Self as ArgumentsField<Uniform>>::SIZE;
    const OFFSET: usize = <Self as ArgumentsField<Uniform>>::OFFSET;
    const STRIDE: usize = <Self as ArgumentsField<Uniform>>::STRIDE;

    type Update = <Self as ArgumentsField<Uniform>>::Update;

    #[cfg_attr(feature = "inline-more", inline(always))]
    fn update(&self) -> <Self as ArgumentsField<Uniform>>::Update {
        <Self as ArgumentsField<Uniform>>::update(self)
    }

    #[cfg_attr(feature = "inline-more", inline(always))]
    fn add_refs(&self, refs: &mut Refs) {
        refs.add_buffer(self.clone());
    }
}

impl ArgumentsField<Uniform> for Buffer {
    const KIND: ArgumentKind = ArgumentKind::UniformBuffer;
    const SIZE: usize = 1;
    const OFFSET: usize = 0;
    const STRIDE: usize = size_of::<vk::DescriptorBufferInfo>();

    type Update = vk::DescriptorBufferInfo;

    #[cfg_attr(feature = "inline-more", inline(always))]
    fn update(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo {
            buffer: self.handle,
            offset: 0,
            range: self.inner.size as u64,
        }
    }

    #[cfg_attr(feature = "inline-more", inline(always))]
    fn add_refs(&self, refs: &mut Refs) {
        refs.add_buffer(self.clone());
    }
}

impl ArgumentsField<Storage> for Buffer {
    const KIND: ArgumentKind = ArgumentKind::StorageBuffer;
    const SIZE: usize = 1;
    const OFFSET: usize = 0;
    const STRIDE: usize = size_of::<vk::DescriptorBufferInfo>();

    type Update = vk::DescriptorBufferInfo;

    #[cfg_attr(feature = "inline-more", inline(always))]
    fn update(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo {
            buffer: self.handle,
            offset: 0,
            range: self.inner.size as u64,
        }
    }

    #[cfg_attr(feature = "inline-more", inline(always))]
    fn add_refs(&self, refs: &mut Refs) {
        refs.add_buffer(self.clone());
    }
}
