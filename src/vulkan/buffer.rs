use core::fmt;
use std::{
    hash::{Hash, Hasher},
    mem::size_of,
    ops::Range,
    ptr::NonNull,
    sync::Arc,
};

use ash::vk;
use gpu_alloc::MemoryBlock;

use crate::{
    backend::handle_host_oom,
    generic::{ArgumentKind, Automatic, BufferUsage, Storage, Uniform},
    BufferMappedRange, BufferMappedRangeMut, DeviceError,
};

use super::{
    arguments::ArgumentsField,
    device::{DeviceMemory, DeviceOwned, WeakDevice},
    refs::Refs,
};

struct Mapped {
    ptr: NonNull<u8>,
    offset: usize,
    size: usize,
}

unsafe impl Send for Mapped {}
unsafe impl Sync for Mapped {}

struct Inner {
    owner: WeakDevice,
    size: usize,
    usage: BufferUsage,
    name: Box<str>,
    block: Option<MemoryBlock<DeviceMemory>>,
    mapped: Option<Mapped>,
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
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
    }
}

impl fmt::Debug for Buffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Buffer")
            .field("handle", &self.handle)
            .finish()
    }
}

impl Drop for Inner {
    #[inline]
    fn drop(&mut self) {
        if let Some(block) = self.block.take() {
            self.owner.drop_buffer(self.idx, block);
        }
    }
}

impl DeviceOwned for Buffer {
    #[inline(always)]
    fn owner(&self) -> &WeakDevice {
        &self.inner.owner
    }
}

impl Buffer {
    #[inline]
    pub(super) fn new(
        owner: WeakDevice,
        handle: vk::Buffer,
        size: usize,
        usage: BufferUsage,
        name: Box<str>,
        block: MemoryBlock<DeviceMemory>,
        idx: usize,
    ) -> Self {
        Buffer {
            handle,
            inner: Arc::new(Inner {
                owner,
                size,
                usage,
                name,
                block: Some(block),
                mapped: None,
                idx,
            }),
        }
    }

    /// Creates a null/invalid Buffer for use when device OOM occurs.
    pub(super) fn null(size: usize, usage: BufferUsage, name: Box<str>) -> Self {
        Buffer {
            handle: vk::Buffer::null(),
            inner: Arc::new(Inner {
                owner: WeakDevice::null(),
                size,
                usage,
                name,
                block: None,
                mapped: None,
                idx: 0,
            }),
        }
    }

    #[inline(always)]
    pub fn handle(&self) -> vk::Buffer {
        self.handle
    }

    pub(crate) fn flush_range(&mut self, offset: usize, size: usize) -> Result<(), DeviceError> {
        let inner =
            Arc::get_mut(&mut self.inner).expect("Buffer must be detached to invalidate it");

        let Some(block) = &mut inner.block else {
            return Err(DeviceError::OutOfMemory);
        };

        let Some(device) = inner.owner.upgrade() else {
            return Err(DeviceError::DeviceLost);
        };

        let result = unsafe { block.flush_range(device.inner(), offset as u64, size as u64) };

        match result {
            Ok(_) => Ok(()),
            Err(gpu_alloc::MapError::NonHostVisible) => {
                panic!("Buffer is not host visible, cannot invalidate it");
            }
            Err(gpu_alloc::MapError::OutOfHostMemory) => {
                handle_host_oom();
            }
            Err(gpu_alloc::MapError::OutOfDeviceMemory) => {
                device.set_oom();
                return Err(DeviceError::OutOfMemory);
            }
            _ => unreachable!(),
        }
    }
}

impl crate::traits::Resource for Buffer {}

#[hidden_trait::expose]
impl crate::traits::Buffer for Buffer {
    /// Returns the buffer size in bytes.
    #[inline(always)]
    fn size(&self) -> usize {
        self.inner.size
    }

    /// Returns the buffer usage.
    #[inline(always)]
    fn usage(&self) -> crate::generic::BufferUsage {
        self.inner.usage
    }

    /// Returns the name of the buffer.
    #[inline(always)]
    fn name(&self) -> &str {
        &*self.inner.name
    }

    /// Returns `true` if the buffer is not shared,
    /// meaning that there are no other references to the buffer
    /// including references that tracks that GPU may be using the buffer.
    ///
    /// If this method returns `true` then it is safe to write to or read from the buffer from host.
    #[inline(always)]
    fn detached(&self) -> bool {
        debug_assert_eq!(Arc::weak_count(&self.inner), 0, "No weak refs allowed");
        Arc::strong_count(&self.inner) == 1
    }

    /// Maps the buffer region for host access.
    ///
    /// This is no-op for persistently mapped buffers.
    /// Buffers created with [`Memory::Upload`](crate::generic::Memory::Upload) or [`Memory::Download`](crate::generic::Memory::Download) are persistently mapped.
    fn map(&mut self, range: Range<usize>) -> Result<(), DeviceError> {
        let inner = Arc::get_mut(&mut self.inner).expect("Buffer must be detached to write to it");

        let Some(block) = &mut inner.block else {
            return Err(DeviceError::OutOfMemory);
        };

        let Some(device) = inner.owner.upgrade() else {
            return Err(DeviceError::DeviceLost);
        };

        let result =
            unsafe { block.map(device.inner(), range.start as u64, range.end - range.start) };

        let ptr = match result {
            Ok(ptr) => ptr,
            Err(gpu_alloc::MapError::AlreadyMapped) => {
                panic!("Buffer is already mapped, cannot map it again");
            }
            Err(gpu_alloc::MapError::NonHostVisible) => {
                panic!("Buffer is not host visible, cannot map it");
            }
            Err(gpu_alloc::MapError::MapFailed) => {
                handle_host_oom();
            }
            Err(gpu_alloc::MapError::OutOfHostMemory) => {
                handle_host_oom();
            }
            Err(gpu_alloc::MapError::OutOfDeviceMemory) => {
                device.set_oom();
                return Err(DeviceError::OutOfMemory);
            }
        };

        debug_assert!(
            inner.mapped.is_none(),
            "Buffer is already mapped, cannot map it again"
        );

        inner.mapped = Some(Mapped {
            ptr,
            offset: range.start,
            size: range.end - range.start,
        });

        Ok(())
    }

    fn unmap(&mut self) {
        let inner = Arc::get_mut(&mut self.inner).expect("Buffer must be detached to unmap it");

        let _mapped = inner
            .mapped
            .take()
            .expect("Buffer is not mapped, cannot unmap it");

        let Some(block) = &mut inner.block else {
            unreachable!()
        };

        let Some(device) = inner.owner.upgrade() else {
            return;
        };

        unsafe {
            block.unmap(device.inner());
        }
    }

    fn read_mapped_range(
        &mut self,
        range: Range<usize>,
    ) -> Result<BufferMappedRange<'_>, DeviceError> {
        assert!(
            range.start <= range.end,
            "Range start must be less than or equal to range end"
        );

        let inner = Arc::get_mut(&mut self.inner).expect("Buffer must be detached to write to it");

        let Some(device) = inner.owner.upgrade() else {
            return Err(DeviceError::DeviceLost);
        };

        let Some(block) = &mut inner.block else {
            return Err(DeviceError::OutOfMemory);
        };

        let Some(mapped) = &inner.mapped else {
            panic!("Buffer is not mapped, cannot read from it");
        };

        assert!(
            mapped.offset <= range.start,
            "Range start is out of mapped region"
        );

        assert!(
            mapped.offset + mapped.size >= range.end,
            "Range end is out of mapped region"
        );

        let result = unsafe {
            block.invalidate_range(
                device.inner(),
                range.start as u64,
                (range.end - range.start) as u64,
            )
        };

        match result {
            Ok(_) => {}
            Err(gpu_alloc::MapError::NonHostVisible) => {
                panic!("Buffer is not host visible, cannot invalidate it");
            }
            Err(gpu_alloc::MapError::OutOfHostMemory) => {
                handle_host_oom();
            }
            Err(gpu_alloc::MapError::OutOfDeviceMemory) => {
                device.set_oom();
                return Err(DeviceError::OutOfMemory);
            }
            _ => unreachable!(),
        }

        let ptr = unsafe { mapped.ptr.add(range.start - mapped.offset) };

        Ok(BufferMappedRange::new(
            self,
            ptr,
            range.start,
            range.end - range.start,
        ))
    }

    fn write_mapped_range(
        &mut self,
        range: Range<usize>,
    ) -> Result<BufferMappedRangeMut<'_>, DeviceError> {
        assert!(
            range.start <= range.end,
            "Range start must be less than or equal to range end"
        );

        let inner = Arc::get_mut(&mut self.inner).expect("Buffer must be detached to write to it");

        let Some(device) = inner.owner.upgrade() else {
            return Err(DeviceError::DeviceLost);
        };

        let Some(block) = &mut inner.block else {
            return Err(DeviceError::OutOfMemory);
        };

        let Some(mapped) = &inner.mapped else {
            panic!("Buffer is not mapped, cannot read from it");
        };

        assert!(
            mapped.offset <= range.start,
            "Range start is out of mapped region"
        );

        assert!(
            mapped.offset + mapped.size >= range.end,
            "Range end is out of mapped region"
        );

        let result = unsafe {
            block.invalidate_range(
                device.inner(),
                range.start as u64,
                (range.end - range.start) as u64,
            )
        };

        match result {
            Ok(_) => {}
            Err(gpu_alloc::MapError::NonHostVisible) => {
                panic!("Buffer is not host visible, cannot invalidate it");
            }
            Err(gpu_alloc::MapError::OutOfHostMemory) => {
                handle_host_oom();
            }
            Err(gpu_alloc::MapError::OutOfDeviceMemory) => {
                device.set_oom();
                return Err(DeviceError::OutOfMemory);
            }
            _ => unreachable!(),
        }

        let ptr = unsafe { mapped.ptr.add(range.start - mapped.offset) };

        Ok(BufferMappedRangeMut::new(
            self,
            ptr,
            range.start,
            range.end - range.start,
        ))
    }

    #[inline]
    fn write(&mut self, offset: usize, data: &[u8]) -> Result<(), DeviceError> {
        let range = offset..offset + data.len();
        self.map(range.clone())?;
        let mut mapped = self.write_mapped_range(range)?;
        mapped.write(0, data);
        mapped.flush()
    }

    #[inline]
    fn read(&mut self, offset: usize, data: &mut [u8]) -> Result<(), DeviceError> {
        let range = offset..offset + data.len();
        self.map(range.clone())?;
        let mapped = self.read_mapped_range(range)?;
        mapped.read(0, data);
        Ok(())
    }
}

impl ArgumentsField<Automatic> for Buffer {
    const KIND: ArgumentKind = <Self as ArgumentsField<Uniform>>::KIND;
    const SIZE: usize = <Self as ArgumentsField<Uniform>>::SIZE;
    const OFFSET: usize = <Self as ArgumentsField<Uniform>>::OFFSET;
    const STRIDE: usize = <Self as ArgumentsField<Uniform>>::STRIDE;

    type Update = <Self as ArgumentsField<Uniform>>::Update;

    #[inline(always)]
    fn update(&self) -> <Self as ArgumentsField<Uniform>>::Update {
        <Self as ArgumentsField<Uniform>>::update(self)
    }

    #[inline(always)]
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

    #[inline(always)]
    fn update(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo {
            buffer: self.handle,
            offset: 0,
            range: self.inner.size as u64,
        }
    }

    #[inline(always)]
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

    #[inline(always)]
    fn update(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo {
            buffer: self.handle,
            offset: 0,
            range: self.inner.size as u64,
        }
    }

    #[inline(always)]
    fn add_refs(&self, refs: &mut Refs) {
        refs.add_buffer(self.clone());
    }
}
