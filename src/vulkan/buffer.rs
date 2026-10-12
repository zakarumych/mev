use super::mapped::{BufferMappedRange, BufferMappedRangeMut};
use core::fmt;
use std::{
    hash::{Hash, Hasher},
    mem::size_of,
    num::NonZero,
    ops::Range,
    ptr::NonNull,
    sync::Arc,
};

use ash::vk;
use gpu_alloc::MemoryBlock;
use mev_proc::hidden_trait;

use crate::{
    DeviceError,
    backend::handle_host_oom,
    generic::{ArgumentKind, Automatic, BufferUsage, Storage, Uniform},
};

use super::{
    arguments::ArgumentsField,
    device::{DeviceMemory, DeviceOwned, WeakDevice},
    refs::Refs,
};

struct Mapped {
    initial: bool,
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
        if f.alternate() {
            f.debug_struct("Buffer")
                .field("handle", &self.handle)
                .field("size", &self.inner.size)
                .field("usage", &self.inner.usage)
                .finish()
        } else {
            f.debug_tuple("Buffer").field(&self.handle).finish()
        }
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
    pub(super) fn map_initial(&mut self) -> Result<(), DeviceError> {
        self.map_impl(.., true)
    }

    fn map_impl<R>(&mut self, range: R, initial: bool) -> Result<(), DeviceError>
    where
        R: crate::generic::BufferRange,
    {
        assert!(
            initial
                || self
                    .usage()
                    .intersects(BufferUsage::HOST_READ | BufferUsage::HOST_WRITE),
            "Buffer requires host usage for mapping"
        );
        let range = range.range(self.size());
        assert!(
            range.start <= range.end && range.end <= self.size(),
            "Buffer range out of bounds"
        );
        assert!(
            initial || range.start != range.end,
            "Cannot map an empty range after creation"
        );

        let inner = self.detatched_inner();

        let Some(block) = &mut inner.block else {
            return Err(DeviceError::OutOfMemory);
        };

        let Some(device) = inner.owner.upgrade() else {
            return Err(DeviceError::DeviceLost);
        };

        assert!(
            inner.mapped.is_none(),
            "Buffer is already mapped, cannot map it again"
        );

        let result = unsafe {
            block.map(
                device.inner(),
                range.start as u64,
                (range.end - range.start).max(1),
            )
        };

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

        inner.mapped = Some(Mapped {
            initial,
            ptr,
            offset: range.start,
            size: range.end - range.start,
        });

        Ok(())
    }

    #[inline]
    pub(super) fn new(
        owner: WeakDevice,
        handle: vk::Buffer,
        size: usize,
        usage: BufferUsage,
        block: MemoryBlock<DeviceMemory>,
        idx: usize,
    ) -> Self {
        Buffer {
            handle,
            inner: Arc::new(Inner {
                owner,
                size,
                usage,
                block: Some(block),
                mapped: None,
                idx,
            }),
        }
    }

    fn detatched_inner(&mut self) -> &mut Inner {
        Arc::get_mut(&mut self.inner).expect("Buffer must be detached to write to it")
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
                mapped: None,
                idx: 0,
            }),
        }
    }

    #[inline(always)]
    pub(super) fn handle(&self) -> vk::Buffer {
        self.handle
    }

    fn invalidate_range(&mut self, offset: usize, size: usize) -> Result<(), DeviceError> {
        let inner = self.detatched_inner();

        let Some(block) = &mut inner.block else {
            return Err(DeviceError::OutOfMemory);
        };

        let Some(device) = inner.owner.upgrade() else {
            return Err(DeviceError::DeviceLost);
        };

        let result = unsafe { block.invalidate_range(device.inner(), offset as u64, size as u64) };

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
                Err(DeviceError::OutOfMemory)
            }
            _ => unreachable!(),
        }
    }

    pub(crate) fn flush_range(&mut self, offset: usize, size: usize) -> Result<(), DeviceError> {
        if size == 0 {
            return Ok(());
        }
        let inner = self.detatched_inner();

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
                Err(DeviceError::OutOfMemory)
            }
            _ => unreachable!(),
        }
    }
}

impl crate::traits::Resource for Buffer {}

#[hidden_trait]
impl crate::traits::Buffer for Buffer {
    /// Returns buffer slice with given range.
    fn slice<R: crate::BufferRange>(&self, range: R) -> crate::BufferSlice<'_> {
        crate::BufferSlice::from(self).slice(range)
    }

    /// Splits buffer into two ranges, from start to `at` and from `at` to end.
    fn split_at(&self, at: usize) -> (crate::BufferSlice<'_>, crate::BufferSlice<'_>) {
        crate::BufferSlice::from(self).split_at(at)
    }
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
    fn map<R>(&mut self, range: R) -> Result<(), DeviceError>
    where
        R: crate::generic::BufferRange,
    {
        self.map_impl(range, false)
    }

    fn unmap(&mut self) {
        let inner = self.detatched_inner();

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

    fn read_mapped_range<R>(&mut self, range: R) -> Result<BufferMappedRange<'_>, DeviceError>
    where
        R: crate::generic::BufferRange,
    {
        assert!(
            self.usage().contains(BufferUsage::HOST_READ),
            "Buffer requires HOST_READ"
        );
        let range = range.range(self.size());

        assert!(range.start <= range.end, "Invalid range");

        let inner = self.detatched_inner();

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
            if range.start == range.end {
                Ok(())
            } else {
                block.invalidate_range(
                    device.inner(),
                    range.start as u64,
                    (range.end - range.start) as u64,
                )
            }
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

    fn write_mapped_range<R>(&mut self, range: R) -> Result<BufferMappedRangeMut<'_>, DeviceError>
    where
        R: crate::generic::BufferRange,
    {
        let range = range.range(self.size());

        assert!(range.start <= range.end, "Invalid range");

        let inner = self.detatched_inner();

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
            mapped.initial || inner.usage.contains(BufferUsage::HOST_WRITE),
            "Buffer requires HOST_WRITE after its initial mapping"
        );

        assert!(
            mapped.offset <= range.start,
            "Range start is out of mapped region"
        );

        assert!(
            mapped.offset + mapped.size >= range.end,
            "Range end is out of mapped region"
        );

        let result = unsafe {
            if range.start == range.end {
                Ok(())
            } else {
                block.invalidate_range(
                    device.inner(),
                    range.start as u64,
                    (range.end - range.start) as u64,
                )
            }
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
        assert!(
            self.usage().contains(BufferUsage::HOST_WRITE),
            "Buffer requires HOST_WRITE"
        );
        let inner = self.detatched_inner();

        let Some(device) = inner.owner.upgrade() else {
            return Err(DeviceError::DeviceLost);
        };

        let Some(block) = &mut inner.block else {
            return Err(DeviceError::OutOfMemory);
        };

        assert!(
            inner.mapped.is_none(),
            "Buffer is already mapped, cannot map it again"
        );

        let result = unsafe { block.map(device.inner(), offset as u64, data.len()) };

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

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.as_ptr(), data.len());
        }

        let result = unsafe { block.flush_range(device.inner(), offset as u64, data.len() as u64) };

        unsafe {
            block.unmap(device.inner());
        }

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
                Err(DeviceError::OutOfMemory)
            }
            _ => unreachable!(),
        }
    }

    #[inline]
    fn read(&mut self, offset: usize, data: &mut [u8]) -> Result<(), DeviceError> {
        assert!(
            self.usage().contains(BufferUsage::HOST_READ),
            "Buffer requires HOST_READ"
        );
        let inner = self.detatched_inner();

        let Some(device) = inner.owner.upgrade() else {
            return Err(DeviceError::DeviceLost);
        };

        let Some(block) = &mut inner.block else {
            return Err(DeviceError::OutOfMemory);
        };

        assert!(
            inner.mapped.is_none(),
            "Buffer is already mapped, cannot map it again"
        );

        let result = unsafe { block.map(device.inner(), offset as u64, data.len()) };

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

        let result =
            unsafe { block.invalidate_range(device.inner(), offset as u64, data.len() as u64) };

        if result.is_ok() {
            unsafe {
                std::ptr::copy_nonoverlapping(ptr.as_ptr(), data.as_mut_ptr(), data.len());
            }
        }

        unsafe {
            block.unmap(device.inner());
        }

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
                Err(DeviceError::OutOfMemory)
            }
            _ => unreachable!(),
        }
    }
}

impl ArgumentsField<Automatic> for Buffer {
    const KIND: ArgumentKind = ArgumentKind::UniformBuffer;
    const SIZE: Option<NonZero<usize>> = None;
    const OFFSET: usize = 0;
    const STRIDE: usize = size_of::<vk::DescriptorBufferInfo>();

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
    const SIZE: Option<NonZero<usize>> = None;
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
    const SIZE: Option<NonZero<usize>> = None;
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

#[cfg(test)]
mod upload_tests {
    use crate::{
        BufferDesc, BufferInitDesc, BufferUsage, DeviceDesc, Features, InitMapped, Instance,
        PipelineStages, QueueFlags,
    };

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn mapped_creation_preserves_usage_and_initializes_contents() {
        let instance = Instance::load().expect("Vulkan instance");
        let (idx, family) = instance
            .capabilities()
            .devices
            .iter()
            .enumerate()
            .find_map(|(idx, caps)| {
                caps.families
                    .iter()
                    .position(|family| family.queue_flags.contains(QueueFlags::TRANSFER))
                    .map(|family| (idx, family as u32))
            })
            .expect("Vulkan transfer queue");
        let features = instance.capabilities().devices[idx].features;
        assert!(features.contains(
            Features::BUFFER_MAP
                | Features::BUFFER_MAP_AT_CREATION
                | Features::BUFFER_WRITE_COMMAND
        ));
        assert!(!features.contains(Features::BUFFER_WRITE_QUEUE));
        let (device, mut queues) = instance
            .new_device(DeviceDesc {
                idx,
                queues: &[family],
                features: Features::BUFFER_MAP | Features::BUFFER_MAP_AT_CREATION,
            })
            .expect("Vulkan device");
        let mut queue = queues.pop().expect("requested queue");
        let usage = BufferUsage::TRANSFER_SRC;
        let mut mapped = device.new_buffer_mapped(BufferDesc {
            size: 4,
            usage,
            name: "mapped source",
        });
        assert_eq!(mapped.usage(), usage);
        {
            let mut view = mapped.write_mapped_range(..).expect("initial mapping");
            view.write(0, &[1, 2, 3, 4]);
            view.flush().expect("flush initial writes");
        }
        mapped.unmap();
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| mapped.map(..))).is_err());
        let mut readback = device.new_buffer(BufferDesc {
            size: 4,
            usage: BufferUsage::TRANSFER_DST | BufferUsage::HOST_READ,
            name: "readback",
        });
        let mut encoder = queue.new_command_encoder();
        encoder.copy().copy_buffer_to_buffer(&mapped, &readback);
        queue
            .submit_checkpoint([encoder.finish()])
            .expect("submit copy");
        queue.wait_idle().expect("wait for copy");
        drop(queue);
        let mut bytes = [0; 4];
        readback.read(0, &mut bytes).expect("readback contents");
        assert_eq!(bytes, [1, 2, 3, 4]);
        for size in [0, 3, 4, 17] {
            let expected = vec![0x39u8; size];
            let mut buffer = device.new_buffer_init(BufferInitDesc {
                data: expected.as_slice(),
                usage: BufferUsage::HOST_READ | BufferUsage::TRANSFER_SRC,
                name: "byte initializer",
            });
            assert_eq!(
                buffer.usage(),
                BufferUsage::HOST_READ | BufferUsage::TRANSFER_SRC
            );
            assert_eq!(buffer.size(), size);
            if size != 0 {
                let mut actual = vec![0; size];
                buffer
                    .read(0, &mut actual)
                    .expect("read byte initialization");
                assert_eq!(actual, expected);
            }
            let mut buffer = device.new_buffer_init(BufferInitDesc {
                data: InitMapped {
                    size,
                    write: |bytes: &mut [u8]| bytes.fill(0x57),
                },
                usage: BufferUsage::HOST_READ | BufferUsage::TRANSFER_SRC,
                name: "closure initializer",
            });
            assert_eq!(
                buffer.usage(),
                BufferUsage::HOST_READ | BufferUsage::TRANSFER_SRC
            );
            if size != 0 {
                buffer.map(..).expect("later read mapping");
                let mapped = buffer.read_mapped_range(..).expect("read mapped contents");
                assert_eq!(mapped.as_ref(), vec![0x57; size]);
                drop(mapped);
                buffer.unmap();
            }
        }
    }
}
