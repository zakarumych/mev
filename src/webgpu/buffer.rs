use super::mapped::{BufferMappedRange, BufferMappedRangeMut};
use std::{cell::Cell, fmt, hash, num::NonZero, rc::Rc};

use mev_proc::hidden_trait;

use crate::{
    ArgumentKind, ArgumentsSealed, Automatic, Storage, Uniform,
    backend::{arguments::ArgumentsField, from::WgpuInto},
    generic::{
        BufferDesc, BufferInitDesc, BufferRange,
        BufferUsage, DeviceError,
    },
};

#[derive(Clone)]
pub struct Buffer {
    inner: wgpu::Buffer,
    size: usize,
    usage: BufferUsage,
    mapped: Rc<Cell<bool>>,
}

impl Buffer {
    pub(super) fn new(buffer: wgpu::Buffer, size: usize, usage: BufferUsage, mapped: bool) -> Self {
        Buffer {
            inner: buffer,
            size,
            usage,
            mapped: Rc::new(Cell::new(mapped)),
        }
    }

    pub(super) fn wgpu(&self) -> &wgpu::Buffer {
        &self.inner
    }

    pub(crate) fn flush_range(&mut self, _offset: usize, _size: usize) -> Result<(), DeviceError> {
        Ok(())
    }
}

impl fmt::Debug for Buffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Buffer")
            .field("size", &self.inner.size())
            .field("usage", &self.inner.usage())
            .finish()
    }
}

impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl Eq for Buffer {}

impl hash::Hash for Buffer {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
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
    fn size(&self) -> usize {
        self.size
    }

    fn usage(&self) -> BufferUsage {
        self.usage
    }

    fn detached(&self) -> bool {
        false
    }

    fn map<R>(&mut self, _range: R) -> Result<(), DeviceError>
    where
        R: BufferRange,
    {
        panic!("buffer mapping is not supported in the WebGPU backend")
    }

    fn unmap(&mut self) {
        assert!(self.mapped.get(), "Buffer is not mapped");
        self.inner.unmap();
        self.mapped.set(false);
    }

    fn read_mapped_range<R>(&mut self, _range: R) -> Result<BufferMappedRange<'_>, DeviceError>
    where
        R: BufferRange,
    {
        panic!("buffer mapping is not supported in the WebGPU backend")
    }

    fn write_mapped_range<R>(&mut self, range: R) -> Result<BufferMappedRangeMut<'_>, DeviceError>
    where
        R: BufferRange,
    {
        assert!(self.mapped.get(), "Buffer is not mapped");
        let range = range.range(self.size);
        assert!(
            range.start <= range.end && range.end <= self.size,
            "Buffer range out of bounds"
        );
        let size = range.end - range.start;
        if size == 0 {
            return Ok(BufferMappedRangeMut::new(
                self,
                None,
                0,
                range.start,
                0,
            ));
        }
        // WebGPU views require aligned bounds; logical writes can address any byte.
        let view_start = range.start & !(wgpu::MAP_ALIGNMENT as usize - 1);
        let view_end = range
            .end
            .checked_add(wgpu::COPY_BUFFER_ALIGNMENT as usize - 1)
            .expect("Buffer range exceeds address space")
            & !(wgpu::COPY_BUFFER_ALIGNMENT as usize - 1);
        let view = self
            .inner
            .get_mapped_range_mut(view_start as u64..view_end as u64)
            .expect("Failed to access buffer initial mapping");
        Ok(BufferMappedRangeMut::new(
            self,
            Some(view),
            range.start - view_start,
            range.start,
            size,
        ))
    }

    fn write(&mut self, _offset: usize, _data: &[u8]) -> Result<(), DeviceError> {
        panic!(
            "buffer host-write is not supported in the WebGPU backend; use Queue::write_buffer instead"
        )
    }

    fn read(&mut self, _offset: usize, _data: &mut [u8]) -> Result<(), DeviceError> {
        panic!("buffer host-read is not supported in the WebGPU backend")
    }
}

impl ArgumentsSealed for Buffer {}

impl ArgumentsField<Automatic> for Buffer {
    const KIND: ArgumentKind = crate::generic::ArgumentKind::UniformBuffer;
    const SIZE: Option<NonZero<usize>> = None;

    fn as_binding_resource(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.inner,
            offset: 0,
            size: None,
        })
    }
}

impl ArgumentsField<Uniform> for Buffer {
    const KIND: ArgumentKind = crate::generic::ArgumentKind::UniformBuffer;
    const SIZE: Option<NonZero<usize>> = None;

    fn as_binding_resource(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.inner,
            offset: 0,
            size: None,
        })
    }
}

impl ArgumentsField<Storage> for Buffer {
    const KIND: ArgumentKind = crate::generic::ArgumentKind::StorageBuffer;
    const SIZE: Option<NonZero<usize>> = None;

    fn as_binding_resource(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.inner,
            offset: 0,
            size: None,
        })
    }
}
