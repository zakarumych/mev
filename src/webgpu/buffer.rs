use std::{fmt, hash, rc::Rc};

use crate::{
    backend::{arguments::ArgumentsField, from::WgpuInto},
    generic::{
        BufferDesc, BufferInitDesc, BufferMappedRange, BufferMappedRangeMut, BufferRange,
        BufferUsage, DeviceError,
    },
    ArgumentKind, ArgumentsSealed, Automatic, Storage, Uniform,
};

#[derive(Clone)]
pub struct Buffer {
    inner: wgpu::Buffer,
}

impl Buffer {
    pub(super) fn new(buffer: wgpu::Buffer) -> Self {
        Buffer { inner: buffer }
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

#[hidden_trait::expose]
impl crate::traits::Buffer for Buffer {
    fn size(&self) -> usize {
        self.inner.size() as usize
    }

    fn usage(&self) -> BufferUsage {
        self.inner.usage().wgpu_into()
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
        panic!("buffer mapping is not supported in the WebGPU backend")
    }

    fn read_mapped_range<R>(&mut self, _range: R) -> Result<BufferMappedRange<'_>, DeviceError>
    where
        R: BufferRange,
    {
        panic!("buffer mapping is not supported in the WebGPU backend")
    }

    fn write_mapped_range<R>(&mut self, _range: R) -> Result<BufferMappedRangeMut<'_>, DeviceError>
    where
        R: BufferRange,
    {
        panic!("buffer mapping is not supported in the WebGPU backend")
    }

    fn write(&mut self, _offset: usize, _data: &[u8]) -> Result<(), DeviceError> {
        panic!("buffer host-write is not supported in the WebGPU backend; use CopyCommandEncoder::write_buffer instead")
    }

    fn read(&mut self, _offset: usize, _data: &mut [u8]) -> Result<(), DeviceError> {
        panic!("buffer host-read is not supported in the WebGPU backend")
    }
}

impl ArgumentsSealed for Buffer {}

impl ArgumentsField<Automatic> for Buffer {
    const KIND: ArgumentKind = crate::generic::ArgumentKind::UniformBuffer;
    const SIZE: usize = 1;

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
    const SIZE: usize = 1;

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
    const SIZE: usize = 1;

    fn as_binding_resource(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.inner,
            offset: 0,
            size: None,
        })
    }
}
