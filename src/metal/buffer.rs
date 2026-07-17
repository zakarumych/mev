use std::{
    fmt,
    hash::{Hash, Hasher},
    num::NonZero,
    ptr::NonNull,
};

use objc2::{msg_send, rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSUInteger};
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLRenderCommandEncoder};

use crate::{
    BufferMappedRange, BufferMappedRangeMut, BufferRange, BufferSlice, BufferUsage, DeviceError,
    generic::{ArgumentKind, Automatic, Storage, Uniform},
};

use super::arguments::ArgumentsField;

#[derive(Clone)]
pub struct Buffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    usage: BufferUsage,
}

impl Buffer {
    #[inline(always)]
    pub(super) fn new(buffer: Retained<ProtocolObject<dyn MTLBuffer>>, usage: BufferUsage) -> Self {
        Buffer { buffer, usage }
    }

    #[inline(always)]
    pub(super) fn metal(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }

    #[inline]
    pub(crate) fn flush_range(&mut self, offset: usize, size: usize) -> Result<(), DeviceError> {
        self.buffer.didModifyRange(NSRange::new(offset, size));
        Ok(())
    }
}

unsafe impl Send for Buffer {}

impl fmt::Debug for Buffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Buffer")
            .field("buffer", &self.buffer)
            .finish()
    }
}

impl Hash for Buffer {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        Retained::as_ptr(&self.buffer).hash(state);
    }
}

impl PartialEq for Buffer {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.buffer == other.buffer
    }
}

impl Eq for Buffer {}

unsafe impl Sync for Buffer {}

impl crate::traits::Resource for Buffer {}

#[hidden_trait::expose]
impl crate::traits::Buffer for Buffer {
    #[inline(always)]
    fn size(&self) -> usize {
        self.buffer.length() as usize
    }

    #[inline(always)]
    fn usage(&self) -> BufferUsage {
        self.usage
    }

    #[inline(always)]
    fn detached(&self) -> bool {
        let count: NSUInteger = unsafe { msg_send![&self.buffer, retainCount] };
        count == 1
    }

    #[inline]
    fn map<R>(&mut self, range: R) -> Result<(), DeviceError>
    where
        R: BufferRange,
    {
        assert!(self.detached());
        assert!(
            self.usage
                .intersects(BufferUsage::HOST_READ | BufferUsage::HOST_WRITE)
        );
        Ok(())
    }

    #[inline(always)]
    fn unmap(&mut self) {}

    fn read_mapped_range<R>(&mut self, range: R) -> Result<BufferMappedRange<'_>, DeviceError>
    where
        R: BufferRange,
    {
        assert!(self.detached());

        let ptr = self.buffer.contents();

        let length = self.buffer.length();

        let range = range.range(length);
        assert!(range.start <= range.end, "Invalid range");

        assert!(range.start <= length, "Range start out of bounds");
        assert!(range.end <= length, "Range end out of bounds");

        unsafe {
            let ptr = ptr.cast::<u8>().add(range.start);

            Ok(BufferMappedRange::new(
                self,
                ptr,
                range.start,
                range.end - range.start,
            ))
        }
    }

    fn write_mapped_range<R>(&mut self, range: R) -> Result<BufferMappedRangeMut<'_>, DeviceError>
    where
        R: BufferRange,
    {
        assert!(self.detached());

        let ptr = self.buffer.contents();

        let length = self.buffer.length();

        let range = range.range(length);
        assert!(range.start <= range.end, "Invalid range");

        assert!(range.start <= length, "Range start out of bounds");
        assert!(range.end <= length, "Range end out of bounds");

        unsafe {
            let ptr = ptr.cast::<u8>().add(range.start);

            Ok(BufferMappedRangeMut::new(
                self,
                ptr,
                range.start,
                range.end - range.start,
            ))
        }
    }

    fn write(&mut self, offset: usize, data: &[u8]) -> Result<(), DeviceError> {
        let length = self.buffer.length() as usize;

        assert!(offset <= length, "Offset out of bounds");
        assert!(offset + data.len() <= length, "Data out of bounds");

        unsafe {
            let ptr = self.buffer.contents().cast::<u8>().add(offset);
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.as_ptr(), data.len());
        }

        Ok(())
    }

    fn read(&mut self, offset: usize, data: &mut [u8]) -> Result<(), DeviceError> {
        let length = self.buffer.length() as usize;

        assert!(offset <= length, "Offset out of bounds");
        assert!(offset + data.len() <= length, "Data out of bounds");

        unsafe {
            let ptr = self.buffer.contents().cast::<u8>().add(offset);
            std::ptr::copy_nonoverlapping(ptr.as_ptr(), data.as_mut_ptr(), data.len());
        }

        Ok(())
    }
}

impl ArgumentsField<Automatic> for Buffer {
    const KIND: ArgumentKind = ArgumentKind::UniformBuffer;
    const SIZE: Option<NonZero<usize>> = None;

    #[inline(always)]
    fn bind_vertex(&self, slot: u32, encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>) {
        unsafe {
            encoder.setVertexBuffer_offset_atIndex(Some(&self.buffer), 0, slot as NSUInteger);
        }
    }

    #[inline(always)]
    fn bind_fragment(&self, slot: u32, encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>) {
        unsafe {
            encoder.setFragmentBuffer_offset_atIndex(Some(&self.buffer), 0, slot as NSUInteger);
        }
    }

    #[inline(always)]
    fn bind_compute(&self, slot: u32, encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&self.buffer), 0, slot as NSUInteger);
        }
    }
}

impl ArgumentsField<Uniform> for Buffer {
    const KIND: ArgumentKind = ArgumentKind::UniformBuffer;
    const SIZE: Option<NonZero<usize>> = None;

    #[inline(always)]
    fn bind_vertex(&self, slot: u32, encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>) {
        unsafe {
            encoder.setVertexBuffer_offset_atIndex(Some(&self.buffer), 0, slot as NSUInteger);
        }
    }

    #[inline(always)]
    fn bind_fragment(&self, slot: u32, encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>) {
        unsafe {
            encoder.setFragmentBuffer_offset_atIndex(Some(&self.buffer), 0, slot as NSUInteger);
        }
    }

    #[inline(always)]
    fn bind_compute(&self, slot: u32, encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&self.buffer), 0, slot as NSUInteger);
        }
    }
}

impl ArgumentsField<Storage> for Buffer {
    const KIND: ArgumentKind = ArgumentKind::StorageBuffer;
    const SIZE: Option<NonZero<usize>> = None;

    #[inline(always)]
    fn bind_vertex(&self, slot: u32, encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>) {
        unsafe {
            encoder.setVertexBuffer_offset_atIndex(Some(&self.buffer), 0, slot as NSUInteger);
        }
    }

    #[inline(always)]
    fn bind_fragment(&self, slot: u32, encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>) {
        unsafe {
            encoder.setFragmentBuffer_offset_atIndex(Some(&self.buffer), 0, slot as NSUInteger);
        }
    }

    #[inline(always)]
    fn bind_compute(&self, slot: u32, encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&self.buffer), 0, slot as NSUInteger);
        }
    }
}

impl ArgumentsField<Automatic> for BufferSlice<'static> {
    const KIND: ArgumentKind = ArgumentKind::UniformBuffer;
    const SIZE: Option<NonZero<usize>> = None;

    #[inline(always)]
    fn bind_vertex(&self, slot: u32, encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>) {
        unsafe {
            encoder.setVertexBuffer_offset_atIndex(
                Some(&self.buffer().buffer),
                self.offset() as NSUInteger,
                slot as NSUInteger,
            );
        }
    }

    #[inline(always)]
    fn bind_fragment(&self, slot: u32, encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>) {
        unsafe {
            encoder.setFragmentBuffer_offset_atIndex(
                Some(&self.buffer().buffer),
                self.offset() as NSUInteger,
                slot as NSUInteger,
            );
        }
    }

    #[inline(always)]
    fn bind_compute(&self, slot: u32, encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
        unsafe {
            encoder.setBuffer_offset_atIndex(
                Some(&self.buffer().buffer),
                self.offset() as NSUInteger,
                slot as NSUInteger,
            );
        }
    }
}
