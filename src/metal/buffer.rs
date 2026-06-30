use std::{
    fmt,
    hash::{Hash, Hasher},
    ptr::NonNull,
};

use foreign_types::ForeignType;
use metal::{NSRange, NSUInteger};
use objc::*;

use crate::{
    generic::{ArgumentKind, Automatic, Storage, Uniform},
    BufferMappedRange, BufferMappedRangeMut, BufferRange, BufferUsage, DeviceError,
};

use super::arguments::ArgumentsField;

#[derive(Clone)]
pub struct Buffer {
    buffer: metal::Buffer,
    usage: BufferUsage,
}

impl Buffer {
    #[inline(always)]
    pub(super) fn new(buffer: metal::Buffer, usage: BufferUsage) -> Self {
        Buffer { buffer, usage }
    }

    #[inline(always)]
    pub(super) fn metal(&self) -> &metal::BufferRef {
        &self.buffer
    }

    #[inline]
    pub(crate) fn flush_range(&mut self, offset: usize, size: usize) -> Result<(), DeviceError> {
        self.buffer
            .did_modify_range(NSRange::new(offset as u64, size as u64));
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
        self.buffer.as_ptr().hash(state);
    }
}

impl PartialEq for Buffer {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.buffer.as_ptr() == other.buffer.as_ptr()
    }
}

impl Eq for Buffer {}

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
        let count: NSUInteger = unsafe { msg_send![(self.buffer.as_ptr()), retainCount] };
        count == 1
    }

    #[inline]
    fn map<R>(&mut self, range: R) -> Result<(), DeviceError>
    where
        R: BufferRange,
    {
        assert!(self.detached());
        assert!(!self.buffer.contents().is_null());
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

        assert!(!ptr.is_null());

        let length = self.buffer.length() as usize;

        let range = range.range(length);
        assert!(range.start <= range.end, "Invalid range");

        assert!(range.start <= length, "Range start out of bounds");
        assert!(range.end <= length, "Range end out of bounds");

        unsafe {
            let ptr = ptr.cast::<u8>().add(range.start);

            Ok(BufferMappedRange::new(
                self,
                NonNull::new_unchecked(ptr),
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

        assert!(!ptr.is_null());

        let length = self.buffer.length() as usize;

        let range = range.range(length);
        assert!(range.start <= range.end, "Invalid range");

        assert!(range.start <= length, "Range start out of bounds");
        assert!(range.end <= length, "Range end out of bounds");

        unsafe {
            let ptr = ptr.cast::<u8>().add(range.start);

            Ok(BufferMappedRangeMut::new(
                self,
                NonNull::new_unchecked(ptr),
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
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }

        Ok(())
    }

    fn read(&mut self, offset: usize, data: &mut [u8]) -> Result<(), DeviceError> {
        let length = self.buffer.length() as usize;

        assert!(offset <= length, "Offset out of bounds");
        assert!(offset + data.len() <= length, "Data out of bounds");

        unsafe {
            let ptr = self.buffer.contents().cast::<u8>().add(offset);
            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), data.len());
        }

        Ok(())
    }
}

impl ArgumentsField<Automatic> for Buffer {
    const KIND: ArgumentKind = ArgumentKind::UniformBuffer;
    const SIZE: usize = 1;

    #[inline(always)]
    fn bind_vertex(&self, slot: u32, encoder: &metal::RenderCommandEncoderRef) {
        encoder.set_vertex_buffer(slot.into(), Some(&self.buffer), 0)
    }

    #[inline(always)]
    fn bind_fragment(&self, slot: u32, encoder: &metal::RenderCommandEncoderRef) {
        encoder.set_fragment_buffer(slot.into(), Some(&self.buffer), 0)
    }

    #[inline(always)]
    fn bind_compute(&self, slot: u32, encoder: &metal::ComputeCommandEncoderRef) {
        encoder.set_buffer(slot.into(), Some(&self.buffer), 0)
    }
}

impl ArgumentsField<Uniform> for Buffer {
    const KIND: ArgumentKind = ArgumentKind::UniformBuffer;
    const SIZE: usize = 1;

    #[inline(always)]
    fn bind_vertex(&self, slot: u32, encoder: &metal::RenderCommandEncoderRef) {
        encoder.set_vertex_buffer(slot.into(), Some(&self.buffer), 0)
    }

    #[inline(always)]
    fn bind_fragment(&self, slot: u32, encoder: &metal::RenderCommandEncoderRef) {
        encoder.set_fragment_buffer(slot.into(), Some(&self.buffer), 0)
    }

    #[inline(always)]
    fn bind_compute(&self, slot: u32, encoder: &metal::ComputeCommandEncoderRef) {
        encoder.set_buffer(slot.into(), Some(&self.buffer), 0)
    }
}

impl ArgumentsField<Storage> for Buffer {
    const KIND: ArgumentKind = ArgumentKind::StorageBuffer;
    const SIZE: usize = 1;

    #[inline(always)]
    fn bind_vertex(&self, slot: u32, encoder: &metal::RenderCommandEncoderRef) {
        encoder.set_vertex_buffer(slot.into(), Some(&self.buffer), 0)
    }

    #[inline(always)]
    fn bind_fragment(&self, slot: u32, encoder: &metal::RenderCommandEncoderRef) {
        encoder.set_fragment_buffer(slot.into(), Some(&self.buffer), 0)
    }

    #[inline(always)]
    fn bind_compute(&self, slot: u32, encoder: &metal::ComputeCommandEncoderRef) {
        encoder.set_buffer(slot.into(), Some(&self.buffer), 0)
    }
}
