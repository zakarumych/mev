use std::ptr::NonNull;

use mev_proc::hidden_trait;

use super::Buffer;
use crate::DeviceError;

/// A borrowed read view of mapped buffer memory.
pub struct BufferMappedRange<'a> {
    _buffer: &'a mut Buffer,
    ptr: NonNull<u8>,
    size: usize,
}

impl<'a> BufferMappedRange<'a> {
    pub(super) fn new(buffer: &'a mut Buffer, ptr: NonNull<u8>, _offset: usize, size: usize) -> Self {
        Self {
            _buffer: buffer,
            ptr,
            size,
        }
    }
}

#[hidden_trait]
impl crate::traits::BufferMappedRange for BufferMappedRange<'_> {
    fn read(&self, offset: usize, data: &mut [u8]) {
        assert!(offset <= self.size && data.len() <= self.size - offset);
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.ptr.as_ptr().add(offset),
                data.as_mut_ptr(),
                data.len(),
            );
        }
    }
}

impl AsRef<[u8]> for BufferMappedRange<'_> {
    fn as_ref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }
}

/// A borrowed write view of mapped buffer memory.
pub struct BufferMappedRangeMut<'a> {
    buffer: &'a mut Buffer,
    ptr: NonNull<u8>,
    offset: usize,
    size: usize,
}

impl<'a> BufferMappedRangeMut<'a> {
    pub(super) fn new(buffer: &'a mut Buffer, ptr: NonNull<u8>, offset: usize, size: usize) -> Self {
        Self {
            buffer,
            ptr,
            offset,
            size,
        }
    }
}

#[hidden_trait]
impl crate::traits::BufferMappedRangeMutRead for BufferMappedRangeMut<'_> {
    fn read(&self, offset: usize, data: &mut [u8]) {
        assert!(offset <= self.size && data.len() <= self.size - offset);
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.ptr.as_ptr().add(offset),
                data.as_mut_ptr(),
                data.len(),
            );
        }
    }
}

#[hidden_trait]
impl crate::traits::BufferMappedRangeMut for BufferMappedRangeMut<'_> {
    fn write(&mut self, offset: usize, data: &[u8]) {
        assert!(offset <= self.size && data.len() <= self.size - offset);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.as_ptr().add(offset), data.len());
        }
    }

    fn flush(&mut self) -> Result<(), DeviceError> {
        self.buffer.flush_range(self.offset, self.size)
    }
}

impl Drop for BufferMappedRangeMut<'_> {
    fn drop(&mut self) {
        let _ = self.buffer.flush_range(self.offset, self.size);
    }
}

impl AsRef<[u8]> for BufferMappedRangeMut<'_> {
    fn as_ref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }
}

impl AsMut<[u8]> for BufferMappedRangeMut<'_> {
    fn as_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }
}