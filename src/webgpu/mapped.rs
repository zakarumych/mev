use std::{convert::Infallible, marker::PhantomData};

use mev_proc::hidden_trait;

use super::Buffer;
use crate::DeviceError;

/// A read view unavailable because synchronous buffer mapping is unsupported.
pub struct BufferMappedRange<'a> {
    unavailable: Infallible,
    _buffer: PhantomData<&'a mut Buffer>,
}

#[hidden_trait]
impl crate::traits::BufferMappedRange for BufferMappedRange<'_> {
    fn read(&self, _offset: usize, _data: &mut [u8]) {
        match self.unavailable {}
    }
}

impl AsRef<[u8]> for BufferMappedRange<'_> {
    fn as_ref(&self) -> &[u8] {
        match self.unavailable {}
    }
}

/// A borrowed write-only view of mapped buffer memory.
/// Drop the view before unmapping the buffer.
pub struct BufferMappedRangeMut<'a> {
    buffer: &'a mut Buffer,
    view: Option<wgpu::BufferViewMut>,
    view_offset: usize,
    offset: usize,
    size: usize,
}

impl<'a> BufferMappedRangeMut<'a> {
    pub(super) fn new(
        buffer: &'a mut Buffer,
        view: Option<wgpu::BufferViewMut>,
        view_offset: usize,
        offset: usize,
        size: usize,
    ) -> Self {
        Self {
            buffer,
            view,
            view_offset,
            offset,
            size,
        }
    }
}

#[hidden_trait]
impl crate::traits::BufferMappedRangeMut for BufferMappedRangeMut<'_> {
    fn write(&mut self, offset: usize, data: &[u8]) {
        assert!(offset <= self.size && data.len() <= self.size - offset);
        if !data.is_empty() {
            self.view
                .as_mut()
                .expect("Nonempty mapped range has a view")
                .slice(self.view_offset + offset..self.view_offset + offset + data.len())
                .copy_from_slice(data);
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
