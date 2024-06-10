use std::{
    borrow::Cow,
    ops::{Index, Range, RangeFrom, RangeFull, RangeTo},
};

use crate::backend::Buffer;

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct BufferUsage: u32 {
        const TRANSFER_SRC = 0x0000_0001;
        const TRANSFER_DST = 0x0000_0002;
        const UNIFORM = 0x0000_0004;
        const STORAGE = 0x0000_0008;
        const INDEX = 0x0000_0010;
        const VERTEX = 0x0000_0020;
        const INDIRECT = 0x0000_0040;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Memory {
    Device,
    Shared,
    Upload,
    Download,
}

/// Buffer description.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferDesc<'a> {
    /// Buffer size.
    pub size: usize,

    /// Buffer usage flags.
    pub usage: BufferUsage,

    /// Buffer memory type.
    pub memory: Memory,

    /// Buffer debug name.
    pub name: &'a str,
}

/// Buffer description with initial contents.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferInitDesc<'a> {
    /// Buffer initial contents.
    pub data: &'a [u8],

    /// Buffer usage flags.
    pub usage: BufferUsage,

    /// Buffer memory type.
    pub memory: Memory,

    /// Buffer debug name.
    pub name: &'a str,
}

pub trait BufferIndex {
    fn range(self, size: usize) -> Range<usize>;
}

impl BufferIndex for Range<usize> {
    #[inline(always)]
    fn range(self, size: usize) -> Range<usize> {
        debug_assert!(self.end <= size, "buffer range out of bounds");
        let end = self.end.min(size);
        let start = self.start.min(end);
        start..end
    }
}

impl BufferIndex for RangeFrom<usize> {
    #[inline(always)]
    fn range(self, size: usize) -> Range<usize> {
        debug_assert!(self.start <= size, "buffer range out of bounds");
        let start = self.start.min(size);
        start..size
    }
}

impl BufferIndex for RangeTo<usize> {
    #[inline(always)]
    fn range(self, size: usize) -> Range<usize> {
        debug_assert!(self.end <= size, "buffer range out of bounds");
        let end = self.end.min(size);
        0..end
    }
}

impl BufferIndex for RangeFull {
    #[inline(always)]
    fn range(self, size: usize) -> Range<usize> {
        0..size
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferSlice<'a> {
    pub(crate) buffer: &'a Buffer,
    pub(crate) offset: usize,
    pub(crate) size: usize,
}

impl PartialEq<Buffer> for BufferSlice<'_> {
    #[inline(always)]
    fn eq(&self, other: &Buffer) -> bool {
        *self.buffer == *other && self.offset == 0 && self.size == other.size()
    }
}

impl PartialEq<BufferSlice<'_>> for Buffer {
    #[inline(always)]
    fn eq(&self, other: &BufferSlice) -> bool {
        *self == *other.buffer && other.offset == 0 && other.size == self.size()
    }
}

impl BufferSlice<'_> {
    #[inline(always)]
    pub fn buffer(&self) -> &Buffer {
        self.buffer
    }

    #[inline(always)]
    pub fn offset(&self) -> usize {
        self.offset
    }

    #[inline(always)]
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Buffer {
    /// Returns range of the buffer.
    #[inline(always)]
    pub fn slice<R>(&self, range: R) -> BufferSlice
    where
        R: BufferIndex,
    {
        let range = range.range(self.size());
        BufferSlice {
            buffer: self,
            offset: range.start,
            size: range.end - range.start,
        }
    }

    /// Returns range of the buffer.
    #[inline(always)]
    pub fn split_at(&self, at: usize) -> (BufferSlice, BufferSlice) {
        let size = self.size();
        debug_assert!(at <= size);
        let at = at.min(size);

        let before = BufferSlice {
            buffer: self,
            offset: 0,
            size: at,
        };
        let after = BufferSlice {
            buffer: self,
            offset: at,
            size: size - at,
        };

        (before, after)
    }
}

impl<'a> BufferSlice<'a> {
    /// Returns sub-range of the buffer range.
    #[inline(always)]
    pub fn slice<R>(self, range: R) -> BufferSlice<'a>
    where
        R: BufferIndex,
    {
        let range = range.range(self.size);
        BufferSlice {
            buffer: self.buffer,
            offset: self.offset + range.start,
            size: range.end - range.start,
        }
    }

    /// Returns range of the buffer.
    #[inline(always)]
    pub fn split_at(&self, at: usize) -> (BufferSlice<'a>, BufferSlice<'a>) {
        let size = self.size();
        debug_assert!(at <= size);
        let at = at.min(size);

        let before = BufferSlice {
            buffer: self.buffer,
            offset: self.offset,
            size: at,
        };

        let after = BufferSlice {
            buffer: self.buffer,
            offset: self.offset + at,
            size: size - at,
        };

        (before, after)
    }
}

impl<'a> From<&'a Buffer> for BufferSlice<'a> {
    #[inline(always)]
    fn from(buffer: &'a Buffer) -> Self {
        BufferSlice {
            offset: 0,
            size: buffer.size(),
            buffer,
        }
    }
}

/// Trait for taking slice from the buffer.
pub trait AsBufferSlice {
    fn as_buffer_slice(&self) -> BufferSlice;
}

impl AsBufferSlice for BufferSlice<'_> {
    #[inline(always)]
    fn as_buffer_slice(&self) -> BufferSlice {
        *self
    }
}

impl AsBufferSlice for Buffer {
    #[inline(always)]
    fn as_buffer_slice(&self) -> BufferSlice {
        BufferSlice {
            offset: 0,
            size: self.size(),
            buffer: self,
        }
    }
}

impl<B> AsBufferSlice for &B
where
    B: AsBufferSlice,
{
    #[inline(always)]
    fn as_buffer_slice(&self) -> BufferSlice {
        (*self).as_buffer_slice()
    }
}
