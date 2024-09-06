use std::{
    borrow::Cow,
    ops::{Index, Range, RangeFrom, RangeFull, RangeTo},
};

use crate::backend::Buffer;

bitflags::bitflags! {
    /// Buffer usage flags.
    /// 
    /// Buffer can only be used according to usage flags specified during creation.
    /// When creating a buffer, choose all flags that apply.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct BufferUsage: u32 {
        /// Buffer can be used as a source for transfer operations.
        /// i.e. it will be copied from.
        const TRANSFER_SRC = 0x0000_0001;

        /// Buffer can be used as a destination for transfer operations.
        /// i.e. it will be copied to.
        const TRANSFER_DST = 0x0000_0002;
        
        /// Buffer can be used as a uniform buffer in shader arguments.
        const UNIFORM = 0x0000_0004;

        /// Buffer can be used as a storage buffer in shader arguments.
        const STORAGE = 0x0000_0008;

        /// Buffer can be used as a index buffer in indexed draw calls.
        const INDEX = 0x0000_0010;

        /// Buffer can be used as a vertex buffer in draw calls.
        const VERTEX = 0x0000_0020;

        /// Buffer can be used as a indirect buffer in indirect draw calls.
        const INDIRECT = 0x0000_0040;
    }
}

/// Specifies what memory type should be allocated for the buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Memory {
    /// Memory is allocated on the device.
    /// This memory is fastest to access by the device,
    /// but may not be accessible by the host.
    Device,

    /// Memory is allocated so that it can be accessed by the host.
    /// It can be used directly in shaders, but it is slower than device memory.
    /// 
    /// Note that memory access must be synchronized between the host and the device.
    Shared,

    /// Memory is allocated on the device and can be accessed by the host.
    /// 
    /// It is designated for upload operations.
    /// 
    /// Typical use case is staging memory to copy data from host to device memory.
    /// e.g. Host memory -> Staging buffer -> Device buffer.
    Upload,

    /// Memory is allocated on the device and can be accessed by the host.
    /// 
    /// It is designated for download operations.
    /// 
    /// Typical use case is staging memory to copy data from device to host memory.
    /// e.g. Device buffer -> Staging buffer -> Host memory.
    Download,
}

/// Description used for buffer creation.
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

/// Description used for buffer creation with initial contents.
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

/// Trait for types that can be used to index a buffer to get a slice of it.
/// It is implemented for different range types over `usize`.
pub trait BufferIndex {
    /// Returns range for given buffer size.
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

/// Slice of a buffer is a reference to a buffer with offset and size.
/// Mostly found in function arguments.
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
    /// Returns buffer slice with given range.
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

    /// Splits buffer into two ranges, from start to `at` and from `at` to end.
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
    /// Returns buffer slice with given range.
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

    /// Splits buffer into two ranges, from start to `at` and from `at` to end.
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

// To accept whole buffer where buffer slice is expected.
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

/// Trait to generalize over types that can be converted to buffer slice.
/// This is a buffer slice itself, a buffer and references.
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
