use std::{
    borrow::Cow,
    mem::ManuallyDrop,
    ops::{Range, RangeFrom, RangeFull, RangeTo},
    ptr::NonNull,
};

use crate::{Extent, HasW, HasX, HasY, HasZ, PixelFormat, Stride, backend::Buffer};

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

        /// Buffer can be read by the host (CPU) using map operations.
        const HOST_READ = 0x0001_0000;

        /// Buffer can be written by the host (CPU) using map operations.
        const HOST_WRITE = 0x0002_0000;

        /// Buffer is transient and will be used for a short time and then discarded.
        /// This is a hint to the backend that it can optimize memory usage for this buffer.
        const TRANSIENT = 0x0100_0000;
    }
}

/// Description used for buffer creation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferDesc<'a> {
    /// Buffer size.
    pub size: usize,

    /// Buffer usage flags.
    pub usage: BufferUsage,

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

    /// Buffer debug name.
    pub name: &'a str,
}

/// Trait for types that can be used to index a buffer to get a slice of it.
/// It is implemented for different range types over `usize`.
pub trait BufferRange {
    /// Returns range for given buffer size.
    fn range(self, size: usize) -> Range<usize>;
}

impl BufferRange for Range<usize> {
    #[inline(always)]
    fn range(self, size: usize) -> Range<usize> {
        debug_assert!(self.start <= self.end, "buffer range out of bounds");
        debug_assert!(self.end <= size, "buffer range out of bounds");
        self.start..self.end
    }
}

impl BufferRange for RangeFrom<usize> {
    #[inline(always)]
    fn range(self, size: usize) -> Range<usize> {
        debug_assert!(self.start <= size, "buffer range out of bounds");
        self.start..size
    }
}

impl BufferRange for RangeTo<usize> {
    #[inline(always)]
    fn range(self, size: usize) -> Range<usize> {
        debug_assert!(self.end <= size, "buffer range out of bounds");
        0..self.end
    }
}

impl BufferRange for RangeFull {
    #[inline(always)]
    fn range(self, size: usize) -> Range<usize> {
        0..size
    }
}

pub struct BufferMappedRange<'a> {
    buffer: &'a mut Buffer,
    ptr: NonNull<u8>,
    offset: usize,
    size: usize,
}

impl<'a> BufferMappedRange<'a> {
    pub(crate) fn new(
        buffer: &'a mut Buffer,
        ptr: NonNull<u8>,
        offset: usize,
        size: usize,
    ) -> Self {
        BufferMappedRange {
            buffer,
            ptr,
            offset,
            size,
        }
    }

    pub fn read(&self, offset: usize, data: &mut [u8]) {
        assert!(offset + data.len() <= self.size);

        unsafe {
            let src = self.ptr.as_ptr().add(offset);
            std::ptr::copy_nonoverlapping(src, data.as_mut_ptr(), data.len());
        }
    }
}

impl<'a> AsRef<[u8]> for BufferMappedRange<'a> {
    fn as_ref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const u8, self.size) }
    }
}

pub struct BufferMappedRangeMut<'a> {
    buffer: &'a mut Buffer,
    ptr: NonNull<u8>,
    offset: usize,
    size: usize,
}

impl<'a> BufferMappedRangeMut<'a> {
    pub(crate) fn new(
        buffer: &'a mut Buffer,
        ptr: NonNull<u8>,
        offset: usize,
        size: usize,
    ) -> Self {
        BufferMappedRangeMut {
            buffer,
            ptr,
            offset,
            size,
        }
    }

    pub fn read(&self, offset: usize, data: &mut [u8]) {
        assert!(offset + data.len() <= self.size);

        unsafe {
            let src = self.ptr.as_ptr().add(offset);
            std::ptr::copy_nonoverlapping(src, data.as_mut_ptr(), data.len());
        }
    }

    pub fn write(&mut self, offset: usize, data: &[u8]) {
        assert!(offset + data.len() <= self.size);

        unsafe {
            let dst = self.ptr.as_ptr().add(offset);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
    }

    pub fn flush(&mut self) -> Result<(), crate::generic::DeviceError> {
        let offset = self.offset;
        let size = self.size;
        let mut me = ManuallyDrop::new(self);
        me.buffer.flush_range(offset, size)
    }
}

impl<'a> Drop for BufferMappedRangeMut<'a> {
    fn drop(&mut self) {
        let _ = self.buffer.flush_range(self.offset, self.size);
    }
}

impl<'a> AsRef<[u8]> for BufferMappedRangeMut<'a> {
    fn as_ref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const u8, self.size) }
    }
}

impl<'a> AsMut<[u8]> for BufferMappedRangeMut<'a> {
    fn as_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut u8, self.size) }
    }
}

/// Slice of a buffer is a reference to a buffer with offset and size.
/// Mostly found in function arguments.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferSlice<'a> {
    pub(crate) buffer: Cow<'a, Buffer>,
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
        self.buffer.as_ref()
    }

    #[inline(always)]
    pub fn offset(&self) -> usize {
        self.offset
    }

    #[inline(always)]
    pub fn size(&self) -> usize {
        self.size
    }

    #[inline(always)]
    pub fn into_owned(self) -> BufferSlice<'static> {
        BufferSlice {
            buffer: Cow::Owned(self.buffer.into_owned()),
            offset: self.offset,
            size: self.size,
        }
    }
}

impl Buffer {
    /// Returns buffer slice with given range.
    #[inline(always)]
    pub fn slice<R>(&self, range: R) -> BufferSlice<'_>
    where
        R: BufferRange,
    {
        let range = range.range(self.size());
        BufferSlice {
            buffer: Cow::Borrowed(self),
            offset: range.start,
            size: range.end - range.start,
        }
    }

    /// Splits buffer into two ranges, from start to `at` and from `at` to end.
    #[inline(always)]
    pub fn split_at(&self, at: usize) -> (BufferSlice<'_>, BufferSlice<'_>) {
        let size = self.size();
        debug_assert!(at <= size);
        let at = at.min(size);

        let before = BufferSlice {
            buffer: Cow::Borrowed(self),
            offset: 0,
            size: at,
        };
        let after = BufferSlice {
            buffer: Cow::Borrowed(self),
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
        R: BufferRange,
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
            buffer: self.buffer.clone(),
            offset: self.offset,
            size: at,
        };

        let after = BufferSlice {
            buffer: self.buffer.clone(),
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
            buffer: Cow::Borrowed(buffer),
        }
    }
}

/// Trait to generalize over types that can be converted to buffer slice.
/// This is a buffer slice itself, a buffer and references.
pub trait AsBufferSlice {
    fn as_buffer_slice(&self) -> BufferSlice<'_>;
}

impl AsBufferSlice for BufferSlice<'_> {
    #[inline(always)]
    fn as_buffer_slice(&self) -> BufferSlice<'_> {
        self.clone()
    }
}

impl AsBufferSlice for Buffer {
    #[inline(always)]
    fn as_buffer_slice(&self) -> BufferSlice<'_> {
        BufferSlice {
            offset: 0,
            size: self.size(),
            buffer: Cow::Borrowed(self),
        }
    }
}

impl<B> AsBufferSlice for &B
where
    B: AsBufferSlice,
{
    #[inline(always)]
    fn as_buffer_slice(&self) -> BufferSlice<'_> {
        (*self).as_buffer_slice()
    }
}

/// Describes how texels of a given `PixelFormat` are laid out in memory:
/// per-axis byte stride and element extent, up to 4 dimensions.
///
/// `stride[i]` is the byte pitch spanning everything at rank `< i`
/// (row-major), not merely that axis's own size - e.g. `stride[1]` (row) is
/// the distance from the start of one row to the next, which is the byte
/// span of a full row of elements; `stride[2]` (plane) is the distance
/// between depth slices, the byte span of a full plane of rows; etc.
///
/// Each axis's stride may be left unset (`None`) - see
/// [`TexelShape::byte_stride`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TexelShape<const N: usize> {
    pub format: PixelFormat,
    pub stride: [Option<usize>; N],
    pub extent: Extent<N>,
}

impl HasX for TexelShape<1> {}
impl HasX for TexelShape<2> {}
impl HasY for TexelShape<2> {}
impl HasX for TexelShape<3> {}
impl HasY for TexelShape<3> {}
impl HasZ for TexelShape<3> {}
impl HasX for TexelShape<4> {}
impl HasY for TexelShape<4> {}
impl HasZ for TexelShape<4> {}
impl HasW for TexelShape<4> {}

impl<const N: usize> TexelShape<N> {
    /// Creates a new `TexelShape` with the given pixel format and extent.
    /// All strides are unset, use [`TexelShape::with_stride`] or named methods to set some of them.
    pub fn new(format: PixelFormat, extent: Extent<N>) -> Self {
        TexelShape {
            format,
            stride: [None; N],
            extent,
        }
    }

    /// Sets every dimension stride explicitly.
    pub fn with_stride(mut self, stride: Stride<N>) -> Self {
        self.stride = stride.0.map(Some);
        self
    }

    /// Sets the stride for a specific dimension.
    pub fn with_dimension_stride<const D: usize>(mut self, stride: usize) -> Self {
        const {
            assert!(D < N, "dimension index out of bounds");
        }
        self.stride[D] = Some(stride);
        self
    }

    /// Sets the stride between texels.
    pub fn with_texel_stride(mut self, stride: usize) -> Self
    where
        Self: HasX,
    {
        self.stride[0] = Some(stride);
        self
    }

    /// Sets the stride between rows.
    pub fn with_row_stride(mut self, stride: usize) -> Self
    where
        Self: HasY,
    {
        self.stride[1] = Some(stride);
        self
    }

    /// Sets the stride between planes.
    pub fn with_plane_stride(mut self, stride: usize) -> Self
    where
        Self: HasZ,
    {
        self.stride[2] = Some(stride);
        self
    }

    /// Sets the stride between volumes.
    pub fn with_volume_stride(mut self, stride: usize) -> Self
    where
        Self: HasW,
    {
        self.stride[3] = Some(stride);
        self
    }

    /// Returns byte strides.
    /// For dimensions with unset stride, computes the tightly-packed default.
    pub fn byte_stride(&self) -> [usize; N] {
        let mut out = [0usize; N];
        for i in 0..N {
            out[i] = match self.stride[i] {
                Some(s) => s,
                None if i == 0 => self.format.block_size(),
                None => out[i - 1] * self.extent.0[i - 1] as usize,
            };
        }
        out
    }
}
