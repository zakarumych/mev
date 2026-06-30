use std::{
    fmt,
    hash::{Hash, Hasher},
    ptr::NonNull,
    sync::Arc,
};

use web_sys::{WebGl2RenderingContext as GL, WebGlBuffer};

use crate::{
    generic::{
        ArgumentKind, Automatic, BufferMappedRange, BufferMappedRangeMut, BufferRange, BufferUsage,
        DeviceError, Storage, Uniform,
    },
    RenderCommandEncoder,
};

use super::arguments::ArgumentsField;

struct ShadowBuffer {
    offset: usize,
    size: usize,
    ptr: NonNull<u8>,
}

impl ShadowBuffer {
    fn new(offset: usize, size: usize) -> Self {
        let ptr = Box::into_raw(Box::<[u8]>::new_zeroed_slice(size));
        // SAFETY: We just created the box, so the pointer is valid and non-null
        // Zeroed slice is valid for u8, so we can safely cast it to a pointer to u8
        let ptr = unsafe { NonNull::new_unchecked(ptr as *mut u8) };
        ShadowBuffer { offset, size, ptr }
    }

    /// # Safety
    ///
    /// This function may only be called when no slice reference to the shadow buffer memory exists.
    unsafe fn as_slice(&self) -> &[u8] {
        // SAFETY: The pointer is valid and non-null, and the size is correct
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }

    /// # Safety
    ///
    /// This function may only be called when no slice reference to the shadow buffer memory exists.
    unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: The pointer is valid and non-null, and the size is correct
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }

    /// # Safety
    ///
    /// This function may only be called when no slice reference to the shadow buffer memory exists.
    unsafe fn deallocate(self) {
        // SAFETY: We are taking ownership of the pointer and size, so we can safely deallocate it
        unsafe {
            Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                self.ptr.as_ptr(),
                self.size,
            ))
        };
    }
}

struct Inner {
    // Store the WebGL buffer object
    buffer: WebGlBuffer,
    size: usize,
    usage: BufferUsage,
    // Reference to the owning WebGL context
    gl: GL,
    shadow: Option<ShadowBuffer>,
}

#[derive(Clone)]
pub struct Buffer {
    inner: Arc<Inner>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        if let Some(shadow) = self.shadow.take() {
            // SAFETY: No slice references to the shadow buffer memory exist, so we can safely deallocate it
            unsafe { shadow.deallocate() };
        }

        self.gl.delete_buffer(Some(&self.buffer));
    }
}

impl Buffer {
    pub(super) fn new(
        buffer: WebGlBuffer,
        size: usize,
        usage: BufferUsage,
        gl: GL,
    ) -> Self {
        Buffer {
            inner: Arc::new(Inner {
                buffer,
                size,
                usage,
                gl,
                shadow: None,
            }),
        }
    }

    fn detatched_inner(&mut self) -> &mut Inner {
        Arc::get_mut(&mut self.inner).expect("Buffer must be detached to write to it")
    }

    pub(super) fn webgl(&self) -> &WebGlBuffer {
        &self.inner.buffer
    }

    fn invalidate_range(inner: &mut Inner, offset: usize, size: usize) -> Result<(), DeviceError> {
        debug_assert!(
            i32::try_from(offset + size).is_ok(),
            "Range exceeds i32::MAX"
        );

        let shadow = inner.shadow.as_mut().expect("Buffer is not mapped");

        assert!(
            offset >= shadow.offset && offset + size <= shadow.offset + shadow.size,
            "Range is out of bounds of the mapped buffer"
        );

        let offset = offset - shadow.offset;

        // SAFETY: This function is called only when no slice reference to shadow buffer exists.
        let shadow_bytes = unsafe { shadow.as_mut_slice() };

        inner.gl.bind_buffer(GL::ARRAY_BUFFER, Some(&inner.buffer));

        inner.gl.get_buffer_sub_data_with_i32_and_u8_array(
            GL::ARRAY_BUFFER,
            offset as i32,
            &mut shadow_bytes[offset..][..size],
        );

        inner.gl.bind_buffer(GL::ARRAY_BUFFER, None);

        Ok(())
    }

    pub(crate) fn flush_range(&mut self, offset: usize, size: usize) -> Result<(), DeviceError> {
        debug_assert!(
            i32::try_from(offset + size).is_ok(),
            "Range exceeds i32::MAX"
        );

        let inner = self.detatched_inner();

        let shadow = inner.shadow.as_mut().expect("Buffer is not mapped");

        assert!(
            offset >= shadow.offset && offset + size <= shadow.offset + shadow.size,
            "Range is out of bounds of the mapped buffer"
        );

        let offset = offset - shadow.offset;

        // SAFETY: This function is called only when no slice reference to shadow buffer exists.
        let shadow_bytes = unsafe { shadow.as_slice() };

        inner.gl.bind_buffer(GL::ARRAY_BUFFER, Some(&inner.buffer));

        inner.gl.buffer_sub_data_with_i32_and_u8_array(
            GL::ARRAY_BUFFER,
            offset as i32,
            &shadow_bytes[offset..][..size],
        );

        inner.gl.bind_buffer(GL::ARRAY_BUFFER, None);

        Ok(())
    }
}

impl fmt::Debug for Buffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            f.debug_struct("Buffer")
                .field("handle", &self.inner.buffer)
                .field("size", &self.inner.size)
                .field("usage", &self.inner.usage)
                .finish()
        } else {
            f.debug_tuple("Buffer").field(&self.inner.buffer).finish()
        }
    }
}

impl Hash for Buffer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::ptr::hash(&*self.inner, state);
    }
}

impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(&*self.inner, &*other.inner)
    }
}

impl Eq for Buffer {}

impl crate::traits::Resource for Buffer {}

#[hidden_trait::expose]
impl crate::traits::Buffer for Buffer {
    #[inline(always)]
    fn size(&self) -> usize {
        self.inner.size
    }

    #[inline(always)]
    fn usage(&self) -> BufferUsage {
        self.inner.usage
    }

    #[inline(always)]
    fn detached(&self) -> bool {
        debug_assert_eq!(Arc::weak_count(&self.inner), 0, "No weak refs allowed");
        Arc::strong_count(&self.inner) == 1
    }

    #[inline(always)]
    fn map<R>(&mut self, range: R) -> Result<(), crate::DeviceError>
    where
        R: crate::generic::BufferRange,
    {
        let inner = self.detatched_inner();

        assert!(inner.shadow.is_none(), "Buffer is already mapped");
        let range = range.range(inner.size);

        assert!(range.start <= range.end, "Invalid range");

        assert!(
            i32::try_from(range.end).is_ok(),
            "Range end exceeds i32::MAX"
        );

        inner.shadow = Some(ShadowBuffer::new(range.start, range.end - range.start));

        Ok(())
    }

    fn unmap(&mut self) {
        let inner = self.detatched_inner();

        if let Some(shadow) = inner.shadow.take() {
            // SAFETY: No slice references to the shadow buffer memory exist, so we can safely deallocate it
            unsafe {
                shadow.deallocate();
            }
        }
    }

    fn read_mapped_range<R>(&mut self, range: R) -> Result<BufferMappedRange<'_>, DeviceError>
    where
        R: BufferRange,
    {
        let inner: &mut Inner = self.detatched_inner();

        let range = range.range(inner.size);

        assert!(range.start <= range.end, "Invalid range");

        assert!(
            i32::try_from(range.end).is_ok(),
            "Range end exceeds i32::MAX"
        );

        let offset = range.start;
        let size = range.end - range.start;

        Self::invalidate_range(inner, offset, size);

        let shadow = inner.shadow.as_mut().unwrap();
        let ptr = unsafe { shadow.ptr.add(offset) };

        Ok(BufferMappedRange::new(self, ptr, offset, size))
    }

    fn write_mapped_range<R>(&mut self, range: R) -> Result<BufferMappedRangeMut<'_>, DeviceError>
    where
        R: BufferRange,
    {
        let inner: &mut Inner = self.detatched_inner();

        let range = range.range(inner.size);

        assert!(range.start <= range.end, "Invalid range");

        assert!(
            i32::try_from(range.end).is_ok(),
            "Range end exceeds i32::MAX"
        );

        let offset = range.start;
        let size = range.end - range.start;

        Self::invalidate_range(inner, offset, size);

        let shadow = inner.shadow.as_mut().unwrap();
        let ptr = unsafe { shadow.ptr.add(offset) };

        Ok(BufferMappedRangeMut::new(self, ptr, offset, size))
    }

    fn write(&mut self, offset: usize, data: &[u8]) -> Result<(), DeviceError> {
        let inner: &mut Inner = self.detatched_inner();

        assert!(
            offset + data.len() <= inner.size,
            "Write range exceeds buffer size"
        );

        inner.gl.bind_buffer(GL::ARRAY_BUFFER, Some(&inner.buffer));

        inner
            .gl
            .buffer_sub_data_with_i32_and_u8_array(GL::ARRAY_BUFFER, offset as i32, data);

        inner.gl.bind_buffer(GL::ARRAY_BUFFER, None);

        Ok(())
    }

    fn read(&mut self, offset: usize, data: &mut [u8]) -> Result<(), DeviceError> {
        let inner: &mut Inner = self.detatched_inner();

        assert!(
            offset + data.len() <= inner.size,
            "Read range exceeds buffer size"
        );

        inner.gl.bind_buffer(GL::ARRAY_BUFFER, Some(&inner.buffer));

        inner
            .gl
            .get_buffer_sub_data_with_i32_and_u8_array(GL::ARRAY_BUFFER, offset as i32, data);

        inner.gl.bind_buffer(GL::ARRAY_BUFFER, None);

        Ok(())
    }
}

// Implement argument field traits similar to Metal
impl ArgumentsField<Automatic> for Buffer {
    const KIND: ArgumentKind = ArgumentKind::UniformBuffer;
    const SIZE: usize = 1;

    fn bind_vertex(&self, slot: u32, encoder: &mut RenderCommandEncoder) {
        encoder.bind_buffer_base(GL::UNIFORM_BUFFER, slot, Some(self.inner.buffer.clone()));
    }

    fn bind_fragment(&self, slot: u32, encoder: &mut RenderCommandEncoder) {
        encoder.bind_buffer_base(GL::UNIFORM_BUFFER, slot, Some(self.inner.buffer.clone()));
    }
}

impl ArgumentsField<Uniform> for Buffer {
    const KIND: ArgumentKind = ArgumentKind::UniformBuffer;
    const SIZE: usize = 1;

    fn bind_vertex(&self, slot: u32, encoder: &mut RenderCommandEncoder) {
        encoder.bind_buffer_base(GL::UNIFORM_BUFFER, slot, Some(self.inner.buffer.clone()));
    }

    fn bind_fragment(&self, slot: u32, encoder: &mut RenderCommandEncoder) {
        encoder.bind_buffer_base(GL::UNIFORM_BUFFER, slot, Some(self.inner.buffer.clone()));
    }
}

impl ArgumentsField<Storage> for Buffer {
    const KIND: ArgumentKind = ArgumentKind::StorageBuffer;
    const SIZE: usize = 1;

    fn bind_vertex(&self, slot: u32, encoder: &mut RenderCommandEncoder) {
        unimplemented!("WebGL does not support storage buffers");
    }

    fn bind_fragment(&self, slot: u32, encoder: &mut RenderCommandEncoder) {
        unimplemented!("WebGL does not support storage buffers");
    }
}
