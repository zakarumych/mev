use std::{
    fmt,
    hash::{Hash, Hasher},
};

use foreign_types::ForeignType;

use crate::generic::{ArgumentKind, Automatic, Storage, Uniform};

use super::{arguments::ArgumentsField, out_of_bounds};

#[derive(Clone)]
#[repr(transparent)]
pub struct Buffer {
    buffer: metal::Buffer,
}

impl Buffer {
    pub(super) fn new(buffer: metal::Buffer) -> Self {
        Buffer { buffer }
    }

    pub(super) fn metal(&self) -> &metal::BufferRef {
        &self.buffer
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
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.buffer.as_ptr().hash(state);
    }
}

impl PartialEq for Buffer {
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
    fn detached(&self) -> bool {
        use foreign_types::ForeignType;
        use metal::NSUInteger;
        use objc::*;

        let count: NSUInteger = unsafe { msg_send![(self.buffer.as_ptr()), retainCount] };
        count == 1
    }

    #[cfg_attr(feature = "inline-more", inline(always))]
    unsafe fn write_unchecked(&mut self, offset: usize, data: &[u8]) {
        let length = self.buffer.length();
        let fits = match (u64::try_from(offset), u64::try_from(data.len())) {
            (Ok(off), Ok(len)) => match off.checked_add(len) {
                Some(end) => end <= length,
                None => false,
            },
            _ => false,
        };
        if !fits {
            out_of_bounds();
        }
        unsafe {
            let ptr = self.buffer.contents().add(offset as usize);
            ptr.cast::<u8>()
                .copy_from_nonoverlapping(data.as_ptr(), data.len());
            self.buffer.did_modify_range(metal::NSRange {
                location: offset as _,
                length: data.len() as _,
            })
        }
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
