use std::{
    fmt,
    hash::{Hash, Hasher},
    ptr::NonNull,
    sync::Arc,
};

use web_sys::{WebGl2RenderingContext as GL, WebGlTexture};

use crate::{
    generic::{
        ArgumentKind, Automatic, BufferMappedRange, BufferMappedRangeMut, BufferRange, BufferUsage,
        DeviceError, Storage, Uniform,
    },
    ImageExtent, ImageUsage, PixelFormat, RenderCommandEncoder,
};

use super::arguments::ArgumentsField;

struct Inner {
    // Store the WebGL buffer object
    texture: WebGlTexture,
    format: PixelFormat,
    extent: ImageExtent,
    usage: ImageUsage,
    name: Box<str>,
    // Reference to the owning WebGL context
    gl: GL,
}

#[derive(Clone)]
pub struct Image {
    inner: Arc<Inner>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.gl.delete_texture(Some(&self.texture));
    }
}

impl Image {
    pub(super) fn new(
        texture: WebGlTexture,
        format: PixelFormat,
        extent: ImageExtent,
        usage: ImageUsage,
        name: Box<str>,
        gl: GL,
    ) -> Self {
        Image {
            inner: Arc::new(Inner {
                texture,
                format,
                extent,
                usage,
                name,
                gl,
            }),
        }
    }

    pub(super) fn webgl(&self) -> &WebGlTexture {
        &self.inner.texture
    }
}

impl fmt::Debug for Image {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            f.debug_struct("Image")
                .field("handle", &self.inner.texture)
                .field("extent", &self.inner.extent)
                .field("usage", &self.inner.usage)
                .field("name", &self.inner.name)
                .finish()
        } else {
            f.debug_tuple("Image").field(&self.inner.texture).finish()
        }
    }
}

impl Hash for Image {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::ptr::hash(&*self.inner, state);
    }
}

impl PartialEq for Image {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(&*self.inner, &*other.inner)
    }
}

impl Eq for Image {}

impl crate::traits::Resource for Image {}

#[hidden_trait::expose]
impl crate::traits::Image for Image {
    fn format(&self) -> PixelFormat {
        self.inner.format
    }

    fn extent(&self) -> ImageExtent {
        self.inner.extent
    }
}

// Implement argument field traits similar to Metal
impl ArgumentsField<Automatic> for Image {
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
