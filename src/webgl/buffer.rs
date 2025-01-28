use web_sys::WebGl2RenderingContext as GL;

use crate::generic::{ArgumentKind, Automatic, Storage, Uniform};
use super::arguments::ArgumentsField;

use std::{
    fmt,
    hash::{Hash, Hasher},
};

#[derive(Clone)]
pub struct Buffer {
    // Store the WebGL buffer object
    buffer: WebGlBuffer,
    // Track buffer size since WebGL doesn't provide a direct way to query it
    size: usize,
    // Reference to the owning WebGL context
    context: GL,
}

impl Buffer {
    pub(super) fn new(buffer: WebGlBuffer, size: usize, context: GL) -> Self {
        Buffer { buffer, size, context }
    }

    pub(super) fn webgl(&self) -> &WebGlBuffer {
        &self.buffer 
    }
}

impl fmt::Debug for Buffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Buffer")
            .field("buffer", &self.buffer)
            .field("size", &self.size)
            .finish()
    }
}

impl Hash for Buffer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.buffer.clone().unchecked_into::<js_sys::Object>().hash(state);
    }
}

impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        self.buffer == other.buffer
    }
}

impl Eq for Buffer {}

#[hidden_trait::expose]
impl crate::traits::Buffer for Buffer {
    #[inline(always)]
    fn size(&self) -> usize {
        self.size
    }

    #[inline(always)] 
    fn detached(&self) -> bool {
        // WebGL buffers are always attached
        false
    }

    #[cfg_attr(feature = "inline-more", inline(always))]
    unsafe fn write_unchecked(&mut self, offset: usize, data: &[u8]) {
        // Ensure write fits within buffer bounds
        if offset + data.len() > self.size {
            panic!("Buffer write out of bounds");
        }

        // In WebGL we need to bind the buffer before writing
        GL.bind_buffer(GL::ARRAY_BUFFER, Some(&self.buffer));
        GL.buffer_sub_data_with_i32_and_u8_array(
            GL::ARRAY_BUFFER,
            offset as i32,
            data
        );
    }
}

// Implement argument field traits similar to Metal
impl ArgumentsField<Automatic> for Buffer {
    const KIND: ArgumentKind = ArgumentKind::UniformBuffer;
    const SIZE: usize = 1;
    
    fn bind_vertex(&self, slot: u32, gl: &GL) {
        gl.bind_buffer_base(GL::UNIFORM_BUFFER, slot, Some(&self.buffer));
    }

    fn bind_fragment(&self, slot: u32, gl: &GL) {
        gl.bind_buffer_base(GL::UNIFORM_BUFFER, slot, Some(&self.buffer));
    }
}

impl ArgumentsField<Uniform> for Buffer {
    const KIND: ArgumentKind = ArgumentKind::UniformBuffer;
    const SIZE: usize = 1;

    fn bind_vertex(&self, slot: u32, gl: &GL) {
        gl.bind_buffer_base(GL::UNIFORM_BUFFER, slot, Some(&self.buffer)); 
    }

    fn bind_fragment(&self, slot: u32, gl: &GL) {
        gl.bind_buffer_base(GL::UNIFORM_BUFFER, slot, Some(&self.buffer));
    }
}

impl ArgumentsField<Storage> for Buffer {
    const KIND: ArgumentKind = ArgumentKind::StorageBuffer;
    const SIZE: usize = 1;

    fn bind_vertex(&self, slot: u32, gl: &GL) {
        gl.bind_buffer_base(GL::SHADER_STORAGE_BUFFER, slot, Some(&self.buffer));
    }

    fn bind_fragment(&self, slot: u32, gl: &GL) {
        gl.bind_buffer_base(GL::SHADER_STORAGE_BUFFER, slot, Some(&self.buffer));
    }
}
impl Buffer {
    // Get buffer contents as bytes
    pub fn contents(&self) -> Vec<u8> {
        let mut data = vec![0; self.size];
        GL.bind_buffer(GL::ARRAY_BUFFER, Some(&self.buffer));
        GL.get_buffer_sub_data_with_i32_and_u8_array_and_dst_offset(
            GL::ARRAY_BUFFER,
            0,
            &mut data,
            0
        );
        data
    }
    
    pub fn options() -> BufferOptions {
        BufferOptions {
            storage_mode: StorageMode::Shared,
            resource_options: ResourceOptions::default() 
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BufferOptions {
    pub storage_mode: StorageMode,
    pub resource_options: ResourceOptions,
}

#[derive(Debug, Clone, Copy)]
pub enum StorageMode {
    Shared,
    Private,
    Managed,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ResourceOptions {
    pub cpu_cache_mode: CpuCacheMode,
    pub storage_mode: StorageMode,
}

#[derive(Debug, Clone, Copy)]
pub enum CpuCacheMode {
    DefaultCache,
    WriteCombined,
}

impl Default for CpuCacheMode {
    fn default() -> Self {
        CpuCacheMode::DefaultCache
    }
}

impl Default for StorageMode {
    fn default() -> Self {
        StorageMode::Shared
    }
}