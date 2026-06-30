use std::ops::Range;
use web_sys::{
    WebGl2RenderingContext as GL, WebGlBuffer, WebGlFramebuffer, WebGlProgram, WebGlRenderbuffer,
    WebGlTexture,
};

pub struct CommandBuffer {
    commands: Vec<Command>,
}

impl CommandBuffer {
    pub fn new() -> Self {
        CommandBuffer {
            commands: Vec::new(),
        }
    }

    pub fn commit(&self, gl: &GL) {
        for command in &self.commands {
            command.execute(gl);
        }
    }
}

pub struct CommandEncoder {
    commands: Vec<Command>,
}

impl CommandEncoder {
    pub fn new() -> Self {
        CommandEncoder {
            commands: Vec::new(),
        }
    }

    pub fn finish(self) -> CommandBuffer {
        CommandBuffer {
            commands: self.commands,
        }
    }
}

pub struct RenderCommandEncoder<'a> {
    commands: &'a mut Vec<Command>,
}

impl<'a> RenderCommandEncoder<'a> {
    pub fn new(commands: &'a mut Vec<Command>) -> Self {
        RenderCommandEncoder { commands }
    }

    pub fn draw_arrays(&mut self, mode: u32, first: i32, count: i32) {
        self.commands
            .push(Command::DrawArrays { mode, first, count });
    }

    pub fn draw_elements(&mut self, mode: u32, count: i32, element_type: u32, offset: i32) {
        self.commands.push(Command::DrawElements {
            mode,
            count,
            element_type,
            offset,
        });
    }

    pub(super) fn bind_buffer_base(
        &mut self,
        target: u32,
        index: u32,
        buffer: Option<WebGlBuffer>,
    ) {
        self.commands.push(Command::BindBufferBase {
            target,
            index,
            buffer,
        });
    }
}

#[hidden_trait::expose]
impl crate::traits::RenderCommandEncoder for RenderCommandEncoder<'_> {
    fn with_pipeline(&mut self, pipeline: &RenderPipeline) {}
}

pub struct CopyCommandEncoder<'a> {
    commands: &'a mut Vec<Command>,
}

#[hidden_trait::expose]
impl crate::traits::CopyCommandEncoder for CopyCommandEncoder<'_> {}

pub struct ComputeCommandEncoder<'a> {
    commands: &'a mut Vec<Command>,
}

#[hidden_trait::expose]
impl crate::traits::ComputeCommandEncoder for ComputeCommandEncoder<'_> {}

pub struct AccelerationStructureCommandEncoder<'a> {
    commands: &'a mut Vec<Command>,
}

impl crate::traits::AccelerationStructureCommandEncoder
    for AccelerationStructureCommandEncoder<'_>
{
}

enum Command {
    BindBufferBase {
        target: u32,
        index: u32,
        buffer: Option<WebGlBuffer>,
    },
    BindTexture {
        target: u32,
        index: u32,
        texture: Option<WebGlTexture>,
    },
    DrawArrays {
        mode: u32,
        first: i32,
        count: i32,
    },
    DrawElements {
        mode: u32,
        count: i32,
        element_type: u32,
        offset: i32,
    },
}

impl Command {
    fn execute(&self, gl: &GL) {
        match *self {
            Command::DrawArrays { mode, first, count } => {
                gl.draw_arrays(mode, first, count);
            }
            Command::DrawElements {
                mode,
                count,
                element_type,
                offset,
            } => {
                gl.draw_elements_with_i32(mode, count, element_type, offset);
            }
            Command::BindBufferBase {
                target,
                index,
                ref buffer,
            } => {
                gl.bind_buffer_base(target, index, buffer.as_ref());
            }
            Command::BindTexture {
                target,
                index,
                texture,
            } => {
                gl.active_texture(GL::TEXTURE0 + index);
                gl.bind_texture(target, texture.as_ref());
            }
        }
    }
}
