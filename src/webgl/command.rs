use std::ops::Range;
use web_sys::{WebGl2RenderingContext as GL, WebGlBuffer, WebGlFramebuffer, WebGlProgram, WebGlRenderbuffer, WebGlTexture};

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
        self.commands.push(Command::DrawArrays { mode, first, count });
    }

    pub fn draw_elements(&mut self, mode: u32, count: i32, element_type: u32, offset: i32) {
        self.commands.push(Command::DrawElements { mode, count, element_type, offset });
    }
}

enum Command {
    DrawArrays { mode: u32, first: i32, count: i32 },
    DrawElements { mode: u32, count: i32, element_type: u32, offset: i32 },
}

impl Command {
    fn execute(&self, gl: &GL) {
        match self {
            Command::DrawArrays { mode, first, count } => {
                gl.draw_arrays(*mode, *first, *count);
            }
            Command::DrawElements { mode, count, element_type, offset } => {
                gl.draw_elements(*mode, *count, *element_type, *offset);
            }
        }
    }
}