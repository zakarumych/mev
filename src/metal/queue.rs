use std::{fmt, ops::Deref};

use foreign_types::ForeignType;

use crate::generic::{DeviceError, OutOfMemory, PipelineStages};

use super::{CommandBuffer, CommandEncoder, Device, Frame};

pub struct Queue {
    device: Device,
    queue: metal::CommandQueue,
}

unsafe impl Send for Queue {}
unsafe impl Sync for Queue {}

impl fmt::Debug for Queue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Queue")
            .field("device", &self.device.metal())
            .field("queue", &self.queue.as_ptr())
            .finish()
    }
}

impl Queue {
    pub(super) fn new(device: Device, queue: metal::CommandQueue) -> Self {
        Queue { device, queue }
    }
}

impl Deref for Queue {
    type Target = Device;

    #[inline(always)]
    fn deref(&self) -> &Device {
        &self.device
    }
}

#[hidden_trait::expose]
impl crate::traits::Queue for Queue {
    fn device(&self) -> &Device {
        &self.device
    }

    fn family(&self) -> u32 {
        0
    }

    fn new_command_encoder(&mut self) -> Result<CommandEncoder, OutOfMemory> {
        Ok(CommandEncoder::new(
            self.device.metal().to_owned(),
            self.queue.new_command_buffer().to_owned(),
        ))
    }

    fn submit<I>(&mut self, command_buffers: I, _check_point: bool) -> Result<(), DeviceError>
    where
        I: IntoIterator<Item = CommandBuffer>,
    {
        command_buffers.into_iter().for_each(CommandBuffer::commit);
        Ok(())
    }

    /// Drop command buffers without submitting them to the queue.
    fn drop_command_buffer<I>(&mut self, command_buffers: I)
    where
        I: IntoIterator<Item = CommandBuffer>,
    {
        command_buffers.into_iter().for_each(drop);
    }

    fn sync_frame(&mut self, _frame: &mut Frame, _before: PipelineStages) {}
}
