use std::{fmt, ops::Deref, rc::Rc};

use crate::generic::{DeviceError, PipelineStages};

use super::{CommandBuffer, CommandEncoder, Device, Frame};

pub struct Queue {
    device: Device,
    queue: wgpu::Queue,
    epoch: u64,
}

impl Queue {
    pub(super) fn new(device: Device, queue: wgpu::Queue) -> Self {
        Queue {
            device,
            queue,
            epoch: 0,
        }
    }
}

impl fmt::Debug for Queue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Queue").field("epoch", &self.epoch).finish()
    }
}

impl Deref for Queue {
    type Target = Device;

    #[inline(always)]
    fn deref(&self) -> &Device {
        &self.device
    }
}

impl crate::traits::Resource for Queue {}

#[hidden_trait::expose]
impl crate::traits::Queue for Queue {
    fn device(&self) -> &Device {
        &self.device
    }

    fn family(&self) -> u32 {
        0
    }

    fn new_command_encoder(&mut self) -> CommandEncoder {
        let encoder = self
            .device
            .wgpu()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        CommandEncoder::new(encoder, self.device.clone(), self.queue.clone())
    }

    fn submit<I>(&mut self, command_buffers: I) -> Result<u64, DeviceError>
    where
        I: IntoIterator<Item = CommandBuffer>,
    {
        let mut wgpu_buffers = Vec::new();
        let mut surface_textures = Vec::new();

        for cb in command_buffers {
            wgpu_buffers.push(cb.buffer);
            surface_textures.extend(cb.surface_textures);
        }

        self.queue.submit(wgpu_buffers);

        for st in surface_textures {
            st.present();
        }

        self.epoch += 1;
        Ok(self.epoch)
    }

    fn submit_checkpoint<I>(&mut self, command_buffers: I) -> Result<u64, DeviceError>
    where
        I: IntoIterator<Item = CommandBuffer>,
    {
        self.submit(command_buffers)
    }

    fn sync_frame(&mut self, _frame: &mut Frame, _before: PipelineStages) {}

    fn wait_idle(&self) -> Result<(), DeviceError> {
        let result = self.device.wgpu().poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        match result {
            Ok(_) => Ok(()),
            Err(wgpu::PollError::Timeout) => Err(DeviceError::DeviceLost),
            Err(wgpu::PollError::WrongSubmissionIndex(_, _)) => unreachable!(),
        }
    }

    fn last_finished_epoch(&self) -> u64 {
        // wgpu handles sync internally; return current epoch as best estimate
        self.epoch
    }
}
