use std::{fmt, ops::Deref, rc::Rc};

use mev_proc::hidden_trait;

use crate::generic::{AsBufferSlice, BufferUsage, DeviceError, PipelineStages};

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

#[hidden_trait]
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
        CommandEncoder::new(encoder, self.device.clone())
    }

    /// Uploads bytes with the native queue operation. Requires `BUFFER_WRITE_QUEUE`.
    ///
    /// Copies `data` before returning. Uploads execute before the command buffers
    /// in the next submission, including an empty submission; encoder barriers
    /// cannot order commands before these uploads. The destination must have
    /// `TRANSFER_DST` usage, and its offset and size must be multiples of four.
    /// The data length must equal the slice size, including for empty writes.
    fn write_buffer_raw(&mut self, slice: impl AsBufferSlice, data: &[u8]) {
        let slice = slice.as_buffer_slice();
        assert_eq!(
            data.len(),
            slice.size(),
            "data length does not match buffer slice size"
        );
        assert!(
            slice.buffer().usage().contains(BufferUsage::TRANSFER_DST),
            "queue buffer writes require TRANSFER_DST usage"
        );
        assert_eq!(
            slice.offset() % 4, 0,
            "queue buffer write offset must be 4-byte aligned"
        );
        assert_eq!(
            slice.size() % 4, 0,
            "queue buffer write size must be 4-byte aligned"
        );
        let end = slice
            .offset()
            .checked_add(slice.size())
            .expect("queue buffer write range overflow");
        assert!(
            end <= slice.buffer().size(),
            "queue buffer write exceeds buffer size"
        );
        if !data.is_empty() {
            self.queue
                .write_buffer(slice.buffer().wgpu(), slice.offset() as u64, data);
        }
    }

    /// Uploads a value with [`Self::write_buffer_raw`] and the same ordering rules.
    fn write_buffer(&mut self, slice: impl AsBufferSlice, data: &impl bytemuck::NoUninit) {
        self.write_buffer_raw(slice, bytemuck::bytes_of(data));
    }

    /// Uploads a slice with [`Self::write_buffer_raw`] and the same ordering rules.
    fn write_buffer_slice(&mut self, slice: impl AsBufferSlice, data: &[impl bytemuck::NoUninit]) {
        self.write_buffer_raw(slice, bytemuck::cast_slice(data));
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
            self.queue.present(st);
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

    fn sync_frame(&mut self, _frame: &mut Frame, _next_use: PipelineStages) {}

    fn begin_frame(&mut self, frame: &mut Frame, next_use: PipelineStages) -> CommandEncoder {
        self.new_command_encoder()
    }

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
