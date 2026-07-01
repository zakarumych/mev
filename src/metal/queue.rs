use std::{fmt, ops::Deref};

use foreign_types::ForeignType;

use crate::{
    backend::from::MetalInto,
    generic::{DeviceError, OutOfMemory, PipelineStages},
};

use super::{CommandBuffer, CommandEncoder, Device, Frame};

/// Maximum number of pending epochs to keep in queue.
/// Queue will wait for earliest epoch to be complete and reuse it
/// when number of epochs exceeds this limit.
///
/// The number is chosen to minimize waiting (ideally epoch would be already complete when it's recycled)
/// and to minimize memory usage (epoch contains resources that are not released until it's complete).
const MAX_EPOCHS: usize = 3;

struct Epoch {
    last_command_buffer: Option<metal::CommandBuffer>,
    epoch_id: u64,
}

pub struct Queue {
    device: Device,
    queue: metal::CommandQueue,
    next_epoch_id: u64,
    current_epoch: Option<Epoch>,
    pending_epochs: Vec<Epoch>,
    last_finished_epoch: u64,
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
        Queue {
            device,
            queue,
            next_epoch_id: 1,
            current_epoch: None,
            pending_epochs: Vec::new(),
            last_finished_epoch: 0,
        }
    }

    fn submit_impl<I>(&mut self, command_buffers: I, checkpoint: bool) -> Result<u64, DeviceError>
    where
        I: IntoIterator<Item = CommandBuffer>,
    {
        let current_epoch = {
            if self.current_epoch.is_none() {
                if self.pending_epochs.len() >= MAX_EPOCHS {
                    if let Some(cbuf) = &self.pending_epochs[0].last_command_buffer {
                        cbuf.wait_until_completed();
                    }
                    self.last_finished_epoch = self.pending_epochs[0].epoch_id;
                    self.pending_epochs.remove(0);
                }

                self.current_epoch = Some(Epoch {
                    last_command_buffer: None,
                    epoch_id: self.next_epoch_id,
                });
                self.next_epoch_id += 1;
            }

            self.current_epoch.as_mut().unwrap()
        };

        let last_cbuf = command_buffers
            .into_iter()
            .map(CommandBuffer::commit)
            .last();

        if let Some(last_cbuf) = last_cbuf {
            current_epoch.last_command_buffer = Some(last_cbuf);
        }

        let epoch_id = current_epoch.epoch_id;

        if checkpoint {
            self.pending_epochs.push(self.current_epoch.take().unwrap());
        }

        Ok(epoch_id)
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
        CommandEncoder::new(
            self.device.metal().to_owned(),
            self.queue.new_command_buffer().to_owned(),
        )
    }

    fn submit<I>(&mut self, command_buffers: I) -> Result<u64, DeviceError>
    where
        I: IntoIterator<Item = CommandBuffer>,
    {
        self.submit_impl(command_buffers, false)
    }

    fn submit_checkpoint<I>(&mut self, command_buffers: I) -> Result<u64, DeviceError>
    where
        I: IntoIterator<Item = crate::backend::CommandBuffer>,
    {
        self.submit_impl(command_buffers, true)
    }

    fn sync_frame(&mut self, _frame: &mut Frame, _before: PipelineStages) {}

    fn wait_idle(&self) -> Result<(), DeviceError> {
        for epoch in &self.pending_epochs {
            if let Some(cbuf) = &epoch.last_command_buffer {
                cbuf.wait_until_completed();
            }
        }
        Ok(())
    }

    fn last_finished_epoch(&self) -> u64 {
        self.last_finished_epoch
    }
}
