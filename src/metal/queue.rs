use std::{fmt, ops::Deref};

use mev_proc::hidden_trait;
use objc2::{
    Message,
    rc::{Retained, Weak},
    runtime::ProtocolObject,
};
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue};

use crate::generic::{AsBufferSlice, DeviceError, PipelineStages, SubmissionSync, SyncError};

use super::{CommandBuffer, CommandEncoder, Device, Frame};

/// Maximum number of pending epochs to keep in queue.
/// Queue will wait for earliest epoch to be complete and reuse it
/// when number of epochs exceeds this limit.
///
/// The number is chosen to minimize waiting (ideally epoch would be already complete when it's recycled)
/// and to minimize memory usage (epoch contains resources that are not released until it's complete).
const MAX_EPOCHS: usize = 3;

pub(super) const MAX_COMMAND_BUFFERS: usize = 64;

struct Epoch {
    last_command_buffer: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    epoch_id: u64,
}

pub struct Queue {
    device: Device,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    next_epoch_id: u64,
    current_epoch: Option<Epoch>,
    pending_epochs: Vec<Epoch>,
    last_finished_epoch: u64,
    allocated_buffers: Vec<Weak<ProtocolObject<dyn MTLCommandBuffer>>>,
}

unsafe impl Send for Queue {}
unsafe impl Sync for Queue {}

impl fmt::Debug for Queue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Queue")
            .field("device", &self.device.metal())
            .field("queue", &Retained::as_ptr(&self.queue))
            .finish()
    }
}

impl Queue {

    fn submit_sync_impl<I>(
        &mut self,
        command_buffers: I,
        sync: SubmissionSync<'_>,
        checkpoint: bool,
    ) -> Result<u64, SyncError>
    where
        I: IntoIterator<Item = CommandBuffer>,
    {
        if sync.waits.is_empty() && sync.signals.is_empty() {
            return self
                .submit_impl(command_buffers, checkpoint)
                .map_err(Into::into);
        }
        crate::generic::validate_submission_sync(sync)?;
        for point in sync.waits.iter().chain(sync.signals) {
            if !std::ptr::eq(self.device.metal(), point.event.device()) {
                return Err(SyncError::InvalidInput(
                    "event belongs to another device".into(),
                ));
            }
        }
        for point in sync.signals {
            if point.value <= point.event.value()? {
                return Err(SyncError::InvalidInput(
                    "event signal must exceed its current value".into(),
                ));
            }
        }
        let commands: Vec<_> = command_buffers.into_iter().collect();
        for command in &commands {
            if !std::ptr::eq(self.device.metal(), command.device().as_ref()) {
                return Err(SyncError::InvalidInput(
                    "command buffer belongs to another device".into(),
                ));
            }
            if Retained::as_ptr(&command.queue()) != Retained::as_ptr(&self.queue) {
                return Err(SyncError::InvalidInput(
                    "command buffer belongs to another queue".into(),
                ));
            }
        }
        self.prune_allocated_buffers();
        let helpers = usize::from(!sync.waits.is_empty()) + usize::from(!sync.signals.is_empty());
        if self.allocated_buffers.len() + helpers > MAX_COMMAND_BUFFERS {
            return Err(SyncError::InvalidInput(
                "Metal command buffer capacity cannot hold submission synchronization boundaries"
                    .into(),
            ));
        }
        // Retaining command buffers keep native event references alive through execution.
        // Both boundary buffers are allocated before any command is committed.
        let prefix = if sync.waits.is_empty() {
            None
        } else {
            let buffer = self.queue.commandBuffer().ok_or_else(|| {
                SyncError::Failure("Failed to create event wait command buffer".into())
            })?;
            self.allocated_buffers.push(Weak::from_retained(&buffer));
            for point in sync.waits {
                buffer.encodeWaitForEvent_value(point.event.metal().as_ref(), point.value);
            }
            Some(CommandBuffer::from_metal(buffer))
        };
        let suffix = if sync.signals.is_empty() {
            None
        } else {
            let buffer = self.queue.commandBuffer().ok_or_else(|| {
                SyncError::Failure("Failed to create event signal command buffer".into())
            })?;
            self.allocated_buffers.push(Weak::from_retained(&buffer));
            for point in sync.signals {
                buffer.encodeSignalEvent_value(point.event.metal().as_ref(), point.value);
            }
            Some(CommandBuffer::from_metal(buffer))
        };
        self.submit_impl(prefix.into_iter().chain(commands).chain(suffix), checkpoint)
            .map_err(Into::into)
    }

    fn prune_allocated_buffers(&mut self) {
        self.allocated_buffers.retain(|weak| {
            weak.load().is_some_and(|buffer| {
                !matches!(
                    buffer.status(),
                    MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
                )
            })
        });
    }

    pub(super) fn new(
        device: Device,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    ) -> Self {
        Queue {
            device,
            queue,
            next_epoch_id: 1,
            current_epoch: None,
            pending_epochs: Vec::new(),
            last_finished_epoch: 0,
            allocated_buffers: Vec::new(),
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
                        cbuf.waitUntilCompleted();
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

#[mev_proc::hidden_trait]
impl crate::traits::QueueSync for Queue {
    /// Submits commands with waits before all commands and signals after all commands.
    /// Signals must execute in strictly increasing order across all event producers.
    /// Returns `InvalidInput` if native command buffer capacity cannot hold the
    /// submission's synchronization boundaries. Dropped encoders may occupy native
    /// capacity until their autorelease pool drains.
    fn submit_with_sync<I>(
        &mut self,
        command_buffers: I,
        sync: SubmissionSync<'_>,
    ) -> Result<u64, SyncError>
    where
        I: IntoIterator<Item = CommandBuffer>,
    {
        self.submit_sync_impl(command_buffers, sync, false)
    }

    /// Submits synchronized commands and closes their completion epoch.
    fn submit_checkpoint_with_sync<I>(
        &mut self,
        command_buffers: I,
        sync: SubmissionSync<'_>,
    ) -> Result<u64, SyncError>
    where
        I: IntoIterator<Item = CommandBuffer>,
    {
        self.submit_sync_impl(command_buffers, sync, true)
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
    /// Requires `BUFFER_WRITE_QUEUE`, which Metal does not support. Panics.
    fn write_buffer_raw(&mut self, _slice: impl AsBufferSlice, _data: &[u8]) {
        panic!("BUFFER_WRITE_QUEUE is not supported in the Metal backend");
    }

    /// Requires `BUFFER_WRITE_QUEUE`, which Metal does not support. Panics.
    fn write_buffer(&mut self, slice: impl AsBufferSlice, data: &impl bytemuck::NoUninit) {
        self.write_buffer_raw(slice, bytemuck::bytes_of(data));
    }

    /// Requires `BUFFER_WRITE_QUEUE`, which Metal does not support. Panics.
    fn write_buffer_slice(&mut self, slice: impl AsBufferSlice, data: &[impl bytemuck::NoUninit]) {
        self.write_buffer_raw(slice, bytemuck::cast_slice(data));
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn family(&self) -> u32 {
        0
    }

    fn new_command_encoder(&mut self) -> CommandEncoder {
        self.prune_allocated_buffers();
        let buffer = self
            .queue
            .commandBuffer()
            .expect("Failed to create command buffer");
        self.prune_allocated_buffers();
        self.allocated_buffers.push(Weak::from_retained(&buffer));
        CommandEncoder::new(self.device.metal().retain(), buffer)
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

    fn sync_frame(&mut self, _frame: &mut Frame, _next_use: PipelineStages) {}

    fn begin_frame(&mut self, frame: &mut Frame, next_use: PipelineStages) -> CommandEncoder {
        self.new_command_encoder()
    }

    fn wait_idle(&self) -> Result<(), DeviceError> {
        for epoch in &self.pending_epochs {
            if let Some(cbuf) = &epoch.last_command_buffer {
                cbuf.waitUntilCompleted();
            }
        }
        if let Some(cbuf) = self
            .current_epoch
            .as_ref()
            .and_then(|epoch| epoch.last_command_buffer.as_ref())
        {
            cbuf.waitUntilCompleted();
        }
        Ok(())
    }

    fn last_finished_epoch(&self) -> u64 {
        self.last_finished_epoch
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DeviceDesc, EventPoint, Features};

    #[test]
    #[ignore = "requires a native Metal device supporting shared events"]
    fn synchronization_rejects_exhausted_native_capacity() {
        let instance = super::super::Instance::load().expect("Metal instance");
        let idx = instance
            .capabilities()
            .devices
            .iter()
            .position(|device| device.features.contains(Features::TIMELINE_EVENTS))
            .expect("shared event capable device");
        let (device, queues) = instance
            .new_device(DeviceDesc {
                idx,
                queues: &[0],
                features: Features::TIMELINE_EVENTS,
            })
            .expect("Metal device");
        let mut queue = queues.into_iter().next().expect("GPU queue");
        let event = device.new_event(0, "capacity timeline").expect("event");
        let waits = [EventPoint {
            event: &event,
            value: 0,
        }];
        let signals = [EventPoint {
            event: &event,
            value: 1,
        }];
        let sync = SubmissionSync {
            waits: &waits,
            signals: &signals,
        };
        let mut held = objc2::rc::autoreleasepool(|_| {
            (0..63)
                .map(|_| queue.new_command_encoder().finish())
                .collect::<Vec<_>>()
        });
        let submitted = held.pop().expect("held command buffer");
        // Other held buffers count even though they are outside this submission.
        assert!(matches!(
            queue.submit_with_sync([submitted], sync),
            Err(SyncError::InvalidInput(_))
        ));
        assert_eq!(event.value().expect("unsignaled event"), 0);
        drop(held);
        objc2::rc::autoreleasepool(|_| {
            queue
                .submit_with_sync([], sync)
                .expect("capacity reclaimed");
            queue.wait_idle().expect("sync completion");
        });
        assert_eq!(event.value().expect("completed event"), 1);
    }
}
