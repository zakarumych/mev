use super::{CommandBuffer, Device, Queue};
use crate::{SubmissionSync, SyncError};

/// Shared timeline events are unavailable on WebGPU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Event {
    unavailable: Unavailable,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Unavailable {}



#[mev_proc::hidden_trait]
impl crate::traits::DeviceEvent for Device {
    /// Returns [`SyncError::Unsupported`]; WebGPU exposes no shared timeline events.
    fn new_event(&self, initial_value: u64, name: &str) -> Result<Event, SyncError> {
        Err(SyncError::Unsupported)
    }
}



#[mev_proc::hidden_trait]
impl crate::traits::Event for Event {
    fn value(&self) -> Result<u64, SyncError> {
        match self.unavailable {}
    }

    fn set_value(&self, value: u64) -> Result<(), SyncError> {
        match self.unavailable {}
    }
}



#[mev_proc::hidden_trait]
impl crate::traits::QueueSync for Queue {
    /// Submits command buffers with shared timeline dependencies.
    /// WebGPU accepts only empty synchronization lists.
    fn submit_with_sync<I>(
        &mut self,
        command_buffers: I,
        sync: SubmissionSync<'_>,
    ) -> Result<u64, SyncError>
    where
        I: IntoIterator<Item = CommandBuffer>,
    {
        if !sync.waits.is_empty() || !sync.signals.is_empty() {
            return Err(SyncError::Unsupported);
        }
        self.submit(command_buffers).map_err(Into::into)
    }

    /// Submits a checkpoint with shared timeline dependencies.
    /// WebGPU accepts only empty synchronization lists.
    fn submit_checkpoint_with_sync<I>(
        &mut self,
        command_buffers: I,
        sync: SubmissionSync<'_>,
    ) -> Result<u64, SyncError>
    where
        I: IntoIterator<Item = CommandBuffer>,
    {
        if !sync.waits.is_empty() || !sync.signals.is_empty() {
            return Err(SyncError::Unsupported);
        }
        self.submit_checkpoint(command_buffers).map_err(Into::into)
    }
}
