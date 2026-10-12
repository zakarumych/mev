use std::{error::Error, fmt};

use crate::{DeviceError, Event};

/// A value on a shared timeline event.
#[derive(Clone, Copy, Debug)]
pub struct EventPoint<'a> {
    pub event: &'a Event,
    pub value: u64,
}

/// Dependencies for a GPU queue submission.
///
/// Waits gate all commands in the submission; signals follow all commands,
/// including when the submission contains no command buffers. A wait completes
/// once its event reaches at least the requested value.
///
/// Signals from CPUs, GPU queues, and I/O queues must execute in strictly
/// increasing value order. Use one producer per timeline, or explicitly order
/// producers with waits. Ordering submission calls alone does not order execution
/// on different queues. Never wait on a value whose producer depends on that wait.
#[derive(Clone, Copy, Debug, Default)]
pub struct SubmissionSync<'a> {
    pub waits: &'a [EventPoint<'a>],
    pub signals: &'a [EventPoint<'a>],
}

/// Error creating, accessing, or submitting timeline synchronization.
#[derive(Clone, Debug)]
pub enum SyncError {
    Unsupported,
    InvalidInput(String),
    Device(DeviceError),
    Failure(String),
}

impl fmt::Display for SyncError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported => f.write_str("timeline events are unsupported"),
            Self::InvalidInput(message) | Self::Failure(message) => f.write_str(message),
            Self::Device(error) => error.fmt(f),
        }
    }
}

impl Error for SyncError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Device(error) => Some(error),
            _ => None,
        }
    }
}

impl From<DeviceError> for SyncError {
    fn from(error: DeviceError) -> Self {
        Self::Device(error)
    }
}

pub(crate) fn validate_submission_sync(sync: SubmissionSync<'_>) -> Result<(), SyncError> {
    for points in [sync.waits, sync.signals] {
        for (index, point) in points.iter().enumerate() {
            if points[..index]
                .iter()
                .any(|other| other.event == point.event)
            {
                return Err(SyncError::InvalidInput(
                    "an event occurs more than once in a submission wait or signal list".into(),
                ));
            }
        }
    }
    for signal in sync.signals {
        if sync
            .waits
            .iter()
            .any(|wait| wait.event == signal.event && signal.value <= wait.value)
        {
            return Err(SyncError::InvalidInput(
                "a submission signal must exceed its wait on the same event".into(),
            ));
        }
    }
    Ok(())
}
