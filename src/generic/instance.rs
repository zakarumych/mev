use std::fmt;

use super::{feature::Features, queue::QueueFlags};

#[derive(Debug)]
pub struct LoadError(pub(crate) crate::backend::LoadErrorKind);

impl fmt::Display for LoadError {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl std::error::Error for LoadError {}

#[derive(Debug)]
pub struct CreateError(pub(crate) crate::backend::CreateErrorKind);

impl fmt::Display for CreateError {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl std::error::Error for CreateError {}

/// Capabilities of a queue family of specific device.
#[derive(Clone, Debug)]
pub struct FamilyCapabilities {
    /// Flags that describe the capabilities of the queue family.
    pub queue_flags: QueueFlags,

    /// Number of queues that can be created in the queue family.
    pub queue_count: usize,
}

/// Capabilities of the specific device.
#[derive(Clone, Debug)]
pub struct DeviceCapabilities {
    /// List of features that are supported by the device.
    pub features: Features,

    /// List of queue families capabilities.
    pub families: Vec<FamilyCapabilities>,
}

/// Capabilities of the devices.
#[derive(Clone, Debug)]
pub struct Capabilities {
    pub devices: Vec<DeviceCapabilities>,
}

/// Specifies how the device should be created.
pub struct DeviceDesc<'a> {
    /// Index of the device.
    pub idx: usize,

    /// List of families to request queues from.
    pub queues: &'a [u32],

    /// List of features that should be enabled.
    pub features: Features,
}
