use std::fmt;

use super::{feature::Features, queue::QueueFlags, OutOfMemory, SurfaceError};

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
    ///
    /// Device created will use physical device at that index in [`Capabilities::devices`].
    pub idx: usize,

    /// Specifies families from which queues should be created.
    /// Same family may be specified more than once, up to maximum number of queues in that family. See [`FamilyCapabilities::queue_count`].
    pub queues: &'a [u32],

    /// List of features that should be enabled.
    ///
    /// It should not include features not supported by the device. See [`DeviceCapabilities::features`].
    pub features: Features,
}
