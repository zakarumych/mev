use std::fmt;

use super::{OutOfMemory, SurfaceError, feature::Features, queue::QueueFlags};

/// Capabilities of a queue family of specific device.
#[derive(Clone, Debug)]
pub struct FamilyCapabilities {
    /// Flags that describe the capabilities of the queue family.
    pub queue_flags: QueueFlags,

    /// Number of queues that can be created in the queue family.
    pub queue_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// Unknown device type.
    Unknown = 0,

    /// Integrated GPU device type.
    IntegratedGpu = 1,

    /// Discrete GPU device type.
    DiscreteGpu = 2,

    /// Virtual GPU device type.
    VirtualGpu = 3,

    /// CPU device type.
    Cpu = 4,
}

/// Capabilities of the specific device.
#[derive(Clone, Debug)]
pub struct DeviceCapabilities {
    /// Name of the device.
    pub name: String,

    /// Performance profile of the device.
    pub device_type: DeviceType,

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
