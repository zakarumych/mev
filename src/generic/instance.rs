use std::fmt;

use super::{feature::Features, queue::QueueFlags, SurfaceError};

/// Error that can occur when creating an instance.
///
/// This signals that backend could not be loaded.
#[derive(Debug)]
pub struct LoadError(pub(crate) crate::backend::LoadErrorKind);

impl fmt::Display for LoadError {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl std::error::Error for LoadError {}

/// Error that can occur when creating device from an instance.
#[derive(Debug)]
pub struct CreateError(pub(crate) crate::backend::CreateErrorKind);

impl fmt::Display for CreateError {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl std::error::Error for CreateError {}

/// Error that can occur when creating device from an instance.
#[derive(Debug)]
pub enum CreateWithSurfaceError {
    CreateError(CreateError),
    SurfaceError(SurfaceError),
}

impl fmt::Display for CreateWithSurfaceError {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CreateWithSurfaceError::CreateError(err) => fmt::Display::fmt(err, f),
            CreateWithSurfaceError::SurfaceError(err) => fmt::Display::fmt(err, f),
        }
    }
}

impl std::error::Error for CreateWithSurfaceError {}

impl From<CreateError> for CreateWithSurfaceError {
    #[inline(always)]
    fn from(err: CreateError) -> Self {
        CreateWithSurfaceError::CreateError(err)
    }
}

impl From<SurfaceError> for CreateWithSurfaceError {
    #[inline(always)]
    fn from(err: SurfaceError) -> Self {
        CreateWithSurfaceError::SurfaceError(err)
    }
}

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
