use std::fmt;

use crate::generic::{DeviceError, OutOfMemory};

/// Error that can occur when working with a surface.
#[derive(Debug)]
pub enum SurfaceError {
    SurfaceLost,
    OutOfMemory,
    DeviceLost,
}

impl From<OutOfMemory> for SurfaceError {
    #[inline(always)]
    fn from(_: OutOfMemory) -> Self {
        SurfaceError::OutOfMemory
    }
}

impl From<DeviceError> for SurfaceError {
    #[inline(always)]
    fn from(err: DeviceError) -> Self {
        match err {
            DeviceError::OutOfMemory => SurfaceError::OutOfMemory,
            DeviceError::DeviceLost => SurfaceError::DeviceLost,
        }
    }
}

impl fmt::Display for SurfaceError {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SurfaceError::SurfaceLost => f.write_str("surface lost"),
            SurfaceError::OutOfMemory => fmt::Display::fmt(&OutOfMemory, f),
            SurfaceError::DeviceLost => f.write_str("device lost"),
        }
    }
}

impl std::error::Error for SurfaceError {}
