use std::fmt;

use crate::generic::OutOfMemory;

#[derive(Debug)]
pub enum SurfaceError {
    OutOfMemory,
    SurfaceLost,
}

impl From<OutOfMemory> for SurfaceError {
    #[inline(always)]
    fn from(_: OutOfMemory) -> Self {
        SurfaceError::OutOfMemory
    }
}

impl fmt::Display for SurfaceError {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SurfaceError::OutOfMemory => fmt::Display::fmt(&OutOfMemory, f),
            SurfaceError::SurfaceLost => f.write_str("surface lost"),
        }
    }
}

impl std::error::Error for SurfaceError {}
