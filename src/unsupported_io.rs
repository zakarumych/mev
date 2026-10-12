//! Native file I/O is unavailable on this backend.
use std::convert::Infallible;

use crate::{AsBufferSlice, AsImageSlice, IoError, TexelShape};

/// Independently recorded file I/O operations.
#[derive(Debug)]
pub struct IoBatch {
    unavailable: Infallible,
}

/// Aggregate completion handle for submitted file I/O operations.
#[derive(Debug)]
pub struct IoStatus {
    unavailable: Infallible,
}

#[mev_proc::hidden_trait]
impl crate::traits::IoBatch for IoBatch {
    /// Record a file byte range into a buffer slice.
    /// The source range length must equal the destination size.
    /// Validation and file-opening errors are recorded in the submitted status.
    fn load_buffer(
        &mut self,
        path: impl AsRef<std::path::Path>,
        source: std::ops::Range<u64>,
        dst: impl AsBufferSlice,
    ) {
        match self.unavailable {}
    }

    /// Record a file byte range into one image layer using the supplied layout.
    /// The source range must contain the layout's byte span.
    /// Validation and file-opening errors are recorded in the submitted status.
    fn load_image<const N: usize>(
        &mut self,
        path: impl AsRef<std::path::Path>,
        source: std::ops::Range<u64>,
        shape: TexelShape<N>,
        dst: &impl AsImageSlice<N>,
    ) {
        match self.unavailable {}
    }

    /// Submit this batch and return its aggregate completion handle.
    /// Loads may execute concurrently: avoid overlapping writes and CPU/GPU access
    /// to destinations until completion. Dropping the handle does not cancel loads.
    fn submit(self) -> IoStatus {
        match self.unavailable {}
    }
}

#[mev_proc::hidden_trait]
impl crate::traits::IoStatus for IoStatus {
    /// Return `None` while pending, or all operation results in recording order.
    /// Each attempted load has one result, including validation and file-opening failures.
    /// An empty batch completes with an empty result slice.
    fn poll(&self) -> Option<&[Result<(), IoError>]> {
        match self.unavailable {}
    }

    /// Block until all operations finish and return their results in recording order.
    /// Check results before consuming loaded resource contents.
    fn wait(&self) -> &[Result<(), IoError>] {
        match self.unavailable {}
    }
}

#[cfg(all(test, mev_backend = "vulkan"))]
mod tests {
    use crate::{DeviceDesc, Features, Instance, QueueFlags};

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn native_io_requires_enabled_feature() {
        let instance = Instance::load().expect("Vulkan instance");
        let (idx, family) = instance
            .capabilities()
            .devices
            .iter()
            .enumerate()
            .find_map(|(idx, caps)| {
                assert!(!caps.features.contains(Features::IO_COMMANDS));
                caps.families
                    .iter()
                    .position(|family| family.queue_flags.contains(QueueFlags::TRANSFER))
                    .map(|family| (idx, family as u32))
            })
            .expect("Vulkan transfer queue");
        let (device, _queues) = instance
            .new_device(DeviceDesc {
                idx,
                queues: &[family],
                features: Features::empty(),
            })
            .expect("Vulkan device");
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            device.new_io_batch();
        }))
        .is_err());
    }
}
