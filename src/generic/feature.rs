bitflags::bitflags! {
    /// Set of features that can be requested from the device.
    /// The device creation will fail if the device does not support all of the requested features.
    /// Check the capabilities of the device to see which features are supported.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct Features: u128 {
        /// If this feature is enabled, surfaces can be created by a device.
        ///
        /// See [`Device::new_surface`](crate::Device::new_surface).
        const SURFACE = 0x0000_0000_0000_0000_0000_0000_0000_0001;
    }
}
