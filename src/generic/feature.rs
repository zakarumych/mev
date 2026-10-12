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

        /// Buffers can be mapped after creation using `Buffer::map`.
        /// Host access requires `BufferUsage::HOST_READ` or `HOST_WRITE`.
        const BUFFER_MAP = 0x0004;

        /// Buffers can be created mapped using `Device::new_buffer_mapped`.
        /// Initial writes do not require `BufferUsage::HOST_WRITE`.
        const BUFFER_MAP_AT_CREATION = 0x0008;

        /// Copy encoders support native buffer upload commands.
        const BUFFER_WRITE_COMMAND = 0x0010;

        /// Queues support native buffer upload operations.
        const BUFFER_WRITE_QUEUE = 0x0020;

        /// Device-owned I/O batches can load file ranges into GPU buffers and images.
        const IO_COMMANDS = 0x0040;

        /// Shared timeline events synchronize GPU queue submissions.
        const TIMELINE_EVENTS = 0x0080;

        /// If this feature is enabled, the device will support BC texture compression formats.
        const TEXTURE_COMPRESSION_BC = 0x0000_0000_0000_0000_0000_0000_0000_0002;
    }
}
