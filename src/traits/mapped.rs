use crate::DeviceError;

pub(crate) trait BufferMappedRange {
    /// Copies bytes from mapped memory into `data`.
    fn read(&self, offset: usize, data: &mut [u8]);
}

pub(crate) trait BufferMappedRangeMut {
    /// Copies bytes into mapped memory.
    fn write(&mut self, offset: usize, data: &[u8]);

    /// Flushes host writes to the device.
    fn flush(&mut self) -> Result<(), DeviceError>;
}

pub(crate) trait BufferMappedRangeMutRead {
    /// Copies bytes from mapped memory into `data`.
    fn read(&self, offset: usize, data: &mut [u8]);
}
