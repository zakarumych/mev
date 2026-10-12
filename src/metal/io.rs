#[path = "io/layout.rs"]
mod layout;

use std::{
    ffi::CString,
    ops::Range,
    os::unix::ffi::OsStrExt,
    path::Path,
    ptr::NonNull,
    sync::{Arc, Mutex, OnceLock},
};

use objc2::{rc::Retained, runtime::ProtocolObject, sel};
use objc2_foundation::{NSObjectProtocol, NSURL};
use objc2_metal::{
    MTLDevice, MTLIOCommandBuffer, MTLIOCommandQueue, MTLIOCommandQueueDescriptor,
    MTLIOCommandQueueType, MTLIOFileHandle, MTLIOPriority, MTLIOStatus, MTLOrigin, MTLResource,
    MTLSize, MTLTexture,
};

use super::{Buffer, Device, Image};
use crate::{AsBufferSlice, AsImageSlice, BufferUsage, ImageUsage, IoError, TexelShape};

fn supported(device: &ProtocolObject<dyn MTLDevice>) -> bool {
    device.respondsToSelector(sel!(newIOCommandQueueWithDescriptor:error:))
        && (device.respondsToSelector(sel!(newIOFileHandleWithURL:error:))
            || device.respondsToSelector(sel!(newIOHandleWithURL:error:)))
}

fn new_queue(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTLIOCommandQueue>>, IoError> {
    let desc = MTLIOCommandQueueDescriptor::new();
    desc.setPriority(MTLIOPriority::Normal);
    desc.setType(MTLIOCommandQueueType::Concurrent);
    unsafe { desc.setMaxCommandBufferCount(64) };
    device
        .newIOCommandQueueWithDescriptor_error(&desc)
        .map_err(|error| IoError::Failure(error.localizedDescription().to_string()))
}

pub(super) fn supports_io(device: &ProtocolObject<dyn MTLDevice>) -> bool {
    supported(device) && new_queue(device).is_ok()
}

pub(super) struct QueueState {
    queue: Retained<ProtocolObject<dyn MTLIOCommandQueue>>,
}

// Queue access is serialized by the owning Device's mutex.
unsafe impl Send for QueueState {}

impl QueueState {
    pub(super) fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self, crate::DeviceError> {
        let queue = new_queue(device).map_err(|_| crate::DeviceError::OutOfMemory)?;
        Ok(Self { queue })
    }
}

fn same_device(
    device: &ProtocolObject<dyn MTLDevice>,
    other: &ProtocolObject<dyn MTLDevice>,
) -> Result<(), IoError> {
    if std::ptr::eq(device, other) {
        Ok(())
    } else {
        Err(IoError::InvalidInput(
            "I/O destination belongs to another device".into(),
        ))
    }
}

fn source_range(source: Range<u64>, required: usize, exact: bool) -> Result<usize, IoError> {
    let length = source
        .end
        .checked_sub(source.start)
        .ok_or_else(|| IoError::InvalidInput("File range is reversed".into()))?;
    if length < required as u64 || (exact && length != required as u64) {
        return Err(IoError::InvalidInput(
            "File range length differs from destination byte span".into(),
        ));
    }
    let offset = usize::try_from(source.start)
        .map_err(|_| IoError::InvalidInput("File offset exceeds native address size".into()))?;
    offset
        .checked_add(required)
        .ok_or_else(|| IoError::InvalidInput("File range overflows native address size".into()))?;
    Ok(offset)
}

#[allow(deprecated)]
fn open_file(
    device: &Device,
    path: &Path,
) -> Result<Retained<ProtocolObject<dyn MTLIOFileHandle>>, IoError> {
    let path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| IoError::InvalidInput("File path contains NUL".into()))?;
    // NSURL copies the filesystem representation during construction.
    let url = unsafe {
        NSURL::fileURLWithFileSystemRepresentation_isDirectory_relativeToURL(
            NonNull::new(path.as_ptr().cast_mut()).expect("CString pointer is non-null"),
            false,
            None,
        )
    };
    let result = if device
        .metal()
        .respondsToSelector(sel!(newIOFileHandleWithURL:error:))
    {
        device.metal().newIOFileHandleWithURL_error(&url)
    } else {
        device.metal().newIOHandleWithURL_error(&url)
    };
    result.map_err(|error| IoError::Failure(error.localizedDescription().to_string()))
}

enum Operation {
    Buffer {
        buffer: Buffer,
        offset: usize,
        size: usize,
        file_offset: usize,
        file: Retained<ProtocolObject<dyn MTLIOFileHandle>>,
    },
    Image {
        image: Image,
        layer: usize,
        level: usize,
        size: MTLSize,
        origin: MTLOrigin,
        row: usize,
        plane: usize,
        file_offset: usize,
        file: Retained<ProtocolObject<dyn MTLIOFileHandle>>,
    },
}

impl Operation {
    fn encode(&self, command: &ProtocolObject<dyn MTLIOCommandBuffer>) {
        match self {
            Self::Buffer {
                buffer,
                offset,
                size,
                file_offset,
                file,
            } => unsafe {
                command.loadBuffer_offset_size_sourceHandle_sourceHandleOffset(
                    buffer.metal(),
                    *offset,
                    *size,
                    file,
                    *file_offset,
                );
            },
            Self::Image {
                image,
                layer,
                level,
                size,
                origin,
                row,
                plane,
                file_offset,
                file,
            } => unsafe {
                command.loadTexture_slice_level_size_sourceBytesPerRow_sourceBytesPerImage_destinationOrigin_sourceHandle_sourceHandleOffset(
                    image.metal(), *layer, *level, *size, *row, *plane, *origin, file, *file_offset,
                );
            },
        }
    }
}

/// Independent collection of file loads, executed when submitted.
pub struct IoBatch {
    device: Device,
    queue: Arc<Mutex<QueueState>>,
    operations: Vec<Result<Operation, IoError>>,
}

impl IoBatch {
    pub(super) fn new(device: Device, queue: Arc<Mutex<QueueState>>) -> Self {
        Self {
            device,
            queue,
            operations: Vec::new(),
        }
    }
}

// The batch owns all referenced native resources and is mutated exclusively.
unsafe impl Send for IoBatch {}

#[mev_proc::hidden_trait]
impl crate::traits::IoBatch for IoBatch {
    /// Records a file range whose length equals the destination buffer region's size.
    /// The destination must belong to this device and allow `TRANSFER_DST`.
    /// Validation, opening, and execution errors occupy this operation's result slot.
    fn load_buffer(&mut self, path: impl AsRef<Path>, source: Range<u64>, dst: impl AsBufferSlice) {
        let result = (|| {
            let dst = dst.as_buffer_slice();
            same_device(self.device.metal(), &dst.buffer().metal().device())?;
            if !dst.buffer().usage().contains(BufferUsage::TRANSFER_DST) {
                return Err(IoError::InvalidInput(
                    "I/O destination buffer requires TRANSFER_DST".into(),
                ));
            }
            if dst
                .offset()
                .checked_add(dst.size())
                .is_none_or(|end| end > dst.buffer().size())
            {
                return Err(IoError::InvalidInput(
                    "I/O destination buffer range exceeds buffer".into(),
                ));
            }
            let file_offset = source_range(source, dst.size(), true)?;
            let file = open_file(&self.device, path.as_ref())?;
            Ok(Operation::Buffer {
                buffer: dst.buffer().clone(),
                offset: dst.offset(),
                size: dst.size(),
                file_offset,
                file,
            })
        })();
        self.operations.push(result);
    }

    /// Records a file load into one destination layer and mip region.
    /// The source range must contain the byte span described by `shape`, including pitches.
    /// Format and extent must match; the destination requires `TRANSFER_DST` and one sample.
    /// Validation, opening, and execution errors occupy this operation's result slot.
    fn load_image<const N: usize>(
        &mut self,
        path: impl AsRef<Path>,
        source: Range<u64>,
        shape: TexelShape<N>,
        dst: &impl AsImageSlice<N>,
    ) {
        let result = (|| {
            let dst = dst.as_image_slice();
            same_device(self.device.metal(), &dst.image().metal().device())?;
            if !dst.image().usage().contains(ImageUsage::TRANSFER_DST) {
                return Err(IoError::InvalidInput(
                    "I/O destination image requires TRANSFER_DST".into(),
                ));
            }
            if dst.image().metal().sampleCount() != 1 {
                return Err(IoError::InvalidInput(
                    "I/O destination image must be single-sampled".into(),
                ));
            }
            let (row, plane, span) = layout::validate_io_image(
                shape,
                dst.format(),
                dst.image().extent(),
                dst.image().layers(),
                dst.image().levels(),
                dst.offset(),
                dst.extent(),
                dst.layers(),
                dst.level(),
            )?;
            let file_offset = source_range(source, span, false)?;
            let file = open_file(&self.device, path.as_ref())?;
            let mut size = [1; 3];
            let mut origin = [0; 3];
            for i in 0..N {
                size[i] = dst.extent().0[i] as usize;
                origin[i] = dst.offset().0[i] as usize;
            }
            Ok(Operation::Image {
                image: dst.image().clone(),
                layer: dst.layers().start as usize,
                level: dst.level() as usize,
                size: MTLSize {
                    width: size[0],
                    height: size[1],
                    depth: size[2],
                },
                origin: MTLOrigin {
                    x: origin[0],
                    y: origin[1],
                    z: origin[2],
                },
                row,
                plane,
                file_offset,
                file,
            })
        })();
        self.operations.push(result);
    }

    /// Submits all recorded loads and returns their aggregate completion handle.
    /// Loads may run concurrently; avoid overlapping destination writes and wait before use.
    /// Submission may wait for native queue capacity. Dropping the handle does not cancel loads.
    fn submit(self) -> IoStatus {
        let queue = self.queue.lock().expect("I/O queue mutex poisoned");
        let mut entries = Vec::with_capacity(self.operations.len());
        for operation in self.operations {
            entries.push(match operation {
                Err(error) => Entry::Error(error),
                Ok(operation) => {
                    // Commit before allocating another buffer: the native queue has a finite pool.
                    // Separate command buffers preserve an execution result for each load.
                    let command = queue.queue.commandBuffer();
                    operation.encode(&command);
                    command.commit();
                    Entry::Native {
                        command,
                        _operation: operation,
                    }
                }
            });
        }
        drop(queue);
        IoStatus {
            _queue: self.queue,
            entries,
            outcomes: OnceLock::new(),
        }
    }
}

enum Entry {
    Error(IoError),
    Native {
        command: Retained<ProtocolObject<dyn MTLIOCommandBuffer>>,
        _operation: Operation,
    },
}

/// Aggregate completion and ordered outcomes of one submitted I/O batch.
pub struct IoStatus {
    _queue: Arc<Mutex<QueueState>>,
    entries: Vec<Entry>,
    outcomes: OnceLock<Vec<Result<(), IoError>>>,
}

// Polling and waiting only inspect submitted native command buffers.
unsafe impl Send for IoStatus {}
unsafe impl Sync for IoStatus {}

impl IoStatus {
    fn completed_outcomes(&self) -> &[Result<(), IoError>] {
        self.outcomes.get_or_init(|| {
            self.entries
                .iter()
                .map(|entry| match entry {
                    Entry::Error(error) => Err(error.clone()),
                    Entry::Native { command, .. } => match command.status() {
                        MTLIOStatus::Complete => Ok(()),
                        MTLIOStatus::Cancelled => Err(IoError::Cancelled),
                        _ => Err(IoError::Failure(
                            command
                                .error()
                                .map(|error| error.localizedDescription().to_string())
                                .unwrap_or_else(|| "Metal I/O command failed".into()),
                        )),
                    },
                })
                .collect()
        })
    }
}

#[mev_proc::hidden_trait]
impl crate::traits::IoStatus for IoStatus {
    /// Returns cached outcomes in recording order once every operation finishes, otherwise `None`.
    fn poll(&self) -> Option<&[Result<(), IoError>]> {
        if let Some(outcomes) = self.outcomes.get() {
            return Some(outcomes);
        }
        if self.entries.iter().any(|entry| matches!(entry, Entry::Native { command, .. } if command.status() == MTLIOStatus::Pending)) {
            None
        } else {
            Some(self.completed_outcomes())
        }
    }

    /// Blocks until every operation finishes and returns cached outcomes in recording order.
    fn wait(&self) -> &[Result<(), IoError>] {
        if let Some(outcomes) = self.outcomes.get() {
            return outcomes;
        }
        for entry in &self.entries {
            if let Entry::Native { command, .. } = entry {
                command.waitUntilCompleted();
            }
        }
        self.completed_outcomes()
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::OpenOptions, io::Write, path::PathBuf, time::SystemTime};

    use super::*;
    use crate::{BufferDesc, DeviceDesc, Features};

    struct Fixture(PathBuf);

    impl Fixture {
        fn new(bytes: &[u8]) -> Self {
            let unique = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir()
                .join(format!("mev-metal-io-{}-{unique}.bin", std::process::id()));
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .unwrap();
            file.write_all(bytes).unwrap();
            file.sync_all().unwrap();
            Self(path)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn device() -> Device {
        let instance = super::super::Instance::load().expect("Metal instance");
        let features = Features::IO_COMMANDS | Features::BUFFER_MAP;
        let idx = instance
            .capabilities()
            .devices
            .iter()
            .position(|device| device.features.contains(features))
            .expect("Metal I/O capable device");
        instance
            .new_device(DeviceDesc {
                idx,
                queues: &[],
                features,
            })
            .unwrap()
            .0
    }

    fn buffer(device: &Device, size: usize) -> Buffer {
        device.new_buffer(BufferDesc {
            size,
            usage: BufferUsage::TRANSFER_DST | BufferUsage::HOST_READ | BufferUsage::HOST_WRITE,
            name: "I/O test destination",
        })
    }

    #[test]
    fn source_range_validation() {
        assert_eq!(source_range(3..11, 8, true), Ok(3));
        assert!(source_range(11..3, 8, true).is_err());
        assert!(source_range(3..12, 8, true).is_err());
        assert!(source_range(u64::MAX - 1..u64::MAX, 8, true).is_err());
        assert_eq!(source_range(3..12, 8, false), Ok(3));
        assert!(source_range(3..10, 8, false).is_err());
    }

    #[test]
    #[ignore = "requires a native Metal device supporting Metal I/O"]
    fn ordered_outcomes_include_open_validation_and_execution_errors() {
        let device = device();
        let file = Fixture::new(&[0, 0, 0, 5, 12, 63, 127, 128, 200, 254, 255]);
        let truncated = Fixture::new(&[42]);
        let mut destination = buffer(&device, 8);
        let failed_destination = buffer(&device, 1);
        let mut batch = device.new_io_batch();
        batch.load_buffer(&file.0, 3..11, &destination);
        batch.load_buffer(file.0.with_extension("missing"), 0..1, &failed_destination);
        batch.load_buffer(&file.0, 2..1, &failed_destination);
        batch.load_buffer(&truncated.0, 0..1, &failed_destination);
        // The range is valid when recorded; the open file changes before execution.
        OpenOptions::new()
            .write(true)
            .open(&truncated.0)
            .unwrap()
            .set_len(0)
            .unwrap();
        let status = batch.submit();
        let outcomes = status.wait();
        assert_eq!(outcomes.len(), 4);
        assert_eq!(outcomes[0], Ok(()));
        assert!(matches!(outcomes[1], Err(IoError::Failure(_))));
        assert!(matches!(outcomes[2], Err(IoError::InvalidInput(_))));
        assert!(matches!(outcomes[3], Err(IoError::Failure(_))));
        assert_eq!(status.poll(), Some(outcomes));
        let mut actual = [0; 8];
        destination.read(0, &mut actual).unwrap();
        assert_eq!(actual, [5, 12, 63, 127, 128, 200, 254, 255]);
    }

    #[test]
    #[ignore = "requires a native Metal device supporting Metal I/O"]
    fn empty_large_and_independent_thread_batches() {
        let device = device();
        let file = Fixture::new(&[42]);
        assert_eq!(device.new_io_batch().submit().poll(), Some([].as_slice()));
        let destinations: Vec<_> = (0..130).map(|_| buffer(&device, 1)).collect();
        let mut batch = device.new_io_batch();
        for dst in &destinations {
            batch.load_buffer(&file.0, 0..1, dst);
        }
        let status = batch.submit();
        assert_eq!(status.wait(), vec![Ok(()); 130]);
        for mut dst in destinations {
            let mut actual = [0];
            dst.read(0, &mut actual).unwrap();
            assert_eq!(actual, [42]);
        }

        let mut threads = Vec::new();
        for should_fail in [false, true] {
            let device = device.clone();
            let path = file.0.clone();
            threads.push(std::thread::spawn(move || {
                let dst = buffer(&device, 1);
                let mut batch = device.new_io_batch();
                batch.load_buffer(path, if should_fail { 0..2 } else { 0..1 }, &dst);
                batch.submit()
            }));
        }
        let good = threads.remove(0).join().unwrap();
        let bad = threads.remove(0).join().unwrap();
        assert_eq!(good.wait(), [Ok(())]);
        assert!(matches!(bad.wait(), [Err(IoError::InvalidInput(_))]));
    }

    #[test]
    #[ignore = "requires a native Metal device supporting Metal I/O"]
    fn dropping_batch_does_not_execute_and_dropping_status_does_not_cancel() {
        let device = device();
        let file = Fixture::new(&[42]);
        let mut dst = buffer(&device, 1);
        dst.write(0, &[77]).unwrap();
        let mut batch = device.new_io_batch();
        batch.load_buffer(&file.0, 0..1, &dst);
        drop(batch);
        let mut actual = [0];
        dst.read(0, &mut actual).unwrap();
        assert_eq!(actual, [77]);

        let mut batch = device.new_io_batch();
        batch.load_buffer(&file.0, 0..1, &dst);
        let status = batch.submit();
        // A native reference lets this regression wait after dropping the public handle.
        let command = match &status.entries[0] {
            Entry::Native { command, .. } => command.clone(),
            _ => panic!("valid operation"),
        };
        drop(status);
        command.waitUntilCompleted();
        dst.read(0, &mut actual).unwrap();
        assert_eq!(actual, [42]);

        let mut batch = device.new_io_batch();
        batch.load_buffer(&file.0, 0..1, &buffer(&device, 1));
        assert_eq!(batch.submit().wait(), [Ok(())]);
    }

    #[test]
    #[ignore = "requires a native Metal device"]
    #[should_panic(expected = "IO_COMMANDS feature is not enabled")]
    fn disabled_feature_panics_even_when_supported() {
        let instance = super::super::Instance::load().unwrap();
        let device = instance
            .new_device(DeviceDesc {
                idx: 0,
                queues: &[],
                features: Features::empty(),
            })
            .unwrap()
            .0;
        let _ = device.new_io_batch();
    }
}
