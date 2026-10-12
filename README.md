# MEV

[![docs.rs](https://img.shields.io/docsrs/mev?style=for-the-badge)](https://docs.rs/mev/)
[![Crates.io Total Downloads](https://img.shields.io/crates/d/mev?style=for-the-badge)](https://crates.io/crates/mev)
[![Discord](https://img.shields.io/discord/1270330377847832646?style=for-the-badge&logo=discord)](https://discord.com/channels/1270330377847832646/1318879687761465445)

Mev is a GAPI that can use Metal et Vulkan GAPIs.

With ergonomics in mind, MEV is designed to be:

- 🪓 **Simple**: MEV is a no-nonsense GAPI keeping WTF per minute low.
- 🎯 **Flexible**: MEV is designed to be used in a variety of applications, exposing the full power of the underlying APIs.
- 🚀 **Fast**: MEV is designed to be fast, adding as little overhead as possible.
- 😵 **Safe-ish**: MEV is designed to be safe, but it's still possible to shoot yourself in the foot.

## Backends

MEV supports two backends: Metal and Vulkan.
It automatically picks one based on the platform it's built for.

On MacOS and iOS, MEV uses Metal.
On other platforms, MEV uses Vulkan 🌋


## Buffer uploads

Request upload features through `DeviceDesc::features`; inspect
`DeviceCapabilities::features` for support. Requested features guarantee
availability. Unsupported feature requests are rejected during device creation.

| Operation | Feature | Vulkan | Metal | WebGPU |
| --- | --- | --- | --- | --- |
| Map after creation | `BUFFER_MAP` | Yes | Yes | No |
| Create mapped | `BUFFER_MAP_AT_CREATION` | Yes | Yes | Yes |
| Encoder upload command | `BUFFER_WRITE_COMMAND` | Yes | Yes | No |
| Queue upload operation | `BUFFER_WRITE_QUEUE` | No | No | Yes |

These expose backend operations without emulating queue writes through commands
or command writes through queue operations. Metal's command-queue API used by
this backend has no direct buffer upload operation.

`Device::new_buffer_mapped` creates a buffer with its full logical range mapped
for writing, preserving the requested usage. Call `write_mapped_range`, write
through its view, drop the view, then `unmap` before GPU use. This initial mapping
does not require `HOST_WRITE`; subsequent mapping and host access require
`BUFFER_MAP` and the matching `HOST_READ` / `HOST_WRITE` usage.

`Device::new_buffer_init` initializes through mapped creation. Byte sources write
directly to mapped memory. WebGPU uses a temporary zeroed CPU buffer for
`InitMapped` closures because its mapped views are write-only. WebGPU mapped
views support `write`; native backends also expose `AsRef` and `AsMut` slices.

`CopyCommandEncoder::write_buffer_raw` / `write_buffer` / `write_buffer_slice`
record uploads within commands and require `TRANSFER_DST`. Synchronize surrounding
GPU accesses with command barriers or `write_buffer_sync`.

`Queue::write_buffer_raw` / `write_buffer` / `write_buffer_slice` copy source data
before returning and require `TRANSFER_DST`. The next submission, including an
empty submission, uploads data before its command buffers. Queue writes follow
already-submitted work; they cannot be placed between commands inside a
submission. Command barriers do not change that placement. WebGPU requires the
write offset and byte count to be multiples of four.

```rust,no_run
# fn example(device: &mev::Device) {
let mut buffer = device.new_buffer_mapped(mev::BufferDesc {
    size: 4,
    usage: mev::BufferUsage::VERTEX,
    name: "vertices",
});
{
    let mut mapped = buffer.write_mapped_range(..).expect("initial mapping");
    mapped.write(0, &[1, 2, 3, 4]);
}
buffer.unmap();
# }
```

## Native file I/O

`IO_COMMANDS` exposes native file loading on supported Metal OS/device
combinations. Vulkan and WebGPU do not advertise this feature. Enable it when
creating the Device; `Device::new_io_batch` panics if it was not enabled.
The Device owns the internal I/O queue. Loads do not perform CPU reads or
emulate I/O with upload commands.

Create an independent `IoBatch`, record file paths and byte ranges with
`load_buffer` / `load_image`, then call `submit` on the batch. Each batch keeps
its own operations, including when recorded through Device clones on different
threads. Dropping an unsubmitted batch discards its operations. Dropping a
submitted status handle does not cancel the submitted operations.

Destinations require `TRANSFER_DST` usage and must belong to the same device.
`load_buffer` requires the source range length to equal the destination slice's
byte count. `load_image` loads a single layer and mip region described by a
matching `TexelShape`; the source range must contain the layout's byte span.
Compressed image row and plane pitches count compression blocks, including
partial blocks at mip edges. Custom pitches must contain full rows/planes and
be aligned to blocks/rows respectively.

`IoStatus::poll` returns `None` while any operation is pending, then returns all
results in recording order. `wait` blocks and returns the same results. Every
load call has one result, including file-opening and validation failures. An
empty batch completes with an empty result slice. Recording order does not
establish execution dependencies: operations may execute concurrently. Avoid
overlapping destination writes, and wait for completion before accessing loaded
resources on the CPU or GPU. Check results before consuming their contents.

```rust,no_run
# fn load(device: &mev::Device, destination: &mev::Buffer) -> Result<(), mev::IoError> {
let mut batch = device.new_io_batch();
batch.load_buffer("assets/vertices.bin", 0..destination.size() as u64, destination);
let status = batch.submit();
for result in status.wait() {
    result.clone()?;
}
// Successful loads are ready for GPU consumption.
# Ok(())
# }
```

For streaming, retain the status handle and call `poll` between frames. Submit
processing or rendering work when the batch has completed successfully, allowing
unrelated GPU work to continue during file loading.

## Cross-queue synchronization

`TIMELINE_EVENTS` exposes shared `Event` timelines on Vulkan and supported Metal
devices. It is independent of `IO_COMMANDS`. WebGPU does not advertise it;
`Device::new_event` returns `SyncError::Unsupported`. Submit GPU work with
`Queue::submit_with_sync` or `submit_checkpoint_with_sync` and a `SubmissionSync`.
Waits gate every command buffer in the submission; signals follow all of them.
An empty submission can still wait and signal. Empty synchronization lists work
on every backend through the ordinary submission operation.

```rust,no_run
# fn handoff(device: &mev::Device, producer: &mut mev::Queue, consumer: &mut mev::Queue,
#     produced: mev::CommandBuffer, consumed: mev::CommandBuffer) -> Result<(), mev::SyncError> {
let ready = device.new_event(0, "produced data")?;
producer.submit_with_sync([produced], mev::SubmissionSync {
    waits: &[],
    signals: &[mev::EventPoint { event: &ready, value: 1 }],
})?;
consumer.submit_with_sync([consumed], mev::SubmissionSync {
    waits: &[mev::EventPoint { event: &ready, value: 1 }],
    signals: &[],
})?;
# Ok(())
# }
```

An event belongs to one device. A wait completes when its timeline reaches at
least the requested value. Each wait/signal list can contain an event only once;
a submission signaling an event it also waits on must signal a greater value.
CPU `Event::set_value` and GPU signals must execute in strictly
increasing order. Use one producer per timeline or explicitly order producers
with waits; submission call order across queues does not establish execution
order. Avoid dependency cycles. Events order execution and memory access;
resource usage/layout rules still apply. Vulkan exclusive resources can pass
between queues in the same family; handing them to another family also requires
queue-family ownership transfer, which this API does not expose.
