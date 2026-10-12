use std::{fmt, sync::Arc};

use ash::vk;

use crate::SyncError;

use super::{Device, map_device_error};

struct Inner {
    device: Device,
    handle: vk::Semaphore,
}

impl Drop for Inner {
    fn drop(&mut self) {
        unsafe { self.device.ash().destroy_semaphore(self.handle, None) };
    }
}

/// Shared timeline event. Signals must execute in strictly increasing value order.
#[derive(Clone)]
pub struct Event(Arc<Inner>);

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Event {}

impl fmt::Debug for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Event").field(&self.0.handle).finish()
    }
}



#[mev_proc::hidden_trait]
impl crate::traits::DeviceEvent for Device {
    /// Creates a shared timeline event; requires enabled `TIMELINE_EVENTS`.
    fn new_event(&self, initial_value: u64, name: &str) -> Result<Event, SyncError> {
        if !self.timeline_events_enabled() {
            return Err(SyncError::Unsupported);
        }
        if name.contains('\0') {
            return Err(SyncError::InvalidInput("event name contains NUL".into()));
        }
        self.get_error()?;
        let mut timeline = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(initial_value);
        let handle = unsafe {
            self.ash().create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut timeline),
                None,
            )
        }
        .map_err(map_device_error)?;
        self.name_event(handle, name);
        Ok(Event(Arc::new(Inner {
            device: self.clone(),
            handle,
        })))
    }
}

impl Event {
    pub(super) fn device(&self) -> &Device {
        &self.0.device
    }
    pub(super) fn handle(&self) -> vk::Semaphore {
        self.0.handle
    }
}

#[mev_proc::hidden_trait]
impl crate::traits::Event for Event {

    /// Reads the completed timeline value.
    fn value(&self) -> Result<u64, SyncError> {
        unsafe {
            self.0
                .device
                .ash()
                .get_semaphore_counter_value(self.0.handle)
        }
        .map_err(|err| map_device_error(err).into())
    }

    /// Signals from the CPU. Order this against all pending GPU signals.
    fn set_value(&self, value: u64) -> Result<(), SyncError> {
        let current = self.value()?;
        if value <= current {
            return Err(SyncError::InvalidInput(
                "signal value must exceed completed value".into(),
            ));
        }
        if value - current > self.0.device.max_timeline_value_difference() {
            return Err(SyncError::InvalidInput(
                "event value exceeds the device timeline difference limit".into(),
            ));
        }
        unsafe {
            self.0.device.ash().signal_semaphore(
                &vk::SemaphoreSignalInfo::default()
                    .semaphore(self.0.handle)
                    .value(value),
            )
        }
        .map_err(|err| map_device_error(err).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BufferDesc, BufferInitDesc, BufferUsage, DeviceDesc, EventPoint, Features, Instance,
        QueueFlags, SubmissionSync,
    };

    #[test]
    #[ignore = "requires a Vulkan 1.2 timeline-semaphore device"]
    fn timeline_cross_queue_submissions() {
        let instance = Instance::load().expect("Vulkan instance");
        let (idx, first, second) = instance
            .capabilities()
            .devices
            .iter()
            .enumerate()
            .find_map(|(idx, caps)| {
                if !caps.features.contains(Features::TIMELINE_EVENTS) {
                    return None;
                }
                if let Some(family) = caps.families.iter().position(|family| {
                    family.queue_count >= 2 && family.queue_flags.contains(QueueFlags::TRANSFER)
                }) {
                    return Some((idx, family as u32, family as u32));
                }
                let mut families = caps
                    .families
                    .iter()
                    .enumerate()
                    .filter(|(_, family)| family.queue_count != 0);
                Some((idx, families.next()?.0 as u32, families.next()?.0 as u32))
            })
            .expect("device with two queues and timeline support");
        let requested = [first, second];
        let (device, mut queues) = instance
            .new_device(DeviceDesc {
                idx,
                queues: &requested,
                features: Features::TIMELINE_EVENTS
                    | Features::BUFFER_MAP
                    | Features::BUFFER_MAP_AT_CREATION,
            })
            .expect("timeline device");
        let mut consumer = queues.pop().expect("consumer queue");
        let mut producer = queues.pop().expect("producer queue");
        assert_ne!(producer.handle_for_test(), consumer.handle_for_test());
        let event = device.new_event(0, "cross queue").expect("event");
        assert_eq!(event.value().expect("counter"), 0);
        assert!(matches!(
            event.set_value(0),
            Err(SyncError::InvalidInput(_))
        ));
        let points = [EventPoint {
            event: &event,
            value: 1,
        }];
        let duplicate = [points[0], points[0]];
        assert!(matches!(
            producer.submit_with_sync(
                [],
                SubmissionSync {
                    signals: &duplicate,
                    ..Default::default()
                }
            ),
            Err(SyncError::InvalidInput(_))
        ));
        assert!(matches!(
            producer.submit_with_sync(
                [],
                SubmissionSync {
                    waits: &points,
                    signals: &points
                }
            ),
            Err(SyncError::InvalidInput(_))
        ));
        let mut readback = None;
        if first == second {
            let source = device.new_buffer_init(BufferInitDesc {
                data: &[1u8, 2, 3, 4][..],
                usage: BufferUsage::TRANSFER_SRC,
                name: "source",
            });
            let intermediate = device.new_buffer(BufferDesc {
                size: 4,
                usage: BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
                name: "intermediate",
            });
            let destination = device.new_buffer(BufferDesc {
                size: 4,
                usage: BufferUsage::TRANSFER_DST | BufferUsage::HOST_READ,
                name: "readback",
            });
            let mut write = producer.new_command_encoder();
            write.copy().copy_buffer_to_buffer(&source, &intermediate);
            producer
                .submit_checkpoint_with_sync(
                    [write.finish()],
                    SubmissionSync {
                        signals: &points,
                        ..Default::default()
                    },
                )
                .expect("producer submit");
            let mut read = consumer.new_command_encoder();
            read.copy()
                .copy_buffer_to_buffer(&intermediate, &destination);
            consumer
                .submit_checkpoint_with_sync(
                    [read.finish()],
                    SubmissionSync {
                        waits: &points,
                        ..Default::default()
                    },
                )
                .expect("consumer submit");
            readback = Some(destination);
        } else {
            eprintln!(
                "Device lacks two transfer queues in one family; testing native cross-family event synchronization only"
            );
            producer
                .submit_checkpoint_with_sync(
                    [],
                    SubmissionSync {
                        signals: &points,
                        ..Default::default()
                    },
                )
                .expect("empty signal");
            consumer
                .submit_checkpoint_with_sync(
                    [],
                    SubmissionSync {
                        waits: &points,
                        ..Default::default()
                    },
                )
                .expect("empty wait");
        }
        consumer.wait_idle().expect("consumer idle");
        producer.wait_idle().expect("producer idle");
        assert_eq!(event.value().expect("counter"), 1);
        if let Some(mut readback) = readback {
            let mut bytes = [0; 4];
            readback.read(0, &mut bytes).expect("readback");
            assert_eq!(bytes, [1, 2, 3, 4]);
        }
        event.set_value(2).expect("CPU signal");
        let points = [EventPoint {
            event: &event,
            value: 3,
        }];
        producer
            .submit_checkpoint_with_sync(
                [],
                SubmissionSync {
                    signals: &points,
                    ..Default::default()
                },
            )
            .expect("empty signal");
        consumer
            .submit_checkpoint_with_sync(
                [],
                SubmissionSync {
                    waits: &points,
                    ..Default::default()
                },
            )
            .expect("empty wait");
        let weak = Arc::downgrade(&event.0);
        drop(event);
        assert!(
            weak.upgrade().is_some(),
            "pending submissions retain events"
        );
        producer.wait_idle().expect("producer idle");
        consumer.wait_idle().expect("consumer idle");
        assert!(weak.upgrade().is_none(), "completed epochs release events");
        let (other, _queues) = instance
            .new_device(DeviceDesc {
                idx,
                queues: &[first],
                features: Features::TIMELINE_EVENTS,
            })
            .expect("second device");
        let foreign = other.new_event(0, "foreign").expect("foreign event");
        let points = [EventPoint {
            event: &foreign,
            value: 1,
        }];
        assert!(matches!(
            producer.submit_with_sync(
                [],
                SubmissionSync {
                    signals: &points,
                    ..Default::default()
                }
            ),
            Err(SyncError::InvalidInput(_))
        ));
        let (disabled, _queues) = instance
            .new_device(DeviceDesc {
                idx,
                queues: &[first],
                features: Features::empty(),
            })
            .expect("feature disabled");
        assert!(matches!(
            disabled.new_event(0, "disabled"),
            Err(SyncError::Unsupported)
        ));
    }
}
