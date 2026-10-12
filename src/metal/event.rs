use objc2::{rc::Retained, runtime::ProtocolObject, sel};
use objc2_foundation::{NSObjectProtocol, NSString};
use objc2_metal::{MTLDevice, MTLEvent, MTLSharedEvent};

use super::Device;
use crate::SyncError;

pub(super) fn supports_events(device: &ProtocolObject<dyn MTLDevice>) -> bool {
    device.respondsToSelector(sel!(newSharedEvent)) && device.newSharedEvent().is_some()
}

/// Shared timeline for CPU, GPU queues, and IO queues on one device.
#[derive(Clone, Debug)]
pub struct Event {
    device: Device,
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        Retained::as_ptr(&self.event) == Retained::as_ptr(&other.event)
    }
}
impl Eq for Event {}

impl Event {
    pub(super) fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        self.device.metal()
    }
    pub(super) fn metal(&self) -> &ProtocolObject<dyn MTLSharedEvent> {
        &self.event
    }
}

#[mev_proc::hidden_trait]
impl crate::traits::Event for Event {
    /// Returns the current signaled value.
    fn value(&self) -> Result<u64, SyncError> {
        Ok(self.event.signaledValue())
    }
    /// Signals from the CPU. Values must execute in strictly increasing order.
    ///
    /// Coordinate all producers before advancing the value. After failed IO,
    /// releasing a GPU wait does not make its incomplete destination data valid.
    fn set_value(&self, value: u64) -> Result<(), SyncError> {
        if value <= self.event.signaledValue() {
            return Err(SyncError::InvalidInput(
                "event signal must exceed its current value".into(),
            ));
        }
        self.event.setSignaledValue(value);
        Ok(())
    }
}

#[mev_proc::hidden_trait]
impl crate::traits::DeviceEvent for Device {
    /// Creates a shared timeline event initialized to `initial_value`.
    fn new_event(&self, initial_value: u64, name: &str) -> Result<Event, SyncError> {
        if !self.metal().respondsToSelector(sel!(newSharedEvent)) {
            return Err(SyncError::Unsupported);
        }
        let event = self
            .metal()
            .newSharedEvent()
            .ok_or_else(|| SyncError::Failure("Failed to create shared event".into()))?;
        event.setLabel(Some(&NSString::from_str(name)));
        event.setSignaledValue(initial_value);
        Ok(Event {
            device: self.clone(),
            event,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{BufferDesc, BufferUsage, DeviceDesc, EventPoint, Features, SubmissionSync};

    #[test]
    #[ignore = "requires a native Metal device supporting timeline events"]
    fn gpu_queue_timeline_handoff() {
        let instance = super::super::Instance::load().unwrap();
        let features = Features::TIMELINE_EVENTS | Features::BUFFER_MAP;
        let idx = instance
            .capabilities()
            .devices
            .iter()
            .position(|device| device.features.contains(features))
            .unwrap();
        let (device, queues) = instance
            .new_device(DeviceDesc {
                idx,
                queues: &[0, 0],
                features,
            })
            .unwrap();
        let mut queues = queues.into_iter();
        let mut producer = queues.next().unwrap();
        let mut consumer = queues.next().unwrap();
        let event = device.new_event(0, "GPU handoff").unwrap();
        let source = device.new_buffer(BufferDesc {
            size: 8,
            usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
            name: "producer output",
        });
        let mut output = device.new_buffer(BufferDesc {
            size: 8,
            usage: BufferUsage::TRANSFER_DST | BufferUsage::HOST_READ,
            name: "consumer output",
        });
        let mut fill = producer.new_command_encoder();
        fill.copy().fill_buffer(&source, 7);
        producer
            .submit_with_sync(
                [fill.finish()],
                SubmissionSync {
                    waits: &[],
                    signals: &[EventPoint {
                        event: &event,
                        value: 1,
                    }],
                },
            )
            .unwrap();
        let mut copy = consumer.new_command_encoder();
        copy.copy().copy_buffer_to_buffer(&source, &output);
        consumer
            .submit_checkpoint_with_sync(
                [copy.finish()],
                SubmissionSync {
                    waits: &[EventPoint {
                        event: &event,
                        value: 1,
                    }],
                    signals: &[EventPoint {
                        event: &event,
                        value: 2,
                    }],
                },
            )
            .unwrap();
        consumer.wait_idle().unwrap();
        let mut actual = [0; 8];
        output.read(0, &mut actual).unwrap();
        assert_eq!(actual, [7; 8]);
        assert_eq!(event.value().unwrap(), 2);
        producer
            .submit_with_sync(
                [],
                SubmissionSync {
                    waits: &[EventPoint {
                        event: &event,
                        value: 2,
                    }],
                    signals: &[EventPoint {
                        event: &event,
                        value: 3,
                    }],
                },
            )
            .unwrap();
        producer.wait_idle().unwrap();
        assert_eq!(event.value().unwrap(), 3);
        assert!(event.set_value(3).is_err());
        let duplicate = [EventPoint {
            event: &event,
            value: 4,
        }; 2];
        assert!(matches!(
            producer.submit_with_sync(
                [],
                SubmissionSync {
                    waits: &[],
                    signals: &duplicate
                }
            ),
            Err(crate::SyncError::InvalidInput(_))
        ));
        let wrong_queue_command = consumer.new_command_encoder().finish();
        assert!(matches!(
            producer.submit_with_sync(
                [wrong_queue_command],
                SubmissionSync {
                    waits: &[],
                    signals: &[EventPoint {
                        event: &event,
                        value: 4
                    }],
                }
            ),
            Err(crate::SyncError::InvalidInput(_))
        ));
    }
}
