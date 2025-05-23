use std::{convert::Infallible, fmt};

use crate::generic::{
    Capabilities, CreateError, DeviceCapabilities, DeviceDesc, FamilyCapabilities, Features,
    LoadError, QueueFlags,
};

use super::{Device, Queue};

pub(crate) type LoadErrorKind = Infallible;

#[derive(Debug)]
pub(crate) enum CreateErrorKind {
    FailedToCreateDevice,
}

impl fmt::Display for CreateErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CreateErrorKind::FailedToCreateDevice => {
                write!(f, "failed to create device")
            }
        }
    }
}

pub struct Instance {
    capabilities: Capabilities,
}

impl Instance {
    pub fn load() -> Result<Self, LoadError>
    where
        Self: Sized,
    {
        Ok(Instance {
            capabilities: Capabilities {
                devices: vec![DeviceCapabilities {
                    features: Features::empty(),
                    families: vec![FamilyCapabilities {
                        queue_flags: QueueFlags::GRAPHICS
                            | QueueFlags::COMPUTE
                            | QueueFlags::TRANSFER,
                        queue_count: 32,
                    }],
                }],
            },
        })
    }
}

impl fmt::Debug for Instance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Instance").finish()
    }
}

impl crate::traits::Resource for Instance {}

#[hidden_trait::expose]
impl crate::traits::Instance for Instance {
    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    fn new_device(&self, info: DeviceDesc) -> Result<(Device, Vec<Queue>), CreateError> {
        assert!(
            info.queues.iter().all(|&f| f == 0),
            "Only one queue family is supported"
        );

        let device = metal::Device::system_default()
            .ok_or(CreateError(CreateErrorKind::FailedToCreateDevice))?;

        let device = Device::new(device, info.queues.len());

        let queues = (0..info.queues.len())
            .map(|_| Queue::new(device.clone(), device.metal().new_command_queue()))
            .collect();

        Ok((device, queues))
    }
}
