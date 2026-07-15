use std::{convert::Infallible, fmt};

use crate::generic::{
    Capabilities, DeviceCapabilities, DeviceDesc, DeviceError, FamilyCapabilities, Features,
    QueueFlags,
};

use super::{Device, Queue};

pub struct Instance {
    devices: Vec<metal::Device>,
    capabilities: Capabilities,
}

impl Instance {
    pub fn load() -> Result<Self, DeviceError>
    where
        Self: Sized,
    {
        let devices = metal::Device::all();

        let mut features = vec![Features::empty(); devices.len()];

        // Any device can be used to create surfaces, so we can always enable this feature.
        for feature in &mut features {
            *feature |= Features::SURFACE;
        }

        for (device, feature) in devices.iter().zip(features.iter_mut()) {
            if device.supports_BC_texture_compression() {
                *feature |= Features::TEXTURE_COMPRESSION_BC;
            }
        }

        let capabilities = Capabilities {
            devices: devices
                .iter()
                .zip(features.iter())
                .map(|(device, &features)| DeviceCapabilities {
                    features,
                    families: vec![FamilyCapabilities {
                        queue_flags: QueueFlags::GRAPHICS
                            | QueueFlags::COMPUTE
                            | QueueFlags::TRANSFER,
                        queue_count: 32,
                    }],
                })
                .collect(),
        };

        Ok(Instance {
            devices,
            capabilities,
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
    #[inline(always)]
    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    fn new_device(&self, info: DeviceDesc) -> Result<(Device, Vec<Queue>), DeviceError> {
        assert!(
            info.queues.iter().all(|&f| f == 0),
            "Only one queue family is supported"
        );

        assert!(
            info.idx < self.capabilities.devices.len(),
            "Device index out of bounds"
        );

        let device = Device::new(self.devices[info.idx].clone(), info.queues.len());

        let queues = (0..info.queues.len())
            .map(|_| Queue::new(device.clone(), device.metal().new_command_queue()))
            .collect();

        Ok((device, queues))
    }
}
