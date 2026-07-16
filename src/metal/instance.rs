use std::fmt;

use objc2::{Message, rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCopyAllDevices, MTLDevice};

use crate::generic::{
    Capabilities, DeviceCapabilities, DeviceDesc, DeviceError, DeviceType, FamilyCapabilities,
    Features, QueueFlags,
};

use super::{Device, Queue};

pub struct Instance {
    devices: Vec<Retained<ProtocolObject<dyn MTLDevice>>>,
    capabilities: Capabilities,
}

impl Instance {
    pub fn load() -> Result<Self, DeviceError>
    where
        Self: Sized,
    {
        let devices = MTLCopyAllDevices();

        let capabilities = Capabilities {
            devices: devices
                .iter()
                .map(|device| {
                    let mut features = Features::empty();

                    if !device.isHeadless() {
                        features |= Features::SURFACE;
                    }

                    if device.supportsBCTextureCompression() {
                        features |= Features::TEXTURE_COMPRESSION_BC;
                    }

                    let device_type = if device.isLowPower() {
                        DeviceType::IntegratedGpu
                    } else {
                        DeviceType::DiscreteGpu
                    };

                    let name = device.name().to_string();

                    DeviceCapabilities {
                        name,
                        device_type,
                        features,
                        families: vec![FamilyCapabilities {
                            queue_flags: QueueFlags::GRAPHICS
                                | QueueFlags::COMPUTE
                                | QueueFlags::TRANSFER,
                            queue_count: 32,
                        }],
                    }
                })
                .collect(),
        };

        let devices = devices.iter().map(|device| device.retain()).collect();

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
            .map(|_| {
                let queue = device
                    .metal()
                    .newCommandQueue()
                    .expect("Failed to create command queue");
                Queue::new(device.clone(), queue)
            })
            .collect();

        Ok((device, queues))
    }
}
