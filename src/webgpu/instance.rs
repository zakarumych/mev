use std::{fmt, rc::Rc};

use crate::generic::{
    Capabilities, DeviceCapabilities, DeviceDesc, DeviceError, FamilyCapabilities, Features,
    QueueFlags,
};

use super::{Device, Queue};

pub struct Instance {
    instance: Rc<wgpu::Instance>,
    adapter: wgpu::Adapter,
    capabilities: Capabilities,
}

impl Instance {
    pub async fn load_async() -> Result<Self, DeviceError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|err| {
                tracing::error!("Failed to request adapter: {err:?}");
                DeviceError::DeviceLost
            })?;

        let capabilities = Capabilities {
            devices: vec![DeviceCapabilities {
                features: Features::empty(),
                families: vec![FamilyCapabilities {
                    queue_flags: QueueFlags::GRAPHICS | QueueFlags::COMPUTE | QueueFlags::TRANSFER,
                    queue_count: 1,
                }],
            }],
        };

        Ok(Instance {
            instance: Rc::new(instance),
            adapter,
            capabilities,
        })
    }

    pub fn load() -> Result<Self, DeviceError> {
        #[cfg(target_arch = "wasm32")]
        {
            panic!("Synchronous loading is not supported on wasm32. Use `load_async` instead.");
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            pollster::block_on(Self::load_async())
        }
    }

    pub(super) fn wgpu_instance(&self) -> Rc<wgpu::Instance> {
        self.instance.clone()
    }

    pub(super) fn wgpu_adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }
}

impl fmt::Debug for Instance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Instance").finish()
    }
}

impl crate::traits::Resource for Instance {}

impl crate::traits::Instance for Instance {
    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    async fn new_device_async(
        &self,
        info: DeviceDesc<'_>,
    ) -> Result<(crate::backend::Device, Vec<crate::backend::Queue>), DeviceError> {
        assert!(
            info.queues.len() <= 1,
            "Only one queue is supported for now"
        );

        let available_features = self.adapter.features();
        let available_limits = self.adapter.limits();

        let mut required_features = wgpu::Features::empty();

        if available_features.contains(wgpu::Features::IMMEDIATES) {
            required_features |= wgpu::Features::IMMEDIATES;
        }

        if info.features.contains(Features::TEXTURE_COMPRESSION_BC) {
            if !available_features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
                tracing::error!(
                    "Adapter does not support required feature: TEXTURE_COMPRESSION_BC"
                );
                return Err(DeviceError::DeviceLost);
            }
            required_features |= wgpu::Features::TEXTURE_COMPRESSION_BC;
        }

        let (wgpu_device, wgpu_queue) = self
            .adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features,
                required_limits: available_limits,
                memory_hints: Default::default(),
                ..Default::default()
            })
            .await
            .map_err(|err| {
                tracing::error!("Failed to request device: {err:?}");
                DeviceError::DeviceLost
            })?;

        let device = Device::new(wgpu_device, self.instance.clone(), self.adapter.clone());

        let queue_count = info.queues.len().max(1);
        let queues = (0..queue_count)
            .map(|_| Queue::new(device.clone(), wgpu_queue.clone()))
            .collect();

        Ok((device, queues))
    }

    fn new_device(&self, info: DeviceDesc) -> Result<(Device, Vec<Queue>), DeviceError> {
        #[cfg(target_arch = "wasm32")]
        {
            panic!("Synchronous device creation is not supported on wasm32. Use `new_device_async` instead.");
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            pollster::block_on(self.new_device_async(info))
        }
    }
}

impl Instance {
    pub fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    pub async fn new_device_async(
        &self,
        info: DeviceDesc<'_>,
    ) -> Result<(crate::backend::Device, Vec<crate::backend::Queue>), DeviceError> {
        crate::traits::Instance::new_device_async(self, info).await
    }

    pub fn new_device(&self, info: DeviceDesc) -> Result<(Device, Vec<Queue>), DeviceError> {
        crate::traits::Instance::new_device(self, info)
    }
}
