use std::{fmt, rc::Rc};

use mev_proc::hidden_trait;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use wgpu::MemoryHints::Performance;

use crate::generic::{
    Capabilities, DeviceCapabilities, DeviceDesc, DeviceError, DeviceSurfaceCapabilities,
    DeviceType, FamilyCapabilities, Features, QueueFlags, SurfaceCapabilities, SurfaceError,
};

use super::{Device, Queue, Surface};

pub struct Instance {
    instance: Rc<wgpu::Instance>,
    adapters: Vec<wgpu::Adapter>,
    capabilities: Capabilities,
}

impl Instance {
    pub(super) fn wgpu_instance(&self) -> Rc<wgpu::Instance> {
        self.instance.clone()
    }

    pub(super) fn wgpu_adapters(&self) -> &[wgpu::Adapter] {
        &self.adapters
    }
}

impl fmt::Debug for Instance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Instance").finish()
    }
}

impl crate::traits::Resource for Instance {}

#[hidden_trait]
impl crate::traits::Instance for Instance {
    async fn load_async() -> Result<Self, DeviceError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });

        let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;

        let capabilities = Capabilities {
            devices: adapters
                .iter()
                .map(|adapter| {
                    let mut features = Features::SURFACE
                        | Features::BUFFER_MAP_AT_CREATION
                        | Features::BUFFER_WRITE_QUEUE;

                    if adapter
                        .features()
                        .contains(wgpu::Features::TEXTURE_COMPRESSION_BC)
                    {
                        features |= Features::TEXTURE_COMPRESSION_BC;
                    }

                    let device_type = match adapter.get_info().device_type {
                        wgpu::DeviceType::IntegratedGpu => DeviceType::IntegratedGpu,
                        wgpu::DeviceType::DiscreteGpu => DeviceType::DiscreteGpu,
                        wgpu::DeviceType::VirtualGpu => DeviceType::VirtualGpu,
                        wgpu::DeviceType::Cpu => DeviceType::Cpu,
                        _ => DeviceType::Unknown,
                    };

                    DeviceCapabilities {
                        name: adapter.get_info().name.clone(),
                        device_type,
                        features,
                        families: vec![FamilyCapabilities {
                            queue_flags: QueueFlags::GRAPHICS
                                | QueueFlags::COMPUTE
                                | QueueFlags::TRANSFER,
                            queue_count: 1,
                        }],
                    }
                })
                .collect(),
        };

        Ok(Instance {
            instance: Rc::new(instance),
            adapters,
            capabilities,
        })
    }

    fn load() -> Result<Self, DeviceError> {
        #[cfg(target_arch = "wasm32")]
        {
            panic!("Synchronous loading is not supported on wasm32. Use `load_async` instead.");
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            pollster::block_on(Self::load_async())
        }
    }

    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    /// Returns capabilities for specific Surface
    fn surface_capabilities(
        &self,
        _window: &impl HasWindowHandle,
        _display: &impl HasDisplayHandle,
    ) -> Result<SurfaceCapabilities, SurfaceError> {
        Ok(SurfaceCapabilities {
            devices: self
                .capabilities
                .devices
                .iter()
                .map(|device| DeviceSurfaceCapabilities {
                    families: vec![
                        device.features.contains(Features::SURFACE);
                        device.families.len()
                    ],
                })
                .collect(),
        })
    }

    async fn new_device_async(
        &self,
        info: DeviceDesc<'_>,
    ) -> Result<(Device, Vec<Queue>), DeviceError> {
        assert!(
            info.queues.len() <= 1,
            "Only one queue is supported for now"
        );

        assert!(
            info.idx < self.capabilities.devices.len(),
            "Device index out of bounds"
        );

        assert!(
            self.capabilities.devices[info.idx]
                .features
                .contains(info.features),
            "Requested device features are unsupported"
        );

        let adapter = &self.adapters[info.idx];

        let available_features = adapter.features();
        let available_limits = adapter.limits();

        let mut required_features = wgpu::Features::empty();

        if available_features.contains(wgpu::Features::IMMEDIATES) {
            required_features |= wgpu::Features::IMMEDIATES;
        }

        if available_features.contains(wgpu::Features::SUBGROUP) {
            required_features |= wgpu::Features::SUBGROUP;
        }

        if available_features.contains(wgpu::Features::SHADER_F16) {
            required_features |= wgpu::Features::SHADER_F16;
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

        let (wgpu_device, wgpu_queue) = adapter
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

        let device = Device::new(wgpu_device, self.instance.clone(), adapter.clone());

        let queue_count = info.queues.len().max(1);
        let queues = (0..queue_count)
            .map(|_| Queue::new(device.clone(), wgpu_queue.clone()))
            .collect();

        Ok((device, queues))
    }

    fn new_device(&self, info: DeviceDesc) -> Result<(Device, Vec<Queue>), DeviceError> {
        #[cfg(target_arch = "wasm32")]
        {
            panic!(
                "Synchronous device creation is not supported on wasm32. Use `new_device_async` instead."
            );
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            pollster::block_on(self.new_device_async(info))
        }
    }

    fn new_device_with_surface(
        &self,
        info: DeviceDesc<'_>,
        window: &impl HasWindowHandle,
        display: &impl HasDisplayHandle,
    ) -> Result<(Device, Vec<Queue>, Surface), SurfaceError> {
        let (device, queues) = self.new_device(info)?;
        let surface = device.new_surface(window, display)?;
        Ok((device, queues, surface))
    }

    async fn new_device_with_surface_async(
        &self,
        info: DeviceDesc<'_>,
        window: &impl HasWindowHandle,
        display: &impl HasDisplayHandle,
    ) -> Result<(Device, Vec<Queue>, Surface), SurfaceError> {
        let (device, queues) = self.new_device(info)?;
        let surface = device.new_surface(window, display)?;
        Ok((device, queues, surface))
    }
}
