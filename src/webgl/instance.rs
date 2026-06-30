use raw_window_handle::{
    HasDisplayHandle, HasRawWindowHandle, HasWindowHandle, RawWindowHandle, WindowHandle,
};
use std::{convert::Infallible, fmt};
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{HtmlCanvasElement, WebGl2RenderingContext};

use crate::{
    generic::{
        Capabilities, DeviceCapabilities, DeviceDesc, DeviceError, FamilyCapabilities, Features,
        QueueFlags,
    },
    SurfaceError,
};

use super::{Device, Queue, Surface};

pub struct Instance {
    capabilities: Capabilities,
}

impl Instance {
    pub fn load() -> Result<Self, DeviceError> {
        // Check if WebGL2RenderingContext is available
        if js_sys::Reflect::get(&js_sys::global(), &"WebGL2RenderingContext".into()).is_err() {
            return Err(DeviceError::DeviceLost);
        }

        // WebGL always has one device (the GPU) with one queue family
        Ok(Instance {
            capabilities: Capabilities {
                devices: vec![DeviceCapabilities {
                    features: Features::SURFACE, // WebGL always supports surfaces
                    families: vec![FamilyCapabilities {
                        queue_flags: QueueFlags::GRAPHICS | QueueFlags::TRANSFER,
                        queue_count: 1, // WebGL has single queue
                    }],
                }],
            },
        })
    }

    fn create_context(canvas: &HtmlCanvasElement) -> Result<WebGl2RenderingContext, DeviceError> {
        let context = canvas
            .get_context("webgl2")
            .map_err(|err| {
                tracing::error!("Failed to get WebGL2 context: {:?}", err);
                DeviceError::DeviceLost
            })?
            .ok_or(DeviceError::DeviceLost)?
            .dyn_into::<WebGl2RenderingContext>()
            .unwrap();

        // Enable standard WebGL2 features
        context.enable(WebGl2RenderingContext::DEPTH_TEST);
        context.enable(WebGl2RenderingContext::CULL_FACE);

        Ok(context)
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

    fn new_device(&self, desc: DeviceDesc) -> Result<(Device, Vec<Queue>), DeviceError> {
        unimplemented!("WebGL does not support device creation without a surface")
    }

    fn new_device_with_surface(
        &self,
        info: DeviceDesc,
        window: &impl HasWindowHandle,
        display: &impl HasDisplayHandle,
    ) -> Result<(Device, Vec<Queue>, Surface), SurfaceError> {
        let handle = window.window_handle().map_err(|err| {
            tracing::error!("Failed to get window handle: {:?}", err);
            SurfaceError::SurfaceLost
        })?;

        match handle.as_raw() {
            RawWindowHandle::WebCanvas(web_canvas) => {}
            RawWindowHandle::Web(web) => {}
        }

        let context = Self::create_context(desc.canvas).map_err(|e| DeviceError::DeviceLost)?;

        let device = Device::new(context);

        let queues = vec![Queue::new(device.clone())];

        Ok((device, queues))
    }
}
