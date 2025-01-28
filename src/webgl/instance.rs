use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::{convert::Infallible, fmt};
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{HtmlCanvasElement, WebGl2RenderingContext};

use crate::generic::{
    Capabilities, CreateError, DeviceCapabilities, DeviceDesc, FamilyCapabilities, Features,
    LoadError, QueueFlags,
};

use super::{Device, Queue};

pub(crate) enum LoadErrorKind {
    WebGL2NotSupported,
}

#[derive(Debug)]
pub(crate) enum CreateErrorKind {
    CanvasNotFound,
    WebGL2NotSupported,
    ContextCreationFailed,
}

impl fmt::Display for CreateErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CreateErrorKind::CanvasNotFound => write!(f, "canvas element not found"),
            CreateErrorKind::WebGL2NotSupported => write!(f, "WebGL 2.0 not supported"),
            CreateErrorKind::ContextCreationFailed => write!(f, "failed to create WebGL context"),
        }
    }
}

pub struct Instance {
    capabilities: Capabilities,
}

impl Instance {
    pub fn load() -> Result<Self, LoadError> {
        // Check if WebGL2RenderingContext is available
        if js_sys::Reflect::get(&js_sys::global(), &"WebGL2RenderingContext".into()).is_ok() {
            return Err(LoadError(LoadErrorKind::WebGL2NotSupported));
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

    fn create_context(
        canvas: &HtmlCanvasElement,
    ) -> Result<WebGl2RenderingContext, CreateErrorKind> {
        let context = canvas
            .get_context("webgl2")
            .map_err(|_| CreateErrorKind::ContextCreationFailed)?
            .ok_or(CreateErrorKind::WebGL2NotSupported)?
            .dyn_into::<WebGl2RenderingContext>()
            .map_err(|_| CreateErrorKind::WebGL2NotSupported)?;

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

    fn new_device(&self, desc: DeviceDesc) -> Result<(Device, Vec<Queue>), CreateError> {
        unimplemented!("WebGL does not support device creation without a surface")
    }

    fn new_device_with_surface(
        &self,
        info: DeviceDesc,
        window: &impl HasWindowHandle,
        display: &impl HasDisplayHandle,
    ) -> Result<(Device, Vec<Queue>, Surface), CreateError> {
        let context = Self::create_context(desc.canvas).map_err(|e| CreateError(e))?;

        let device = Device::new(context);

        let queues = vec![Queue::new(device.clone())];

        Ok((device, queues))
    }
}
