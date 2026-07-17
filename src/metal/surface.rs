use std::fmt;

use objc2::{msg_send, rc::Retained, runtime::ProtocolObject};
use objc2_core_foundation::{CGFloat, CGRect, CGSize};
use objc2_foundation::NSObject;
use objc2_metal::MTLDrawable;
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};

use crate::{
    ColorSpace, Extent2, PixelFormat, PresentMode,
    backend::from::{IntoMetal, TryIntoMetal, TryMetalInto},
    generic::{PipelineStages, SurfaceError},
};

use super::{Image, Queue};

const RECONFIGURE_COOLDOWN: u64 = 10;

enum Backing {
    Layer {
        layer: Retained<CAMetalLayer>,
        view: Option<Retained<NSObject>>,
    },
    Fake {
        image: Image,
        format: (PixelFormat, ColorSpace),
    },
}

pub struct Surface {
    preferred_extent: Option<Extent2>,
    reconfigure_cooldown: u64,
    reconfigure: bool,
    backing: Backing,
}

unsafe impl Sync for Surface {}
unsafe impl Send for Surface {}

impl crate::traits::Resource for Surface {}

impl Surface {
    pub(super) fn new(layer: Retained<CAMetalLayer>, view: Option<Retained<NSObject>>) -> Self {
        Surface {
            preferred_extent: None,
            reconfigure_cooldown: 0,
            reconfigure: false,
            backing: Backing::Layer { layer, view },
        }
    }

    pub(super) fn fake(image: Image) -> Self {
        let format = image.format();
        Surface {
            preferred_extent: None,
            reconfigure_cooldown: 0,
            reconfigure: false,
            backing: Backing::Fake {
                image,
                format: (format, ColorSpace::SrgbNonLinear),
            },
        }
    }
}

unsafe fn window_scale_factor(view: *const NSObject) -> (f64, f64) {
    let mut scale_factor_x: CGFloat = 1.0;
    let mut scale_factor_y: CGFloat = 1.0;

    #[cfg(target_os = "macos")]
    unsafe {
        let window: *const NSObject = msg_send![view, window];
        if !window.is_null() {
            let scale_factor: CGFloat = msg_send![window, backingScaleFactor];
            scale_factor_x = scale_factor;
            scale_factor_y = scale_factor;
        }
    }

    #[cfg(not(target_os = "macos"))]
    unsafe {
        let transform: CGAffineTransform = msg_send![view, transform];
        scale_factor_x = transform.tx;
        scale_factor_y = transform.ty;
    }

    (scale_factor_x, scale_factor_y)
}

unsafe fn view_size(view: *const NSObject) -> CGSize {
    unsafe {
        let frame: CGRect = msg_send![view, bounds];
        frame.size
    }
}

#[hidden_trait::expose]
impl crate::traits::Surface for Surface {
    fn available_formats(&self) -> &[(PixelFormat, ColorSpace)] {
        match &self.backing {
            Backing::Layer { .. } => supported_formats(),
            Backing::Fake { format, .. } => std::slice::from_ref(format),
        }
    }

    fn preferred_format(&mut self, format: PixelFormat) {
        let Backing::Layer { layer, .. } = &self.backing else {
            // Fake surface has a fixed format tied to its backing image.
            return;
        };

        if supported_formats().contains(&(format, ColorSpace::SrgbNonLinear)) {
            let metal_format = format.expect_into_metal();
            if layer.pixelFormat() != metal_format {
                layer.setPixelFormat(metal_format);
                self.reconfigure = true;
            }
        }
    }

    fn preferred_color_space(&mut self, color_space: ColorSpace) {}

    fn preferred_extent(&mut self, extent: Extent2) {
        let Backing::Layer { layer, .. } = &self.backing else {
            // Fake surface has a fixed extent tied to its backing image.
            return;
        };

        let draw_size = layer.drawableSize();

        if draw_size.width != extent.width() as f64 || draw_size.height != extent.height() as f64 {
            self.reconfigure = true;
        }

        self.preferred_extent = Some(extent);
    }

    fn preferred_present_mode(&mut self, _mode: PresentMode) {}

    fn preferred_usage(&mut self, _usage: crate::ImageUsage) {}

    fn preferred_image_count(&mut self, count: u32) {
        let Backing::Layer { layer, .. } = &self.backing else {
            // Fake surface has a fixed extent tied to its backing image.
            return;
        };

        match count {
            0..=2 => layer.setMaximumDrawableCount(2),
            3.. => layer.setMaximumDrawableCount(3),
        }
    }

    fn next_frame(&mut self) -> Result<Frame, SurfaceError> {
        let (layer, view) = match &self.backing {
            Backing::Fake { image, .. } => {
                return Ok(Frame {
                    image: image.clone(),
                    drawable: None,
                });
            }
            Backing::Layer { layer, view } => (layer, view),
        };

        self.reconfigure_cooldown = self.reconfigure_cooldown.saturating_sub(1);
        if self.reconfigure && self.reconfigure_cooldown == 0 {
            if let Some(view) = view {
                unsafe {
                    let draw_size = layer.drawableSize();

                    let scale = window_scale_factor(Retained::as_ptr(view));
                    let size = view_size(Retained::as_ptr(view));

                    if draw_size.width != size.width * scale.0
                        || draw_size.height != size.height * scale.1
                    {
                        layer.setDrawableSize(CGSize {
                            width: size.width * scale.0,
                            height: size.height * scale.1,
                        });

                        self.reconfigure_cooldown = RECONFIGURE_COOLDOWN;
                        self.reconfigure = false;
                    }
                }
            } else {
                if let Some(preferred_extent) = self.preferred_extent {
                    layer.setDrawableSize(CGSize {
                        width: preferred_extent.width() as f64,
                        height: preferred_extent.height() as f64,
                    });
                }
            }
        }

        let drawable = layer.nextDrawable().ok_or(SurfaceError::SurfaceLost)?;

        let image = Image::new(drawable.texture());
        Ok(Frame {
            drawable: Some(drawable),
            image,
        })
    }
}

pub struct Frame {
    image: Image,
    drawable: Option<Retained<ProtocolObject<dyn CAMetalDrawable>>>,
}

unsafe impl Send for Frame {}
unsafe impl Sync for Frame {}

impl crate::traits::Resource for Frame {}

impl Frame {
    #[inline]
    pub(super) fn drawable(&self) -> Option<&ProtocolObject<dyn MTLDrawable>> {
        self.drawable.as_deref().map(AsRef::as_ref)
    }
}

#[hidden_trait::expose]
impl crate::traits::Frame for Frame {
    #[inline]
    fn image(&self) -> &Image {
        &self.image
    }
}

const fn supported_formats() -> &'static [(PixelFormat, ColorSpace)] {
    &[
        (PixelFormat::Bgra8Unorm, ColorSpace::SrgbNonLinear),
        (PixelFormat::Bgra8Srgb, ColorSpace::SrgbNonLinear),
        (PixelFormat::Rgba16Float, ColorSpace::SrgbNonLinear),
    ]
}
