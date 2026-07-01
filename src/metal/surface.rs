use std::fmt;

use core_graphics_types::{
    base::CGFloat,
    geometry::{CGRect, CGSize},
};
use objc::{msg_send, runtime::Object, sel, sel_impl};

use crate::{
    Extent2, PixelFormat,
    backend::from::{IntoMetal, TryIntoMetal, TryMetalInto},
    generic::{PipelineStages, SurfaceError},
};

use super::{Image, Queue};

const RECONFIGURE_COOLDOWN: u64 = 10;

pub struct Surface {
    layer: metal::MetalLayer,
    view: *mut objc::runtime::Object,
    preferred_extent: Option<Extent2>,
    reconfigure_cooldown: u64,
    reconfigure: bool,
}

unsafe impl Sync for Surface {}
unsafe impl Send for Surface {}

impl crate::traits::Resource for Surface {}

impl Drop for Surface {
    fn drop(&mut self) {
        if !self.view.is_null() {
            unsafe {
                let () = msg_send![self.view, release];
            }
        }
    }
}

impl Surface {
    pub(super) fn new(layer: metal::MetalLayer, view: *mut Object) -> Self {
        if !view.is_null() {
            unsafe {
                let () = msg_send![view, retain];
            }
        }

        Surface {
            layer,
            view,
            preferred_extent: None,
            reconfigure_cooldown: 0,
            reconfigure: false,
        }
    }
}

unsafe fn window_scale_factor(view: *mut Object) -> f64 {
    let mut scale_factor: CGFloat = 1.0;
    unsafe {
        let window: *mut Object = msg_send![view, window];
        if !window.is_null() {
            scale_factor = msg_send![window, backingScaleFactor];
        }
    }
    scale_factor
}

unsafe fn view_size(view: *mut Object) -> CGSize {
    unsafe {
        let frame: CGRect = msg_send![view, bounds];
        frame.size
    }
}

#[hidden_trait::expose]
impl crate::traits::Surface for Surface {
    fn available_formats(&self) -> &[PixelFormat] {
        supported_formats()
    }

    fn preferred_format(&mut self, format: PixelFormat) {
        if supported_formats().contains(&format) {
            let metal_format = format.expect_into_metal();
            if self.layer.pixel_format() != metal_format {
                self.layer.set_pixel_format(metal_format);
                self.reconfigure = true;
            }
        }
    }

    fn preferred_extent(&mut self, extent: Extent2) {
        let draw_size = self.layer.drawable_size();

        if draw_size.width != extent.width() as f64 || draw_size.height != extent.height() as f64 {
            self.reconfigure = true;
        }

        self.preferred_extent = Some(extent);
    }

    fn next_frame(&mut self) -> Result<Frame, SurfaceError> {
        self.reconfigure_cooldown = self.reconfigure_cooldown.saturating_sub(1);
        if self.reconfigure && self.reconfigure_cooldown == 0 {
            if !self.view.is_null() {
                unsafe {
                    let draw_size = self.layer.drawable_size();

                    let scale = window_scale_factor(self.view);
                    let size = view_size(self.view);

                    if draw_size.width != size.width * scale
                        || draw_size.height != size.height * scale
                    {
                        self.layer.set_drawable_size(CGSize {
                            width: size.width * scale,
                            height: size.height * scale,
                        });

                        self.reconfigure_cooldown = RECONFIGURE_COOLDOWN;
                        self.reconfigure = false;
                    }
                }
            } else {
                if let Some(preferred_extent) = self.preferred_extent {
                    self.layer.set_drawable_size(CGSize {
                        width: preferred_extent.width() as f64,
                        height: preferred_extent.height() as f64,
                    });
                }
            }
        }

        let drawable = self
            .layer
            .next_drawable()
            .ok_or(SurfaceError::SurfaceLost)?;

        let image = Image::new(drawable.texture().to_owned());
        Ok(Frame {
            drawable: Some(drawable.to_owned()),
            image,
        })
    }
}

pub struct Frame {
    image: Image,
    drawable: Option<metal::MetalDrawable>,
}

impl crate::traits::Resource for Frame {}

impl Frame {
    #[inline]
    pub(super) fn drawable(&self) -> Option<&metal::MetalDrawableRef> {
        self.drawable.as_deref()
    }
}

#[hidden_trait::expose]
impl crate::traits::Frame for Frame {
    #[inline]
    fn image(&self) -> &Image {
        &self.image
    }
}

const fn supported_formats() -> &'static [PixelFormat] {
    &[
        PixelFormat::Bgra8Unorm,
        PixelFormat::Bgra8Srgb,
        PixelFormat::Rgba16Float,
    ]
}
