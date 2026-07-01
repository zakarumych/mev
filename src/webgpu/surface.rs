use std::fmt;

use crate::{
    Extent2,
    generic::{PixelFormat, SurfaceError},
};

use super::{
    Device, Image,
    from::{IntoWgpu, WgpuInto},
};

const RECONFIGURE_COOLDOWN: u64 = 10;

pub struct Surface {
    surface: wgpu::Surface<'static>,
    device: Device,
    config: wgpu::SurfaceConfiguration,
    available_formats: Vec<PixelFormat>,
    reconfigure_cooldown: u64,
    reconfigure: bool,
}

impl fmt::Debug for Surface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Surface").finish()
    }
}

impl Surface {
    pub(super) fn new(
        surface: wgpu::Surface<'static>,
        device: Device,
    ) -> Result<Self, SurfaceError> {
        let caps = surface.get_capabilities(device.wgpu_adapter());
        if caps.formats.is_empty() {
            return Err(SurfaceError::SurfaceLost);
        }

        let preferred_format = pick_format(&caps.formats, None);
        let available_formats = caps
            .formats
            .iter()
            .map(|&f| f.wgpu_into())
            .collect::<Vec<_>>();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: preferred_format,
            width: 800,
            height: 600,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(device.wgpu(), &config);

        Ok(Surface {
            surface,
            device,
            available_formats,
            config,
            reconfigure_cooldown: RECONFIGURE_COOLDOWN,
            reconfigure: false,
        })
    }

    fn configure(&mut self) -> Result<(), SurfaceError> {
        self.surface.configure(self.device.wgpu(), &self.config);
        self.reconfigure_cooldown = RECONFIGURE_COOLDOWN;
        self.reconfigure = false;

        Ok(())
    }
}

impl crate::traits::Resource for Surface {}

#[hidden_trait::expose]
impl crate::traits::Surface for Surface {
    fn available_formats(&self) -> &[PixelFormat] {
        &self.available_formats
    }

    fn preferred_format(&mut self, format: PixelFormat) {
        if self.available_formats.contains(&format) {
            self.config.format = format.into_wgpu();
        }
    }

    fn preferred_extent(&mut self, extent: Extent2) {
        if self.config.width != extent.width() || self.config.height != extent.height() {
            self.config.width = extent.width();
            self.config.height = extent.height();
            self.reconfigure = true;
        }
    }

    fn next_frame(&mut self) -> Result<Frame, SurfaceError> {
        self.reconfigure_cooldown = self.reconfigure_cooldown.saturating_sub(1);
        if self.reconfigure && self.reconfigure_cooldown == 0 {
            self.configure()?;
        }

        let texture = loop {
            match self.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t) => break t,
                wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                    self.reconfigure = true;
                    break t;
                }
                wgpu::CurrentSurfaceTexture::Outdated => {
                    self.configure()?;
                }
                wgpu::CurrentSurfaceTexture::Timeout => return Err(SurfaceError::SurfaceLost),
                wgpu::CurrentSurfaceTexture::Occluded => return Err(SurfaceError::SurfaceLost),
                wgpu::CurrentSurfaceTexture::Lost => return Err(SurfaceError::SurfaceLost),
                wgpu::CurrentSurfaceTexture::Validation => return Err(SurfaceError::SurfaceLost),
            }
        };

        let extent = crate::generic::ImageExtent::D2(crate::generic::Extent2::new(
            self.config.width,
            self.config.height,
        ));

        let view = texture.texture.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: None,
            dimension: None,
            usage: None,
            aspect: wgpu::TextureAspect::All,
            base_array_layer: 0,
            array_layer_count: None,
            base_mip_level: 0,
            mip_level_count: None,
        });

        let image = Image::new(view, 1, 1);

        Ok(Frame { texture, image })
    }
}

pub struct Frame {
    pub(super) texture: wgpu::SurfaceTexture,
    image: Image,
}

impl crate::traits::Resource for Frame {}

#[hidden_trait::expose]
impl crate::traits::Frame for Frame {
    fn image(&self) -> &Image {
        &self.image
    }
}

fn pick_format(
    formats: &[wgpu::TextureFormat],
    pixel_format: Option<PixelFormat>,
) -> wgpu::TextureFormat {
    if let Some(pixel_format) = pixel_format {
        let wgpu_format = pixel_format.into_wgpu();
        for &format in formats {
            if format == wgpu_format {
                return format;
            }
        }
    }
    for &format in formats {
        if format == wgpu::TextureFormat::Bgra8UnormSrgb {
            return format;
        }
    }
    for &format in formats {
        if format == wgpu::TextureFormat::Rgba8UnormSrgb {
            return format;
        }
    }
    for &format in formats {
        if format == wgpu::TextureFormat::Rgba8Unorm {
            return format;
        }
    }
    for &format in formats {
        if format == wgpu::TextureFormat::Bgra8Unorm {
            return format;
        }
    }

    return formats[0];
}
