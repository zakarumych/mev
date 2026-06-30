use std::fmt;

use crate::generic::{PixelFormat, SurfaceError};

use super::{image::Image, Device};

const SUBOPTIMAL_RETIRE_COOLDOWN: u64 = 10;

#[derive(Clone, Copy, PartialEq, Eq)]
enum SuboptimalRetire {
    Cooldown(u64),
    Retire,
}

pub struct Surface {
    surface: wgpu::Surface<'static>,
    device: Device,
    format: PixelFormat,
    config: wgpu::SurfaceConfiguration,
    suboptimal_retire: SuboptimalRetire,
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

        let wgpu_format = caps.formats[0];
        let format = wgpu_format_to_pixel(wgpu_format).unwrap_or(PixelFormat::Bgra8Unorm);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu_format,
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
            format,
            config,
            suboptimal_retire: SuboptimalRetire::Cooldown(SUBOPTIMAL_RETIRE_COOLDOWN),
        })
    }

    fn configure(&mut self) -> Result<(), SurfaceError> {
        let caps = self.surface.get_capabilities(self.device.wgpu_adapter());
        if caps.formats.is_empty() {
            return Err(SurfaceError::SurfaceLost);
        }

        let wgpu_format = caps.formats[0];
        let format = wgpu_format_to_pixel(wgpu_format).unwrap_or(PixelFormat::Bgra8Unorm);

        self.config.format = wgpu_format;
        self.surface.configure(self.device.wgpu(), &self.config);
        self.format = format;
        self.suboptimal_retire = SuboptimalRetire::Cooldown(SUBOPTIMAL_RETIRE_COOLDOWN);

        Ok(())
    }
}

impl crate::traits::Resource for Surface {}

#[hidden_trait::expose]
impl crate::traits::Surface for Surface {
    fn next_frame(&mut self) -> Result<Frame, SurfaceError> {
        if let SuboptimalRetire::Retire = self.suboptimal_retire {
            self.configure()?;
        }

        let texture = loop {
            match self.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t) => break t,
                wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                    match self.suboptimal_retire {
                        SuboptimalRetire::Retire => {}
                        SuboptimalRetire::Cooldown(0) => {
                            self.suboptimal_retire = SuboptimalRetire::Retire;
                        }
                        SuboptimalRetire::Cooldown(n) => {
                            self.suboptimal_retire = SuboptimalRetire::Cooldown(n - 1);
                        }
                    }

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

fn wgpu_format_to_pixel(f: wgpu::TextureFormat) -> Option<PixelFormat> {
    Some(match f {
        wgpu::TextureFormat::Bgra8Unorm => PixelFormat::Bgra8Unorm,
        wgpu::TextureFormat::Bgra8UnormSrgb => PixelFormat::Bgra8Srgb,
        wgpu::TextureFormat::Rgba8Unorm => PixelFormat::Rgba8Unorm,
        wgpu::TextureFormat::Rgba8UnormSrgb => PixelFormat::Rgba8Srgb,
        wgpu::TextureFormat::Rgba16Float => PixelFormat::Rgba16Float,
        _ => return None,
    })
}
