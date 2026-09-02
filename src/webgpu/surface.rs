use std::fmt;

use mev_proc::hidden_trait;

use crate::{
    ColorSpace, Extent2, ImageUsage, PresentMode,
    backend::from::{TryFromWgpu, TryIntoWgpu, TryWgpuInto, WgpuFrom},
    generic::{Image2D, ImageN, PixelFormat, SurfaceError},
};

use super::{
    Device, Image,
    from::{IntoWgpu, WgpuInto},
};

const RECONFIGURE_COOLDOWN: u64 = 10;

pub struct Surface {
    surface: wgpu::Surface<'static>,
    device: Device,
    caps: wgpu::SurfaceCapabilities,
    mev_formats: Vec<(PixelFormat, ColorSpace)>,
    config: wgpu::SurfaceConfiguration,
    preferred_extent: Option<Extent2>,
    preferred_format: Option<PixelFormat>,
    preferred_color_space: Option<ColorSpace>,
    preferred_mode: Option<PresentMode>,
    preferred_usage: Option<ImageUsage>,
    preferred_image_count: Option<u32>,
    cooldown: u64,
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
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::empty(),
            format: wgpu::TextureFormat::Bgra8Unorm,
            color_space: wgpu::SurfaceColorSpace::Srgb,
            width: 0,
            height: 0,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
            desired_maximum_frame_latency: 0,
        };

        let mut surface = Surface {
            surface,
            device,
            caps: wgpu::SurfaceCapabilities {
                formats: vec![],
                format_capabilities: vec![],
                alpha_modes: vec![],
                usages: wgpu::TextureUsages::empty(),
                present_modes: vec![],
            },
            mev_formats: Vec::new(),
            config,

            preferred_extent: None,
            preferred_format: None,
            preferred_color_space: None,
            preferred_mode: None,
            preferred_usage: None,
            preferred_image_count: None,

            cooldown: 0,
        };

        surface.update_surface_parameters();

        Ok(surface)
    }

    fn preprocess(&mut self) -> Result<(), SurfaceError> {
        self.cooldown = self.cooldown.saturating_sub(1);
        if self.cooldown > 0 {
            return Ok(());
        }

        self.update_surface_parameters();
        self.configure()?;
        Ok(())
    }

    fn update_surface_parameters(&mut self) {
        self.caps = self.surface.get_capabilities(self.device.wgpu_adapter());
    }

    fn configure(&mut self) -> Result<(), SurfaceError> {
        let (use_format, use_color_space) = pick_format(
            &self.caps.format_capabilities,
            self.preferred_format,
            self.preferred_color_space,
        );

        let use_mode = pick_mode(&self.caps.present_modes, self.preferred_mode);
        let use_usage = pick_usage(&self.caps, self.preferred_usage);
        let use_extent = match self.preferred_extent {
            Some(extent) if extent.width() > 0 && extent.height() > 0 => extent,
            _ => Extent2::new(1, 1),
        };

        let config = wgpu::SurfaceConfiguration {
            usage: use_usage,
            format: use_format,
            color_space: use_color_space,
            width: use_extent.width(),
            height: use_extent.height(),
            present_mode: use_mode,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: self.preferred_image_count.unwrap_or(1),
        };

        if config != self.config {
            self.config = config;
            self.surface.configure(self.device.wgpu(), &self.config);
            self.cooldown = RECONFIGURE_COOLDOWN;
        }

        Ok(())
    }
}

impl crate::traits::Resource for Surface {}

#[hidden_trait]
impl crate::traits::Surface for Surface {
    fn available_formats(&self) -> &[(PixelFormat, ColorSpace)] {
        &self.mev_formats
    }

    fn preferred_format(&mut self, format: PixelFormat) {
        if self.preferred_format != Some(format) {
            self.preferred_format = Some(format);
            self.cooldown = 0;
        }
    }

    fn preferred_color_space(&mut self, color_space: ColorSpace) {
        if self.preferred_color_space != Some(color_space) {
            self.preferred_color_space = Some(color_space);
            self.cooldown = 0;
        }
    }

    fn preferred_extent(&mut self, extent: Extent2) {
        if self.preferred_extent != Some(extent) {
            self.preferred_extent = Some(extent);
            self.cooldown = 0;
        }
    }

    fn preferred_present_mode(&mut self, mode: PresentMode) {
        if self.preferred_mode != Some(mode) {
            self.preferred_mode = Some(mode);
            self.cooldown = 0;
        }
    }

    fn preferred_usage(&mut self, usage: ImageUsage) {
        if self.preferred_usage != Some(usage) {
            self.preferred_usage = Some(usage);
            self.cooldown = 0;
        }
    }

    fn preferred_image_count(&mut self, count: u32) {
        if self.preferred_image_count != Some(count) {
            self.preferred_image_count = Some(count);
            self.cooldown = 0;
        }
    }

    fn next_frame(&mut self) -> Result<Frame, SurfaceError> {
        self.preprocess()?;

        let texture = loop {
            match self.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t) => break t,
                wgpu::CurrentSurfaceTexture::Suboptimal(t) => break t,
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

        Ok(Frame {
            texture,
            image: ImageN::new_unchecked(image),
        })
    }
}

pub struct Frame {
    pub(super) texture: wgpu::SurfaceTexture,
    image: Image2D,
}

impl crate::traits::Resource for Frame {}

#[hidden_trait]
impl crate::traits::Frame for Frame {
    fn image(&self) -> &Image2D {
        &self.image
    }
}

fn pick_format(
    formats: &[wgpu::SurfaceFormatCapabilities],
    preferred_format: Option<PixelFormat>,
    preferred_color_space: Option<ColorSpace>,
) -> (wgpu::TextureFormat, wgpu::SurfaceColorSpace) {
    match formats.len() {
        0 => panic!("No surface formats available"),
        _ => {
            let format_priority = [
                Some(wgpu::TextureFormat::Bgra8Unorm),
                Some(wgpu::TextureFormat::Rgba8Unorm),
                Some(wgpu::TextureFormat::Bgra8UnormSrgb),
                Some(wgpu::TextureFormat::Rgba8UnormSrgb),
                preferred_format.and_then(TryIntoWgpu::try_into_wgpu),
            ];

            let color_space_priority = [
                Some(wgpu::SurfaceColorSpaces::EXTENDED_SRGB_LINEAR),
                Some(wgpu::SurfaceColorSpaces::SRGB),
                preferred_color_space.map(IntoWgpu::into_wgpu),
            ];

            let mut best_candidate = (formats[0].format, wgpu::SurfaceColorSpace::Srgb);
            let mut best_score = 0usize;

            for &format_caps in formats {
                if PixelFormat::try_from_wgpu(format_caps.format).is_none() {
                    continue;
                }

                let format = format_caps.format;

                for color_space in format_caps.color_spaces {
                    if color_space
                        .intersection(
                            wgpu::SurfaceColorSpaces::SRGB
                                | wgpu::SurfaceColorSpaces::EXTENDED_SRGB_LINEAR,
                        )
                        .is_empty()
                    {
                        continue;
                    }

                    let mut score: usize = 1;

                    for (p, f) in format_priority.iter().enumerate() {
                        if Some(format) == *f {
                            score += (p + 1) * 256;
                        }
                    }

                    for (p, cs) in color_space_priority.iter().enumerate() {
                        if Some(color_space) == *cs {
                            score += (p + 1);
                        }
                    }

                    if score >= best_score {
                        best_candidate =
                            (format_caps.format, one_color_space(color_space).unwrap());
                        best_score = score;
                    }
                }
            }

            if best_score > 0 {
                return best_candidate;
            }

            panic!(
                "No usable format found for surface. Available formats: {:?}",
                formats
            );
        }
    }
}

fn one_color_space(spaces: wgpu::SurfaceColorSpaces) -> Option<wgpu::SurfaceColorSpace> {
    let known_spaces = [
        wgpu::SurfaceColorSpace::Srgb,
        wgpu::SurfaceColorSpace::ExtendedSrgbLinear,
    ];

    for known_space in known_spaces {
        if let Some(bit) = known_space.to_color_spaces() {
            if spaces.contains(bit) {
                return Some(known_space);
            }
        }
    }

    None
}

fn pick_mode(
    modes: &[wgpu::PresentMode],
    preferred_mode: Option<PresentMode>,
) -> wgpu::PresentMode {
    match modes.len() {
        0 => panic!("No surface present modes available"),
        _ => {
            let mode_priority = [
                Some(wgpu::PresentMode::Fifo),
                Some(wgpu::PresentMode::Immediate),
                Some(wgpu::PresentMode::Mailbox),
                preferred_mode.map(IntoWgpu::into_wgpu),
            ];

            let mut best_candidate = modes[0];
            let mut best_score = 0usize;

            for &candidate in modes {
                if PresentMode::try_from_wgpu(candidate).is_none() {
                    continue;
                }

                let mut score: usize = 1;

                for (p, mode) in mode_priority.iter().enumerate() {
                    if Some(candidate) == *mode {
                        score += (p + 1) * 256;
                    }
                }

                if score > best_score {
                    best_candidate = candidate;
                    best_score = score;
                }
            }

            if best_score > 0 {
                return best_candidate;
            }

            panic!(
                "No usable present modes found for surface. Available present modes: {:?}",
                modes
            );
        }
    }
}

fn pick_usage(
    caps: &wgpu::SurfaceCapabilities,
    preferred_usage: Option<ImageUsage>,
) -> wgpu::TextureUsages {
    let usage = wgpu::TextureUsages::wgpu_from(preferred_usage.unwrap_or(ImageUsage::all()));
    usage & caps.usages
}
