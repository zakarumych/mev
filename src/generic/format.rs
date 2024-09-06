

/// Format of the pixel.
///
/// It specifies channels, channel bits and data type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// 8-bit unsigned normalized red channel.
    R8Unorm,

    /// 8-bit signed normalized red channel.
    R8Snorm,

    /// 8-bit unsigned red channel.
    R8Uint,

    /// 8-bit signed red channel.
    R8Sint,

    /// 8-bit unsigned normalized red channel in sRGB color space.
    R8Srgb,

    /// 16-bit unsigned normalized red channel.
    R16Unorm,

    /// 16-bit signed normalized red channel.
    R16Snorm,

    /// 16-bit unsigned red channel.
    R16Uint,

    /// 16-bit signed red channel.
    R16Sint,

    /// 16-bit floating-point red channel.
    R16Float,

    /// 32-bit unsigned normalized red channel.
    R32Unorm,

    /// 32-bit signed normalized red channel.
    R32Snorm,

    /// 32-bit unsigned red channel.
    R32Uint,

    /// 32-bit signed red channel.
    R32Sint,

    /// 32-bit floating-point red channel.
    R32Float,

    /// 8-bit unsigned normalized red and green channels.
    Rg8Unorm,

    /// 8-bit signed normalized red and green channels.
    Rg8Snorm,

    /// 8-bit unsigned red and green channels.
    Rg8Uint,

    /// 8-bit signed red and green channels.
    Rg8Sint,

    /// 8-bit unsigned normalized red and green channels in sRGB color space.
    Rg8Srgb,

    /// 16-bit unsigned normalized red and green channels.
    Rg16Unorm,

    /// 16-bit signed normalized red and green channels.
    Rg16Snorm,

    /// 16-bit unsigned red and green channels.
    Rg16Uint,

    /// 16-bit signed red and green channels.
    Rg16Sint,

    /// 16-bit floating-point red and green channels.
    Rg16Float,

    /// 32-bit unsigned normalized red and green channels.
    Rg32Unorm,

    /// 32-bit signed normalized red and green channels.
    Rg32Snorm,

    /// 32-bit unsigned red and green channels.
    Rg32Uint,

    /// 32-bit signed red and green channels.
    Rg32Sint,

    /// 32-bit floating-point red and green channels.
    Rg32Float,

    /// 8-bit unsigned normalized red, green and blue channels.
    Rgb8Unorm,

    /// 8-bit signed normalized red, green and blue channels.
    Rgb8Snorm,

    /// 8-bit unsigned red, green and blue channels.
    Rgb8Uint,

    /// 8-bit signed red, green and blue channels.
    Rgb8Sint,

    /// 8-bit unsigned normalized red, green and blue channels in sRGB color space.
    Rgb8Srgb,

    /// 16-bit unsigned normalized red, green and blue channels.
    Rgb16Unorm,

    /// 16-bit signed normalized red, green and blue channels.
    Rgb16Snorm,

    /// 16-bit unsigned red, green and blue channels.
    Rgb16Uint,

    /// 16-bit signed red, green and blue channels.
    Rgb16Sint,

    /// 16-bit floating-point red, green and blue channels.
    Rgb16Float,

    /// 32-bit unsigned normalized red, green and blue channels.
    Rgb32Unorm,

    /// 32-bit signed normalized red, green and blue channels.
    Rgb32Snorm,

    /// 32-bit unsigned red, green and blue channels.
    Rgb32Uint,

    /// 32-bit signed red, green and blue channels.
    Rgb32Sint,

    /// 32-bit floating-point red, green and blue channels.
    Rgb32Float,

    /// 8-bit unsigned normalized red, green, blue and alpha channels.
    Rgba8Unorm,

    /// 8-bit signed normalized red, green, blue and alpha channels.
    Rgba8Snorm,

    /// 8-bit unsigned red, green, blue and alpha channels.
    Rgba8Uint,

    /// 8-bit signed red, green, blue and alpha channels.
    Rgba8Sint,

    /// 8-bit unsigned normalized red, green, blue and alpha channels in sRGB color space.
    Rgba8Srgb,

    /// 16-bit unsigned normalized red, green, blue and alpha channels.
    Rgba16Unorm,

    /// 16-bit signed normalized red, green, blue and alpha channels.
    Rgba16Snorm,

    /// 16-bit unsigned red, green, blue and alpha channels.
    Rgba16Uint,

    /// 16-bit signed red, green, blue and alpha channels.
    Rgba16Sint,

    /// 16-bit floating-point red, green, blue and alpha channels.
    Rgba16Float,

    /// 32-bit unsigned normalized red, green, blue and alpha channels.
    Rgba32Unorm,

    /// 32-bit signed normalized red, green, blue and alpha channels.
    Rgba32Snorm,

    /// 32-bit unsigned red, green, blue and alpha channels.
    Rgba32Uint,

    /// 32-bit signed red, green, blue and alpha channels.
    Rgba32Sint,

    /// 32-bit floating-point red, green, blue and alpha channels.
    Rgba32Float,

    /// 8-bit unsigned normalized blue, green and red channels.
    Bgr8Unorm,

    /// 8-bit signed normalized blue, green and red channels.
    Bgr8Snorm,

    /// 8-bit unsigned blue, green and red channels.
    Bgr8Uint,

    /// 8-bit signed blue, green and red channels.
    Bgr8Sint,

    /// 8-bit unsigned normalized blue, green and red channels in sRGB color space.
    Bgr8Srgb,

    /// 8-bit unsigned normalized blue, green, red and alpha channels.
    Bgra8Unorm,

    /// 8-bit signed normalized blue, green, red and alpha channels.
    Bgra8Snorm,

    /// 8-bit unsigned blue, green, red and alpha channels.
    Bgra8Uint,

    /// 8-bit signed blue, green, red and alpha channels.
    Bgra8Sint,

    /// 8-bit unsigned normalized blue, green, red and alpha channels in sRGB color space.
    Bgra8Srgb,

    /// 16-bit unsigned normalized depth channel.
    D16Unorm,

    /// 32-bit floating-point depth channel.
    D32Float,

    /// 8-bit unsigned stencil channel.
    S8Uint,

    /// 16-bit unsigned normalized depth and 8-bit unsigned stencil channels.
    D16UnormS8Uint,

    /// 24-bit unsigned normalized depth and 8-bit unsigned stencil channels.
    D24UnormS8Uint,

    /// 32-bit floating-point depth and 8-bit unsigned stencil channels.
    D32FloatS8Uint,
}

impl PixelFormat {
    #[cfg_attr(feature = "inline-more", inline(always))]
    pub fn is_color(&self) -> bool {
        match self {
            PixelFormat::R8Unorm
            | PixelFormat::R8Snorm
            | PixelFormat::R8Uint
            | PixelFormat::R8Sint
            | PixelFormat::R8Srgb
            | PixelFormat::R16Unorm
            | PixelFormat::R16Snorm
            | PixelFormat::R16Uint
            | PixelFormat::R16Sint
            | PixelFormat::R16Float
            | PixelFormat::R32Unorm
            | PixelFormat::R32Snorm
            | PixelFormat::R32Uint
            | PixelFormat::R32Sint
            | PixelFormat::R32Float
            | PixelFormat::Rg8Unorm
            | PixelFormat::Rg8Snorm
            | PixelFormat::Rg8Uint
            | PixelFormat::Rg8Sint
            | PixelFormat::Rg8Srgb
            | PixelFormat::Rg16Unorm
            | PixelFormat::Rg16Snorm
            | PixelFormat::Rg16Uint
            | PixelFormat::Rg16Sint
            | PixelFormat::Rg16Float
            | PixelFormat::Rg32Unorm
            | PixelFormat::Rg32Snorm
            | PixelFormat::Rg32Uint
            | PixelFormat::Rg32Sint
            | PixelFormat::Rg32Float
            | PixelFormat::Rgb8Unorm
            | PixelFormat::Rgb8Snorm
            | PixelFormat::Rgb8Uint
            | PixelFormat::Rgb8Sint
            | PixelFormat::Rgb8Srgb
            | PixelFormat::Rgb16Unorm
            | PixelFormat::Rgb16Snorm
            | PixelFormat::Rgb16Uint
            | PixelFormat::Rgb16Sint
            | PixelFormat::Rgb16Float
            | PixelFormat::Rgb32Unorm
            | PixelFormat::Rgb32Snorm
            | PixelFormat::Rgb32Uint
            | PixelFormat::Rgb32Sint
            | PixelFormat::Rgb32Float
            | PixelFormat::Rgba8Unorm
            | PixelFormat::Rgba8Snorm
            | PixelFormat::Rgba8Uint
            | PixelFormat::Rgba8Sint
            | PixelFormat::Rgba8Srgb
            | PixelFormat::Rgba16Unorm
            | PixelFormat::Rgba16Snorm
            | PixelFormat::Rgba16Uint
            | PixelFormat::Rgba16Sint
            | PixelFormat::Rgba16Float
            | PixelFormat::Rgba32Unorm
            | PixelFormat::Rgba32Snorm
            | PixelFormat::Rgba32Uint
            | PixelFormat::Rgba32Sint
            | PixelFormat::Rgba32Float
            | PixelFormat::Bgr8Unorm
            | PixelFormat::Bgr8Snorm
            | PixelFormat::Bgr8Uint
            | PixelFormat::Bgr8Sint
            | PixelFormat::Bgr8Srgb
            | PixelFormat::Bgra8Unorm
            | PixelFormat::Bgra8Snorm
            | PixelFormat::Bgra8Uint
            | PixelFormat::Bgra8Sint
            | PixelFormat::Bgra8Srgb => true,
            PixelFormat::D16Unorm
            | PixelFormat::D32Float
            | PixelFormat::S8Uint
            | PixelFormat::D16UnormS8Uint
            | PixelFormat::D24UnormS8Uint
            | PixelFormat::D32FloatS8Uint => false,
        }
    }

    #[cfg_attr(feature = "inline-more", inline(always))]
    pub fn is_depth(&self) -> bool {
        match self {
            PixelFormat::R8Unorm
            | PixelFormat::R8Srgb
            | PixelFormat::R8Snorm
            | PixelFormat::R8Uint
            | PixelFormat::R8Sint
            | PixelFormat::R16Unorm
            | PixelFormat::R16Snorm
            | PixelFormat::R16Uint
            | PixelFormat::R16Sint
            | PixelFormat::R16Float
            | PixelFormat::R32Unorm
            | PixelFormat::R32Snorm
            | PixelFormat::R32Uint
            | PixelFormat::R32Sint
            | PixelFormat::R32Float
            | PixelFormat::Rg8Unorm
            | PixelFormat::Rg8Srgb
            | PixelFormat::Rg8Snorm
            | PixelFormat::Rg8Uint
            | PixelFormat::Rg8Sint
            | PixelFormat::Rg16Unorm
            | PixelFormat::Rg16Snorm
            | PixelFormat::Rg16Uint
            | PixelFormat::Rg16Sint
            | PixelFormat::Rg16Float
            | PixelFormat::Rg32Unorm
            | PixelFormat::Rg32Snorm
            | PixelFormat::Rg32Uint
            | PixelFormat::Rg32Sint
            | PixelFormat::Rg32Float
            | PixelFormat::Rgb8Unorm
            | PixelFormat::Rgb8Srgb
            | PixelFormat::Rgb8Snorm
            | PixelFormat::Rgb8Uint
            | PixelFormat::Rgb8Sint
            | PixelFormat::Rgb16Unorm
            | PixelFormat::Rgb16Snorm
            | PixelFormat::Rgb16Uint
            | PixelFormat::Rgb16Sint
            | PixelFormat::Rgb16Float
            | PixelFormat::Rgb32Unorm
            | PixelFormat::Rgb32Snorm
            | PixelFormat::Rgb32Uint
            | PixelFormat::Rgb32Sint
            | PixelFormat::Rgb32Float
            | PixelFormat::Rgba8Unorm
            | PixelFormat::Rgba8Srgb
            | PixelFormat::Rgba8Snorm
            | PixelFormat::Rgba8Uint
            | PixelFormat::Rgba8Sint
            | PixelFormat::Rgba16Unorm
            | PixelFormat::Rgba16Snorm
            | PixelFormat::Rgba16Uint
            | PixelFormat::Rgba16Sint
            | PixelFormat::Rgba16Float
            | PixelFormat::Rgba32Unorm
            | PixelFormat::Rgba32Snorm
            | PixelFormat::Rgba32Uint
            | PixelFormat::Rgba32Sint
            | PixelFormat::Rgba32Float
            | PixelFormat::Bgr8Unorm
            | PixelFormat::Bgr8Srgb
            | PixelFormat::Bgr8Snorm
            | PixelFormat::Bgr8Uint
            | PixelFormat::Bgr8Sint
            | PixelFormat::Bgra8Unorm
            | PixelFormat::Bgra8Srgb
            | PixelFormat::Bgra8Snorm
            | PixelFormat::Bgra8Uint
            | PixelFormat::Bgra8Sint => false,
            PixelFormat::S8Uint => false,
            PixelFormat::D16Unorm
            | PixelFormat::D32Float
            | PixelFormat::D16UnormS8Uint
            | PixelFormat::D24UnormS8Uint
            | PixelFormat::D32FloatS8Uint => true,
        }
    }

    #[cfg_attr(feature = "inline-more", inline(always))]
    pub fn is_stencil(&self) -> bool {
        match self {
            PixelFormat::R8Unorm
            | PixelFormat::R8Srgb
            | PixelFormat::R8Snorm
            | PixelFormat::R8Uint
            | PixelFormat::R8Sint
            | PixelFormat::R16Unorm
            | PixelFormat::R16Snorm
            | PixelFormat::R16Uint
            | PixelFormat::R16Sint
            | PixelFormat::R16Float
            | PixelFormat::R32Unorm
            | PixelFormat::R32Snorm
            | PixelFormat::R32Uint
            | PixelFormat::R32Sint
            | PixelFormat::R32Float
            | PixelFormat::Rg8Unorm
            | PixelFormat::Rg8Srgb
            | PixelFormat::Rg8Snorm
            | PixelFormat::Rg8Uint
            | PixelFormat::Rg8Sint
            | PixelFormat::Rg16Unorm
            | PixelFormat::Rg16Snorm
            | PixelFormat::Rg16Uint
            | PixelFormat::Rg16Sint
            | PixelFormat::Rg16Float
            | PixelFormat::Rg32Unorm
            | PixelFormat::Rg32Snorm
            | PixelFormat::Rg32Uint
            | PixelFormat::Rg32Sint
            | PixelFormat::Rg32Float
            | PixelFormat::Rgb8Unorm
            | PixelFormat::Rgb8Srgb
            | PixelFormat::Rgb8Snorm
            | PixelFormat::Rgb8Uint
            | PixelFormat::Rgb8Sint
            | PixelFormat::Rgb16Unorm
            | PixelFormat::Rgb16Snorm
            | PixelFormat::Rgb16Uint
            | PixelFormat::Rgb16Sint
            | PixelFormat::Rgb16Float
            | PixelFormat::Rgb32Unorm
            | PixelFormat::Rgb32Snorm
            | PixelFormat::Rgb32Uint
            | PixelFormat::Rgb32Sint
            | PixelFormat::Rgb32Float
            | PixelFormat::Rgba8Unorm
            | PixelFormat::Rgba8Srgb
            | PixelFormat::Rgba8Snorm
            | PixelFormat::Rgba8Uint
            | PixelFormat::Rgba8Sint
            | PixelFormat::Rgba16Unorm
            | PixelFormat::Rgba16Snorm
            | PixelFormat::Rgba16Uint
            | PixelFormat::Rgba16Sint
            | PixelFormat::Rgba16Float
            | PixelFormat::Rgba32Unorm
            | PixelFormat::Rgba32Snorm
            | PixelFormat::Rgba32Uint
            | PixelFormat::Rgba32Sint
            | PixelFormat::Rgba32Float
            | PixelFormat::Bgr8Unorm
            | PixelFormat::Bgr8Srgb
            | PixelFormat::Bgr8Snorm
            | PixelFormat::Bgr8Uint
            | PixelFormat::Bgr8Sint
            | PixelFormat::Bgra8Unorm
            | PixelFormat::Bgra8Srgb
            | PixelFormat::Bgra8Snorm
            | PixelFormat::Bgra8Uint
            | PixelFormat::Bgra8Sint => false,
            PixelFormat::D16Unorm | PixelFormat::D32Float => false,
            PixelFormat::S8Uint
            | PixelFormat::D16UnormS8Uint
            | PixelFormat::D24UnormS8Uint
            | PixelFormat::D32FloatS8Uint => true,
        }
    }

    #[cfg_attr(feature = "inline-more", inline(always))]
    pub fn size(&self) -> usize {
        match self {
            PixelFormat::R8Unorm
            | PixelFormat::R8Snorm
            | PixelFormat::R8Uint
            | PixelFormat::R8Sint
            | PixelFormat::R8Srgb => 1,
            PixelFormat::R16Unorm
            | PixelFormat::R16Snorm
            | PixelFormat::R16Uint
            | PixelFormat::R16Sint
            | PixelFormat::R16Float => 2,
            PixelFormat::R32Unorm
            | PixelFormat::R32Snorm
            | PixelFormat::R32Uint
            | PixelFormat::R32Sint
            | PixelFormat::R32Float => 4,
            PixelFormat::Rg8Unorm
            | PixelFormat::Rg8Snorm
            | PixelFormat::Rg8Uint
            | PixelFormat::Rg8Sint
            | PixelFormat::Rg8Srgb => 2,
            PixelFormat::Rg16Unorm
            | PixelFormat::Rg16Snorm
            | PixelFormat::Rg16Uint
            | PixelFormat::Rg16Sint
            | PixelFormat::Rg16Float => 4,
            PixelFormat::Rg32Unorm
            | PixelFormat::Rg32Snorm
            | PixelFormat::Rg32Uint
            | PixelFormat::Rg32Sint
            | PixelFormat::Rg32Float => 8,
            PixelFormat::Rgb8Unorm
            | PixelFormat::Rgb8Snorm
            | PixelFormat::Rgb8Uint
            | PixelFormat::Rgb8Sint
            | PixelFormat::Rgb8Srgb => 3,
            PixelFormat::Rgb16Unorm
            | PixelFormat::Rgb16Snorm
            | PixelFormat::Rgb16Uint
            | PixelFormat::Rgb16Sint
            | PixelFormat::Rgb16Float => 6,
            PixelFormat::Rgb32Unorm
            | PixelFormat::Rgb32Snorm
            | PixelFormat::Rgb32Uint
            | PixelFormat::Rgb32Sint
            | PixelFormat::Rgb32Float => 12,
            PixelFormat::Rgba8Unorm
            | PixelFormat::Rgba8Snorm
            | PixelFormat::Rgba8Uint
            | PixelFormat::Rgba8Sint
            | PixelFormat::Rgba8Srgb => 4,
            PixelFormat::Rgba16Unorm
            | PixelFormat::Rgba16Snorm
            | PixelFormat::Rgba16Uint
            | PixelFormat::Rgba16Sint
            | PixelFormat::Rgba16Float => 8,
            PixelFormat::Rgba32Unorm
            | PixelFormat::Rgba32Snorm
            | PixelFormat::Rgba32Uint
            | PixelFormat::Rgba32Sint
            | PixelFormat::Rgba32Float => 16,
            PixelFormat::Bgr8Unorm
            | PixelFormat::Bgr8Snorm
            | PixelFormat::Bgr8Uint
            | PixelFormat::Bgr8Sint
            | PixelFormat::Bgr8Srgb => 3,
            PixelFormat::Bgra8Unorm
            | PixelFormat::Bgra8Snorm
            | PixelFormat::Bgra8Uint
            | PixelFormat::Bgra8Sint
            | PixelFormat::Bgra8Srgb => 4,
            PixelFormat::D16Unorm => 2,
            PixelFormat::D32Float => 4,
            PixelFormat::S8Uint => 1,
            PixelFormat::D16UnormS8Uint => 3,
            PixelFormat::D24UnormS8Uint => 4,
            PixelFormat::D32FloatS8Uint => 5,
        }
    }

    #[cfg_attr(feature = "inline-more", inline(always))]
    pub fn is_srgb(&self) -> bool {
        match self {
            PixelFormat::R8Srgb
            | PixelFormat::Rg8Srgb
            | PixelFormat::Rgb8Srgb
            | PixelFormat::Rgba8Srgb
            | PixelFormat::Bgr8Srgb
            | PixelFormat::Bgra8Srgb => true,
            _ => false,
        }
    }
}

/// Format of the vertex attribute.
/// 
/// It specifies the data type and number of components.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VertexFormat {
    /// 8-bit unsigned integer.
    Uint8,

    /// 16-bit unsigned integer.
    Uint16,

    /// 32-bit unsigned integer.
    Uint32,

    /// 8-bit signed integer.
    Sint8,

    /// 16-bit signed integer.
    Sint16,

    /// 32-bit signed integer.
    Sint32,

    /// 8-bit unsigned normalized integer.
    Unorm8,

    /// 16-bit unsigned normalized integer.
    Unorm16,

    /// 32-bit unsigned normalized integer.
    Unorm32,

    /// 8-bit signed normalized integer.
    Snorm8,

    /// 16-bit signed normalized integer.
    Snorm16,

    /// 32-bit signed normalized integer.
    Snorm32,

    /// 16-bit floating-point number.
    Float16,

    /// 32-bit floating-point number.
    Float32,

    /// 8-bit unsigned integer pair.
    Uint8x2,

    /// 16-bit unsigned integer pair.
    Uint16x2,

    /// 32-bit unsigned integer pair.
    Uint32x2,

    /// 8-bit signed integer pair.
    Sint8x2,

    /// 16-bit signed integer pair.
    Sint16x2,

    /// 32-bit signed integer pair.
    Sint32x2,

    /// 8-bit unsigned normalized integer pair.
    Unorm8x2,

    /// 16-bit unsigned normalized integer pair.
    Unorm16x2,

    /// 32-bit unsigned normalized integer pair.
    Unorm32x2,

    /// 8-bit signed normalized integer pair.
    Snorm8x2,

    /// 16-bit signed normalized integer pair.
    Snorm16x2,

    /// 32-bit signed normalized integer pair.
    Snorm32x2,

    /// 16-bit floating-point number pair.
    Float16x2,

    /// 32-bit floating-point number pair.
    Float32x2,

    /// 8-bit unsigned integer triple.
    Uint8x3,

    /// 16-bit unsigned integer triple.
    Uint16x3,

    /// 32-bit unsigned integer triple.
    Uint32x3,

    /// 8-bit signed integer triple.
    Sint8x3,

    /// 16-bit signed integer triple.
    Sint16x3,

    /// 32-bit signed integer triple.
    Sint32x3,

    /// 8-bit unsigned normalized integer triple.
    Unorm8x3,

    /// 16-bit unsigned normalized integer triple.
    Unorm16x3,

    /// 32-bit unsigned normalized integer triple.
    Unorm32x3,

    /// 8-bit signed normalized integer triple.
    Snorm8x3,

    /// 16-bit signed normalized integer triple.
    Snorm16x3,

    /// 32-bit signed normalized integer triple.
    Snorm32x3,

    /// 16-bit floating-point number triple.
    Float16x3,

    /// 32-bit floating-point number triple.
    Float32x3,

    /// 8-bit unsigned integer quadruple.
    Uint8x4,

    /// 16-bit unsigned integer quadruple.
    Uint16x4,

    /// 32-bit unsigned integer quadruple.
    Uint32x4,

    /// 8-bit signed integer quadruple.
    Sint8x4,

    /// 16-bit signed integer quadruple.
    Sint16x4,

    /// 32-bit signed integer quadruple.
    Sint32x4,

    /// 8-bit unsigned normalized integer quadruple.
    Unorm8x4,

    /// 16-bit unsigned normalized integer quadruple.
    Unorm16x4,

    /// 32-bit unsigned normalized integer quadruple.
    Unorm32x4,

    /// 8-bit signed normalized integer quadruple.
    Snorm8x4,

    /// 16-bit signed normalized integer quadruple.
    Snorm16x4,

    /// 32-bit signed normalized integer quadruple.
    Snorm32x4,

    /// 16-bit floating-point number quadruple.
    Float16x4,

    /// 32-bit floating-point number quadruple.
    Float32x4,
}
