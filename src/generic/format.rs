#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    R8Unorm,
    R8Snorm,
    R8Uint,
    R8Sint,
    R8Srgb,
    R16Unorm,
    R16Snorm,
    R16Uint,
    R16Sint,
    R16Float,
    R32Unorm,
    R32Snorm,
    R32Uint,
    R32Sint,
    R32Float,
    Rg8Unorm,
    Rg8Snorm,
    Rg8Uint,
    Rg8Sint,
    Rg8Srgb,
    Rg16Unorm,
    Rg16Snorm,
    Rg16Uint,
    Rg16Sint,
    Rg16Float,
    Rg32Unorm,
    Rg32Snorm,
    Rg32Uint,
    Rg32Sint,
    Rg32Float,
    Rgb8Unorm,
    Rgb8Snorm,
    Rgb8Uint,
    Rgb8Sint,
    Rgb8Srgb,
    Rgb16Unorm,
    Rgb16Snorm,
    Rgb16Uint,
    Rgb16Sint,
    Rgb16Float,
    Rgb32Unorm,
    Rgb32Snorm,
    Rgb32Uint,
    Rgb32Sint,
    Rgb32Float,
    Rgba8Unorm,
    Rgba8Snorm,
    Rgba8Uint,
    Rgba8Sint,
    Rgba8Srgb,
    Rgba16Unorm,
    Rgba16Snorm,
    Rgba16Uint,
    Rgba16Sint,
    Rgba16Float,
    Rgba32Unorm,
    Rgba32Snorm,
    Rgba32Uint,
    Rgba32Sint,
    Rgba32Float,
    Bgr8Unorm,
    Bgr8Snorm,
    Bgr8Uint,
    Bgr8Sint,
    Bgr8Srgb,
    Bgra8Unorm,
    Bgra8Snorm,
    Bgra8Uint,
    Bgra8Sint,
    Bgra8Srgb,
    D16Unorm,
    D32Float,
    S8Uint,
    D16UnormS8Uint,
    D24UnormS8Uint,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VertexFormat {
    Uint8,
    Uint16,
    Uint32,
    Sint8,
    Sint16,
    Sint32,
    Unorm8,
    Unorm16,
    Unorm32,
    Snorm8,
    Snorm16,
    Snorm32,
    Float16,
    Float32,
    Uint8x2,
    Uint16x2,
    Uint32x2,
    Sint8x2,
    Sint16x2,
    Sint32x2,
    Unorm8x2,
    Unorm16x2,
    Unorm32x2,
    Snorm8x2,
    Snorm16x2,
    Snorm32x2,
    Float16x2,
    Float32x2,
    Uint8x3,
    Uint16x3,
    Uint32x3,
    Sint8x3,
    Sint16x3,
    Sint32x3,
    Unorm8x3,
    Unorm16x3,
    Unorm32x3,
    Snorm8x3,
    Snorm16x3,
    Snorm32x3,
    Float16x3,
    Float32x3,
    Uint8x4,
    Uint16x4,
    Uint32x4,
    Sint8x4,
    Sint16x4,
    Sint32x4,
    Unorm8x4,
    Unorm16x4,
    Unorm32x4,
    Snorm8x4,
    Snorm16x4,
    Snorm32x4,
    Float16x4,
    Float32x4,
}
