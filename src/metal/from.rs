use metal::{MTLSamplerAddressMode, MTLSamplerMinMagFilter, MTLSamplerMipFilter};

use crate::generic::{
    AddressMode, BlendFactor, BlendOp, CompareFunction, Filter, ImageUsage, MipMapMode,
    PixelFormat, PrimitiveTopology, VertexFormat, WriteMask,
};

pub trait FromMetal<T> {
    fn from_metal(t: T) -> Self;
}

pub trait MetalInto<T> {
    fn metal_into(self) -> T;
}

impl<T, U> MetalInto<U> for T
where
    U: FromMetal<T>,
{
    #[inline(always)]
    fn metal_into(self) -> U {
        U::from_metal(self)
    }
}

pub trait MetalFrom<T> {
    fn metal_from(t: T) -> Self;
}

pub trait IntoMetal<T> {
    fn into_metal(self) -> T;
}

impl<T, U> IntoMetal<U> for T
where
    U: MetalFrom<T>,
{
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn into_metal(self) -> U {
        U::metal_from(self)
    }
}

pub trait TryFromMetal<T>: Sized {
    fn try_from_metal(t: T) -> Option<Self>;

    #[cfg_attr(feature = "inline-more", inline(always))]
    fn expect_from_metal(t: T) -> Self {
        Self::try_from_metal(t).expect("Failed to convert from metal")
    }
}

pub trait TryMetalInto<T>: Sized {
    fn try_metal_into(self) -> Option<T>;

    #[cfg_attr(feature = "inline-more", inline(always))]
    fn expect_metal_into(self) -> T {
        self.try_metal_into().expect("Failed to convert from metal")
    }
}

impl<T, U> TryMetalInto<U> for T
where
    U: TryFromMetal<T>,
{
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn try_metal_into(self) -> Option<U> {
        U::try_from_metal(self)
    }

    #[cfg_attr(feature = "inline-more", inline(always))]
    fn expect_metal_into(self) -> U {
        U::expect_from_metal(self)
    }
}

pub trait TryMetalFrom<T>: Sized {
    fn try_metal_from(t: T) -> Option<Self>;

    #[cfg_attr(feature = "inline-more", inline(always))]
    fn expect_metal_from(t: T) -> Self {
        Self::try_metal_from(t).expect("Failed to convert to metal")
    }
}

pub trait TryIntoMetal<T>: Sized {
    fn try_into_metal(self) -> Option<T>;

    #[cfg_attr(feature = "inline-more", inline(always))]
    fn expect_into_metal(self) -> T {
        self.try_into_metal().expect("Failed to convert to metal")
    }
}

impl<T, U> TryIntoMetal<U> for T
where
    U: TryMetalFrom<T>,
{
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn try_into_metal(self) -> Option<U> {
        U::try_metal_from(self)
    }

    #[cfg_attr(feature = "inline-more", inline(always))]
    fn expect_into_metal(self) -> U {
        U::expect_metal_from(self)
    }
}

impl TryMetalFrom<PixelFormat> for metal::MTLPixelFormat {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn try_metal_from(t: PixelFormat) -> Option<Self> {
        Some(match t {
            PixelFormat::R8Unorm => metal::MTLPixelFormat::R8Unorm,
            PixelFormat::R8Srgb => metal::MTLPixelFormat::R8Unorm_sRGB,
            PixelFormat::R8Snorm => metal::MTLPixelFormat::R8Snorm,
            PixelFormat::R8Uint => metal::MTLPixelFormat::R8Uint,
            PixelFormat::R8Sint => metal::MTLPixelFormat::R8Sint,
            PixelFormat::R16Unorm => metal::MTLPixelFormat::R16Unorm,
            PixelFormat::R16Snorm => metal::MTLPixelFormat::R16Snorm,
            PixelFormat::R16Uint => metal::MTLPixelFormat::R16Uint,
            PixelFormat::R16Sint => metal::MTLPixelFormat::R16Sint,
            PixelFormat::R16Float => metal::MTLPixelFormat::R16Float,
            // PixelFormat::R32Unorm => metal::MTLPixelFormat::R32Unorm,
            // PixelFormat::R32Snorm => metal::MTLPixelFormat::R32Snorm,
            PixelFormat::R32Uint => metal::MTLPixelFormat::R32Uint,
            PixelFormat::R32Sint => metal::MTLPixelFormat::R32Sint,
            PixelFormat::R32Float => metal::MTLPixelFormat::R32Float,
            PixelFormat::Rg8Unorm => metal::MTLPixelFormat::RG8Unorm,
            PixelFormat::Rg8Srgb => metal::MTLPixelFormat::RG8Unorm_sRGB,
            PixelFormat::Rg8Snorm => metal::MTLPixelFormat::RG8Snorm,
            PixelFormat::Rg8Uint => metal::MTLPixelFormat::RG8Uint,
            PixelFormat::Rg8Sint => metal::MTLPixelFormat::RG8Sint,
            PixelFormat::Rg16Unorm => metal::MTLPixelFormat::RG16Unorm,
            PixelFormat::Rg16Snorm => metal::MTLPixelFormat::RG16Snorm,
            PixelFormat::Rg16Uint => metal::MTLPixelFormat::RG16Uint,
            PixelFormat::Rg16Sint => metal::MTLPixelFormat::RG16Sint,
            PixelFormat::Rg16Float => metal::MTLPixelFormat::RG16Float,
            // PixelFormat::Rg32Unorm => metal::MTLPixelFormat::RG32Unorm,
            // PixelFormat::Rg32Snorm => metal::MTLPixelFormat::RG32Snorm,
            PixelFormat::Rg32Uint => metal::MTLPixelFormat::RG32Uint,
            PixelFormat::Rg32Sint => metal::MTLPixelFormat::RG32Sint,
            PixelFormat::Rg32Float => metal::MTLPixelFormat::RG32Float,
            // PixelFormat::Rgb8Unorm => metal::MTLPixelFormat::RGB8Unorm,
            // PixelFormat::Rgb8Srgb => metal::MTLPixelFormat::RGB8Unorm_sRGB,
            // PixelFormat::Rgb8Snorm => metal::MTLPixelFormat::RGB8Snorm,
            // PixelFormat::Rgb8Uint => metal::MTLPixelFormat::RGB8Uint,
            // PixelFormat::Rgb8Sint => metal::MTLPixelFormat::RGB8Sint,
            // PixelFormat::Rgb16Unorm => metal::MTLPixelFormat::RGB16Unorm,
            // PixelFormat::Rgb16Snorm => metal::MTLPixelFormat::RGB16Snorm,
            // PixelFormat::Rgb16Uint => metal::MTLPixelFormat::RGB16Uint,
            // PixelFormat::Rgb16Sint => metal::MTLPixelFormat::RGB16Sint,
            // PixelFormat::Rgb16Float => metal::MTLPixelFormat::RGB16Float,
            // PixelFormat::Rgb32Unorm => metal::MTLPixelFormat::RGB32Unorm,
            // PixelFormat::Rgb32Snorm => metal::MTLPixelFormat::RGB32Snorm,
            // PixelFormat::Rgb32Uint => metal::MTLPixelFormat::RGB32Uint,
            // PixelFormat::Rgb32Sint => metal::MTLPixelFormat::RGB32Sint,
            // PixelFormat::Rgb32Float => metal::MTLPixelFormat::RGB32Float,
            PixelFormat::Rgba8Unorm => metal::MTLPixelFormat::RGBA8Unorm,
            PixelFormat::Rgba8Srgb => metal::MTLPixelFormat::RGBA8Unorm_sRGB,
            PixelFormat::Rgba8Snorm => metal::MTLPixelFormat::RGBA8Snorm,
            PixelFormat::Rgba8Uint => metal::MTLPixelFormat::RGBA8Uint,
            PixelFormat::Rgba8Sint => metal::MTLPixelFormat::RGBA8Sint,
            PixelFormat::Rgba16Unorm => metal::MTLPixelFormat::RGBA16Unorm,
            PixelFormat::Rgba16Snorm => metal::MTLPixelFormat::RGBA16Snorm,
            PixelFormat::Rgba16Uint => metal::MTLPixelFormat::RGBA16Uint,
            PixelFormat::Rgba16Sint => metal::MTLPixelFormat::RGBA16Sint,
            PixelFormat::Rgba16Float => metal::MTLPixelFormat::RGBA16Float,
            // PixelFormat::Rgba32Unorm => metal::MTLPixelFormat::RGBA32Unorm,
            // PixelFormat::Rgba32Snorm => metal::MTLPixelFormat::RGBA32Snorm,
            PixelFormat::Rgba32Uint => metal::MTLPixelFormat::RGBA32Uint,
            PixelFormat::Rgba32Sint => metal::MTLPixelFormat::RGBA32Sint,
            PixelFormat::Rgba32Float => metal::MTLPixelFormat::RGBA32Float,
            // PixelFormat::Bgr8Unorm => metal::MTLPixelFormat::BGR8Unorm,
            // PixelFormat::Bgr8Srgb => metal::MTLPixelFormat::BGR8Unorm_sRGB,
            // PixelFormat::Bgr8Snorm => metal::MTLPixelFormat::BGR8Snorm,
            // PixelFormat::Bgr8Uint => metal::MTLPixelFormat::BGR8Uint,
            // PixelFormat::Bgr8Sint => metal::MTLPixelFormat::BGR8Sint,
            PixelFormat::Bgra8Unorm => metal::MTLPixelFormat::BGRA8Unorm,
            PixelFormat::Bgra8Srgb => metal::MTLPixelFormat::BGRA8Unorm_sRGB,
            // PixelFormat::Bgra8Snorm => metal::MTLPixelFormat::BGRA8Snorm,
            // PixelFormat::Bgra8Uint => metal::MTLPixelFormat::BGRA8Uint,
            // PixelFormat::Bgra8Sint => metal::MTLPixelFormat::BGRA8Sint,
            PixelFormat::D16Unorm => metal::MTLPixelFormat::Depth16Unorm,
            PixelFormat::D32Float => metal::MTLPixelFormat::Depth32Float,
            PixelFormat::S8Uint => metal::MTLPixelFormat::Stencil8,
            // PixelFormat::D16UnormS8Uint => metal::MTLPixelFormat::Depth16Unorm_Stencil8,
            PixelFormat::D24UnormS8Uint => metal::MTLPixelFormat::Depth24Unorm_Stencil8,
            PixelFormat::D32FloatS8Uint => metal::MTLPixelFormat::Depth32Float_Stencil8,
            _ => return None,
        })
    }
}

impl TryFromMetal<metal::MTLPixelFormat> for PixelFormat {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn try_from_metal(t: metal::MTLPixelFormat) -> Option<Self> {
        Some(match t {
            metal::MTLPixelFormat::R8Unorm => PixelFormat::R8Unorm,
            metal::MTLPixelFormat::R8Unorm_sRGB => PixelFormat::R8Srgb,
            metal::MTLPixelFormat::R8Snorm => PixelFormat::R8Snorm,
            metal::MTLPixelFormat::R8Uint => PixelFormat::R8Uint,
            metal::MTLPixelFormat::R8Sint => PixelFormat::R8Sint,
            metal::MTLPixelFormat::R16Unorm => PixelFormat::R16Unorm,
            metal::MTLPixelFormat::R16Snorm => PixelFormat::R16Snorm,
            metal::MTLPixelFormat::R16Uint => PixelFormat::R16Uint,
            metal::MTLPixelFormat::R16Sint => PixelFormat::R16Sint,
            metal::MTLPixelFormat::R16Float => PixelFormat::R16Float,
            // metal::MTLPixelFormat::R32Unorm => PixelFormat::R32Unorm,
            // metal::MTLPixelFormat::R32Snorm => PixelFormat::R32Snorm,
            metal::MTLPixelFormat::R32Uint => PixelFormat::R32Uint,
            metal::MTLPixelFormat::R32Sint => PixelFormat::R32Sint,
            metal::MTLPixelFormat::R32Float => PixelFormat::R32Float,
            metal::MTLPixelFormat::RG8Unorm => PixelFormat::Rg8Unorm,
            metal::MTLPixelFormat::RG8Unorm_sRGB => PixelFormat::Rg8Srgb,
            metal::MTLPixelFormat::RG8Snorm => PixelFormat::Rg8Snorm,
            metal::MTLPixelFormat::RG8Uint => PixelFormat::Rg8Uint,
            metal::MTLPixelFormat::RG8Sint => PixelFormat::Rg8Sint,
            metal::MTLPixelFormat::RG16Unorm => PixelFormat::Rg16Unorm,
            metal::MTLPixelFormat::RG16Snorm => PixelFormat::Rg16Snorm,
            metal::MTLPixelFormat::RG16Uint => PixelFormat::Rg16Uint,
            metal::MTLPixelFormat::RG16Sint => PixelFormat::Rg16Sint,
            metal::MTLPixelFormat::RG16Float => PixelFormat::Rg16Float,
            // metal::MTLPixelFormat::RG32Unorm => PixelFormat::Rg32Unorm,
            // metal::MTLPixelFormat::RG32Snorm => PixelFormat::Rg32Snorm,
            metal::MTLPixelFormat::RG32Uint => PixelFormat::Rg32Uint,
            metal::MTLPixelFormat::RG32Sint => PixelFormat::Rg32Sint,
            metal::MTLPixelFormat::RG32Float => PixelFormat::Rg32Float,
            // metal::MTLPixelFormat::RGB8Unorm => PixelFormat::Rgb8Unorm,
            // metal::MTLPixelFormat::RGB8Snorm => PixelFormat::Rgb8Snorm,
            // metal::MTLPixelFormat::RGB8Uint => PixelFormat::Rgb8Uint,
            // metal::MTLPixelFormat::RGB8Sint => PixelFormat::Rgb8Sint,
            // metal::MTLPixelFormat::RGB16Unorm => PixelFormat::Rgb16Unorm,
            // metal::MTLPixelFormat::RGB16Snorm => PixelFormat::Rgb16Snorm,
            // metal::MTLPixelFormat::RGB16Uint => PixelFormat::Rgb16Uint,
            // metal::MTLPixelFormat::RGB16Sint => PixelFormat::Rgb16Sint,
            // metal::MTLPixelFormat::RGB16Float => PixelFormat::Rgb16Float,
            // metal::MTLPixelFormat::RGB32Unorm => PixelFormat::Rgb32Unorm,
            // metal::MTLPixelFormat::RGB32Snorm => PixelFormat::Rgb32Snorm,
            // metal::MTLPixelFormat::RGB32Uint => PixelFormat::Rgb32Uint,
            // metal::MTLPixelFormat::RGB32Sint => PixelFormat::Rgb32Sint,
            // metal::MTLPixelFormat::RGB32Float => PixelFormat::Rgb32Float,
            metal::MTLPixelFormat::RGBA8Unorm => PixelFormat::Rgba8Unorm,
            metal::MTLPixelFormat::RGBA8Unorm_sRGB => PixelFormat::Rgba8Srgb,
            metal::MTLPixelFormat::RGBA8Snorm => PixelFormat::Rgba8Snorm,
            metal::MTLPixelFormat::RGBA8Uint => PixelFormat::Rgba8Uint,
            metal::MTLPixelFormat::RGBA8Sint => PixelFormat::Rgba8Sint,
            metal::MTLPixelFormat::RGBA16Unorm => PixelFormat::Rgba16Unorm,
            metal::MTLPixelFormat::RGBA16Snorm => PixelFormat::Rgba16Snorm,
            metal::MTLPixelFormat::RGBA16Uint => PixelFormat::Rgba16Uint,
            metal::MTLPixelFormat::RGBA16Sint => PixelFormat::Rgba16Sint,
            metal::MTLPixelFormat::RGBA16Float => PixelFormat::Rgba16Float,
            // metal::MTLPixelFormat::RGBA32Unorm => PixelFormat::Rgba32Unorm,
            // metal::MTLPixelFormat::RGBA32Snorm => PixelFormat::Rgba32Snorm,
            metal::MTLPixelFormat::RGBA32Uint => PixelFormat::Rgba32Uint,
            metal::MTLPixelFormat::RGBA32Sint => PixelFormat::Rgba32Sint,
            metal::MTLPixelFormat::RGBA32Float => PixelFormat::Rgba32Float,
            // metal::MTLPixelFormat::BGR8Unorm => PixelFormat::Bgr8Unorm,
            // metal::MTLPixelFormat::BGR8Unorm_sRGB => PixelFormat::Bgr8Srgb,
            // metal::MTLPixelFormat::BGR8Snorm => PixelFormat::Bgr8Snorm,
            // metal::MTLPixelFormat::BGR8Uint => PixelFormat::Bgr8Uint,
            // metal::MTLPixelFormat::BGR8Sint => PixelFormat::Bgr8Sint,
            metal::MTLPixelFormat::BGRA8Unorm => PixelFormat::Bgra8Unorm,
            metal::MTLPixelFormat::BGRA8Unorm_sRGB => PixelFormat::Bgra8Srgb,
            // metal::MTLPixelFormat::BGRA8Snorm => PixelFormat::Bgra8Snorm,
            // metal::MTLPixelFormat::BGRA8Uint => PixelFormat::Bgra8Uint,
            // metal::MTLPixelFormat::BGRA8Sint => PixelFormat::Bgra8Sint,
            metal::MTLPixelFormat::Depth16Unorm => PixelFormat::D16Unorm,
            metal::MTLPixelFormat::Depth32Float => PixelFormat::D32Float,
            metal::MTLPixelFormat::Stencil8 => PixelFormat::S8Uint,
            // metal::MTLPixelFormat::Depth16Unorm_Stencil8 => PixelFormat::D16UnormS8Uint,
            metal::MTLPixelFormat::Depth24Unorm_Stencil8 => PixelFormat::D24UnormS8Uint,
            metal::MTLPixelFormat::Depth32Float_Stencil8 => PixelFormat::D32FloatS8Uint,
            _ => return None,
        })
    }
}

impl TryMetalFrom<VertexFormat> for metal::MTLVertexFormat {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn try_metal_from(format: VertexFormat) -> Option<Self> {
        Some(match format {
            VertexFormat::Uint8 => metal::MTLVertexFormat::UChar,
            VertexFormat::Uint16 => metal::MTLVertexFormat::UShort,
            VertexFormat::Uint32 => metal::MTLVertexFormat::UInt,
            VertexFormat::Sint8 => metal::MTLVertexFormat::Char,
            VertexFormat::Sint16 => metal::MTLVertexFormat::Short,
            VertexFormat::Sint32 => metal::MTLVertexFormat::Int,
            VertexFormat::Unorm8 => metal::MTLVertexFormat::UCharNormalized,
            VertexFormat::Unorm16 => metal::MTLVertexFormat::UShortNormalized,
            // VertexFormat::Unorm32 => metal::MTLVertexFormat::UIntNormalized,
            VertexFormat::Snorm8 => metal::MTLVertexFormat::CharNormalized,
            VertexFormat::Snorm16 => metal::MTLVertexFormat::ShortNormalized,
            // VertexFormat::Snorm32 => metal::MTLVertexFormat::IntNormalized,
            VertexFormat::Float16 => metal::MTLVertexFormat::Half,
            VertexFormat::Float32 => metal::MTLVertexFormat::Float,
            VertexFormat::Uint8x2 => metal::MTLVertexFormat::UChar2,
            VertexFormat::Uint16x2 => metal::MTLVertexFormat::UShort2,
            VertexFormat::Uint32x2 => metal::MTLVertexFormat::UInt2,
            VertexFormat::Sint8x2 => metal::MTLVertexFormat::Char2,
            VertexFormat::Sint16x2 => metal::MTLVertexFormat::Short2,
            VertexFormat::Sint32x2 => metal::MTLVertexFormat::Int2,
            VertexFormat::Unorm8x2 => metal::MTLVertexFormat::UChar2Normalized,
            VertexFormat::Unorm16x2 => metal::MTLVertexFormat::UShort2Normalized,
            // VertexFormat::Unorm32x2 => metal::MTLVertexFormat::UInt2Normalized,
            VertexFormat::Snorm8x2 => metal::MTLVertexFormat::Char2Normalized,
            VertexFormat::Snorm16x2 => metal::MTLVertexFormat::Short2Normalized,
            // VertexFormat::Snorm32x2 => metal::MTLVertexFormat::Int2Normalized,
            VertexFormat::Float16x2 => metal::MTLVertexFormat::Half2,
            VertexFormat::Float32x2 => metal::MTLVertexFormat::Float2,
            VertexFormat::Uint8x3 => metal::MTLVertexFormat::UChar3,
            VertexFormat::Uint16x3 => metal::MTLVertexFormat::UShort3,
            VertexFormat::Uint32x3 => metal::MTLVertexFormat::UInt3,
            VertexFormat::Sint8x3 => metal::MTLVertexFormat::Char3,
            VertexFormat::Sint16x3 => metal::MTLVertexFormat::Short3,
            VertexFormat::Sint32x3 => metal::MTLVertexFormat::Int3,
            VertexFormat::Unorm8x3 => metal::MTLVertexFormat::UChar3Normalized,
            VertexFormat::Unorm16x3 => metal::MTLVertexFormat::UShort3Normalized,
            // VertexFormat::Unorm32x3 => metal::MTLVertexFormat::UInt3Normalized,
            VertexFormat::Snorm8x3 => metal::MTLVertexFormat::Char3Normalized,
            VertexFormat::Snorm16x3 => metal::MTLVertexFormat::Short3Normalized,
            // VertexFormat::Snorm32x3 => metal::MTLVertexFormat::Int3Normalized,
            VertexFormat::Float16x3 => metal::MTLVertexFormat::Half3,
            VertexFormat::Float32x3 => metal::MTLVertexFormat::Float3,
            VertexFormat::Uint8x4 => metal::MTLVertexFormat::UChar4,
            VertexFormat::Uint16x4 => metal::MTLVertexFormat::UShort4,
            VertexFormat::Uint32x4 => metal::MTLVertexFormat::UInt4,
            VertexFormat::Sint8x4 => metal::MTLVertexFormat::Char4,
            VertexFormat::Sint16x4 => metal::MTLVertexFormat::Short4,
            VertexFormat::Sint32x4 => metal::MTLVertexFormat::Int4,
            VertexFormat::Unorm8x4 => metal::MTLVertexFormat::UChar4Normalized,
            VertexFormat::Unorm16x4 => metal::MTLVertexFormat::UShort4Normalized,
            // VertexFormat::Unorm32x4 => metal::MTLVertexFormat::UInt4Normalized,
            VertexFormat::Snorm8x4 => metal::MTLVertexFormat::Char4Normalized,
            VertexFormat::Snorm16x4 => metal::MTLVertexFormat::Short4Normalized,
            // VertexFormat::Snorm32x4 => metal::MTLVertexFormat::Int4Normalized,
            VertexFormat::Float16x4 => metal::MTLVertexFormat::Half4,
            VertexFormat::Float32x4 => metal::MTLVertexFormat::Float4,
            _ => return None,
        })
    }
}

impl TryMetalFrom<VertexFormat> for metal::MTLAttributeFormat {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn try_metal_from(format: VertexFormat) -> Option<Self> {
        Some(match format {
            VertexFormat::Uint8 => metal::MTLAttributeFormat::UChar,
            VertexFormat::Uint16 => metal::MTLAttributeFormat::UShort,
            VertexFormat::Uint32 => metal::MTLAttributeFormat::UInt,
            VertexFormat::Sint8 => metal::MTLAttributeFormat::Char,
            VertexFormat::Sint16 => metal::MTLAttributeFormat::Short,
            VertexFormat::Sint32 => metal::MTLAttributeFormat::Int,
            VertexFormat::Unorm8 => metal::MTLAttributeFormat::UCharNormalized,
            VertexFormat::Unorm16 => metal::MTLAttributeFormat::UShortNormalized,
            // VertexFormat::Unorm32 => metal::MTLAttributeFormat::UIntNormalized,
            VertexFormat::Snorm8 => metal::MTLAttributeFormat::CharNormalized,
            VertexFormat::Snorm16 => metal::MTLAttributeFormat::ShortNormalized,
            // VertexFormat::Snorm32 => metal::MTLAttributeFormat::IntNormalized,
            VertexFormat::Float16 => metal::MTLAttributeFormat::Half,
            VertexFormat::Float32 => metal::MTLAttributeFormat::Float,
            VertexFormat::Uint8x2 => metal::MTLAttributeFormat::UChar2,
            VertexFormat::Uint16x2 => metal::MTLAttributeFormat::UShort2,
            VertexFormat::Uint32x2 => metal::MTLAttributeFormat::UInt2,
            VertexFormat::Sint8x2 => metal::MTLAttributeFormat::Char2,
            VertexFormat::Sint16x2 => metal::MTLAttributeFormat::Short2,
            VertexFormat::Sint32x2 => metal::MTLAttributeFormat::Int2,
            VertexFormat::Unorm8x2 => metal::MTLAttributeFormat::UChar2Normalized,
            VertexFormat::Unorm16x2 => metal::MTLAttributeFormat::UShort2Normalized,
            // VertexFormat::Unorm32x2 => metal::MTLAttributeFormat::UInt2Normalized,
            VertexFormat::Snorm8x2 => metal::MTLAttributeFormat::Char2Normalized,
            VertexFormat::Snorm16x2 => metal::MTLAttributeFormat::Short2Normalized,
            // VertexFormat::Snorm32x2 => metal::MTLAttributeFormat::Int2Normalized,
            VertexFormat::Float16x2 => metal::MTLAttributeFormat::Half2,
            VertexFormat::Float32x2 => metal::MTLAttributeFormat::Float2,
            VertexFormat::Uint8x3 => metal::MTLAttributeFormat::UChar3,
            VertexFormat::Uint16x3 => metal::MTLAttributeFormat::UShort3,
            VertexFormat::Uint32x3 => metal::MTLAttributeFormat::UInt3,
            VertexFormat::Sint8x3 => metal::MTLAttributeFormat::Char3,
            VertexFormat::Sint16x3 => metal::MTLAttributeFormat::Short3,
            VertexFormat::Sint32x3 => metal::MTLAttributeFormat::Int3,
            VertexFormat::Unorm8x3 => metal::MTLAttributeFormat::UChar3Normalized,
            VertexFormat::Unorm16x3 => metal::MTLAttributeFormat::UShort3Normalized,
            // VertexFormat::Unorm32x3 => metal::MTLAttributeFormat::UInt3Normalized,
            VertexFormat::Snorm8x3 => metal::MTLAttributeFormat::Char3Normalized,
            VertexFormat::Snorm16x3 => metal::MTLAttributeFormat::Short3Normalized,
            // VertexFormat::Snorm32x3 => metal::MTLAttributeFormat::Int3Normalized,
            VertexFormat::Float16x3 => metal::MTLAttributeFormat::Half3,
            VertexFormat::Float32x3 => metal::MTLAttributeFormat::Float3,
            VertexFormat::Uint8x4 => metal::MTLAttributeFormat::UChar4,
            VertexFormat::Uint16x4 => metal::MTLAttributeFormat::UShort4,
            VertexFormat::Uint32x4 => metal::MTLAttributeFormat::UInt4,
            VertexFormat::Sint8x4 => metal::MTLAttributeFormat::Char4,
            VertexFormat::Sint16x4 => metal::MTLAttributeFormat::Short4,
            VertexFormat::Sint32x4 => metal::MTLAttributeFormat::Int4,
            VertexFormat::Unorm8x4 => metal::MTLAttributeFormat::UChar4Normalized,
            VertexFormat::Unorm16x4 => metal::MTLAttributeFormat::UShort4Normalized,
            // VertexFormat::Unorm32x4 => metal::MTLAttributeFormat::UInt4Normalized,
            VertexFormat::Snorm8x4 => metal::MTLAttributeFormat::Char4Normalized,
            VertexFormat::Snorm16x4 => metal::MTLAttributeFormat::Short4Normalized,
            // VertexFormat::Snorm32x4 => metal::MTLAttributeFormat::Int4Normalized,
            VertexFormat::Float16x4 => metal::MTLAttributeFormat::Half4,
            VertexFormat::Float32x4 => metal::MTLAttributeFormat::Float4,
            _ => return None,
        })
    }
}

impl MetalFrom<PrimitiveTopology> for metal::MTLPrimitiveTopologyClass {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn metal_from(t: PrimitiveTopology) -> Self {
        match t {
            PrimitiveTopology::Point => metal::MTLPrimitiveTopologyClass::Point,
            PrimitiveTopology::Line => metal::MTLPrimitiveTopologyClass::Line,
            PrimitiveTopology::Triangle => metal::MTLPrimitiveTopologyClass::Triangle,
        }
    }
}

impl MetalFrom<PrimitiveTopology> for metal::MTLPrimitiveType {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn metal_from(t: PrimitiveTopology) -> Self {
        match t {
            PrimitiveTopology::Point => metal::MTLPrimitiveType::Point,
            PrimitiveTopology::Line => metal::MTLPrimitiveType::Line,
            PrimitiveTopology::Triangle => metal::MTLPrimitiveType::Triangle,
        }
    }
}

impl MetalFrom<BlendOp> for metal::MTLBlendOperation {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn metal_from(t: BlendOp) -> Self {
        match t {
            BlendOp::Add => metal::MTLBlendOperation::Add,
            BlendOp::Subtract => metal::MTLBlendOperation::Subtract,
            BlendOp::ReverseSubtract => metal::MTLBlendOperation::ReverseSubtract,
            BlendOp::Min => metal::MTLBlendOperation::Min,
            BlendOp::Max => metal::MTLBlendOperation::Max,
        }
    }
}

impl MetalFrom<BlendFactor> for metal::MTLBlendFactor {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn metal_from(t: BlendFactor) -> Self {
        match t {
            BlendFactor::Zero => metal::MTLBlendFactor::Zero,
            BlendFactor::One => metal::MTLBlendFactor::One,
            BlendFactor::SrcColor => metal::MTLBlendFactor::SourceColor,
            BlendFactor::OneMinusSrcColor => metal::MTLBlendFactor::OneMinusSourceColor,
            BlendFactor::SrcAlpha => metal::MTLBlendFactor::SourceAlpha,
            BlendFactor::OneMinusSrcAlpha => metal::MTLBlendFactor::OneMinusSourceAlpha,
            BlendFactor::DstColor => metal::MTLBlendFactor::DestinationColor,
            BlendFactor::OneMinusDstColor => metal::MTLBlendFactor::OneMinusDestinationColor,
            BlendFactor::DstAlpha => metal::MTLBlendFactor::DestinationAlpha,
            BlendFactor::OneMinusDstAlpha => metal::MTLBlendFactor::OneMinusDestinationAlpha,
            BlendFactor::SrcAlphaSaturated => metal::MTLBlendFactor::SourceAlphaSaturated,
            // BlendFactor::BlendColor => metal::MTLBlendFactor::BlendColor,
            // BlendFactor::OneMinusBlendColor => metal::MTLBlendFactor::OneMinusBlendColor,
        }
    }
}

impl MetalFrom<CompareFunction> for metal::MTLCompareFunction {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn metal_from(t: CompareFunction) -> Self {
        match t {
            CompareFunction::Never => metal::MTLCompareFunction::Never,
            CompareFunction::Less => metal::MTLCompareFunction::Less,
            CompareFunction::Equal => metal::MTLCompareFunction::Equal,
            CompareFunction::LessEqual => metal::MTLCompareFunction::LessEqual,
            CompareFunction::Greater => metal::MTLCompareFunction::Greater,
            CompareFunction::NotEqual => metal::MTLCompareFunction::NotEqual,
            CompareFunction::GreaterEqual => metal::MTLCompareFunction::GreaterEqual,
            CompareFunction::Always => metal::MTLCompareFunction::Always,
        }
    }
}

impl MetalFrom<WriteMask> for metal::MTLColorWriteMask {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn metal_from(t: WriteMask) -> Self {
        let mut mask = metal::MTLColorWriteMask::empty();
        if t.contains(WriteMask::RED) {
            mask |= metal::MTLColorWriteMask::Red;
        }
        if t.contains(WriteMask::GREEN) {
            mask |= metal::MTLColorWriteMask::Green;
        }
        if t.contains(WriteMask::BLUE) {
            mask |= metal::MTLColorWriteMask::Blue;
        }
        if t.contains(WriteMask::ALPHA) {
            mask |= metal::MTLColorWriteMask::Alpha;
        }
        mask
    }
}

impl MetalFrom<ImageUsage> for metal::MTLTextureUsage {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn metal_from(t: ImageUsage) -> Self {
        let mut mask = metal::MTLTextureUsage::empty();
        if t.contains(ImageUsage::SAMPLED) {
            mask |= metal::MTLTextureUsage::ShaderRead;
        }
        if t.contains(ImageUsage::STORAGE) {
            mask |= metal::MTLTextureUsage::ShaderWrite;
        }
        if t.contains(ImageUsage::TARGET) {
            mask |= metal::MTLTextureUsage::RenderTarget;
        }
        // if t.contains(ImageUsage::TRANSFER_SRC) {
        //     mask |= metal::MTLTextureUsage::Unknown;
        // }
        // if t.contains(ImageUsage::TRANSFER_DST) {
        //     mask |= metal::MTLTextureUsage::Unknown;
        // }
        mask
    }
}

impl FromMetal<metal::MTLTextureUsage> for ImageUsage {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn from_metal(t: metal::MTLTextureUsage) -> Self {
        let mut mask = ImageUsage::empty();
        if t.contains(metal::MTLTextureUsage::ShaderRead) {
            mask |= ImageUsage::SAMPLED;
        }
        if t.contains(metal::MTLTextureUsage::ShaderWrite) {
            mask |= ImageUsage::STORAGE;
        }
        if t.contains(metal::MTLTextureUsage::RenderTarget) {
            mask |= ImageUsage::TARGET;
        }

        mask |= ImageUsage::TRANSFER_SRC;
        mask |= ImageUsage::TRANSFER_DST;

        mask
    }
}

// impl MetalFrom<RenderStages> for metal::MTLRenderStages {
//     fn metal_from(stages: RenderStages) -> Self {
//         let mut mask = metal::MTLRenderStages::empty();
//         if stages.intersects(
//             RenderStages::DRAW_INDIRECT | RenderStages::VERTEX_INPUT | RenderStages::VERTEX_SHADER,
//         ) {
//             mask |= metal::MTLRenderStages::Vertex;
//         }
//         if stages.contains(
//             RenderStages::EARLY_FRAGMENT_TEST
//                 | RenderStages::FRAGMENT_SHADER
//                 | RenderStages::LATE_FRAGMENT_TEST
//                 | RenderStages::COLOR_OUTPUT,
//         ) {
//             mask |= metal::MTLRenderStages::Fragment;
//         }
//         mask
//     }
// }

impl MetalFrom<Filter> for MTLSamplerMinMagFilter {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn metal_from(filter: Filter) -> Self {
        match filter {
            Filter::Nearest => MTLSamplerMinMagFilter::Nearest,
            Filter::Linear => MTLSamplerMinMagFilter::Linear,
        }
    }
}

impl MetalFrom<MipMapMode> for MTLSamplerMipFilter {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn metal_from(mode: MipMapMode) -> Self {
        match mode {
            MipMapMode::Nearest => MTLSamplerMipFilter::Nearest,
            MipMapMode::Linear => MTLSamplerMipFilter::Linear,
        }
    }
}

impl MetalFrom<AddressMode> for MTLSamplerAddressMode {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn metal_from(mode: AddressMode) -> Self {
        match mode {
            AddressMode::ClampToEdge => MTLSamplerAddressMode::ClampToEdge,
            AddressMode::Repeat => MTLSamplerAddressMode::Repeat,
            AddressMode::MirrorRepeat => MTLSamplerAddressMode::MirrorRepeat,
        }
    }
}
