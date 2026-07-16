use objc2_metal::{
    MTLAttributeFormat, MTLBlendFactor, MTLBlendOperation, MTLColorWriteMask, MTLCompareFunction,
    MTLPixelFormat, MTLPrimitiveTopologyClass, MTLPrimitiveType, MTLSamplerAddressMode,
    MTLSamplerMinMagFilter, MTLSamplerMipFilter, MTLTextureUsage, MTLVertexFormat,
};

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
    #[inline(always)]
    fn into_metal(self) -> U {
        U::metal_from(self)
    }
}

pub trait TryFromMetal<T>: Sized {
    fn try_from_metal(t: T) -> Option<Self>;

    #[inline(always)]
    fn expect_from_metal(t: T) -> Self {
        Self::try_from_metal(t).expect("Failed to convert from metal")
    }
}

pub trait TryMetalInto<T>: Sized {
    fn try_metal_into(self) -> Option<T>;

    #[inline(always)]
    fn expect_metal_into(self) -> T {
        self.try_metal_into().expect("Failed to convert from metal")
    }
}

impl<T, U> TryMetalInto<U> for T
where
    U: TryFromMetal<T>,
{
    #[inline(always)]
    fn try_metal_into(self) -> Option<U> {
        U::try_from_metal(self)
    }

    #[inline(always)]
    fn expect_metal_into(self) -> U {
        U::expect_from_metal(self)
    }
}

pub trait TryMetalFrom<T>: Sized {
    fn try_metal_from(t: T) -> Option<Self>;

    #[inline(always)]
    fn expect_metal_from(t: T) -> Self {
        Self::try_metal_from(t).expect("Failed to convert to metal")
    }
}

pub trait TryIntoMetal<T>: Sized {
    fn try_into_metal(self) -> Option<T>;

    #[inline(always)]
    fn expect_into_metal(self) -> T {
        self.try_into_metal().expect("Failed to convert to metal")
    }
}

impl<T, U> TryIntoMetal<U> for T
where
    U: TryMetalFrom<T>,
{
    #[inline(always)]
    fn try_into_metal(self) -> Option<U> {
        U::try_metal_from(self)
    }

    #[inline(always)]
    fn expect_into_metal(self) -> U {
        U::expect_metal_from(self)
    }
}

impl TryMetalFrom<PixelFormat> for MTLPixelFormat {
    #[inline]
    fn try_metal_from(t: PixelFormat) -> Option<Self> {
        Some(match t {
            PixelFormat::R8Unorm => MTLPixelFormat::R8Unorm,
            PixelFormat::R8Srgb => MTLPixelFormat::R8Unorm_sRGB,
            PixelFormat::R8Snorm => MTLPixelFormat::R8Snorm,
            PixelFormat::R8Uint => MTLPixelFormat::R8Uint,
            PixelFormat::R8Sint => MTLPixelFormat::R8Sint,
            PixelFormat::R16Unorm => MTLPixelFormat::R16Unorm,
            PixelFormat::R16Snorm => MTLPixelFormat::R16Snorm,
            PixelFormat::R16Uint => MTLPixelFormat::R16Uint,
            PixelFormat::R16Sint => MTLPixelFormat::R16Sint,
            PixelFormat::R16Float => MTLPixelFormat::R16Float,
            // PixelFormat::R32Unorm => MTLPixelFormat::R32Unorm,
            // PixelFormat::R32Snorm => MTLPixelFormat::R32Snorm,
            PixelFormat::R32Uint => MTLPixelFormat::R32Uint,
            PixelFormat::R32Sint => MTLPixelFormat::R32Sint,
            PixelFormat::R32Float => MTLPixelFormat::R32Float,
            PixelFormat::Rg8Unorm => MTLPixelFormat::RG8Unorm,
            PixelFormat::Rg8Srgb => MTLPixelFormat::RG8Unorm_sRGB,
            PixelFormat::Rg8Snorm => MTLPixelFormat::RG8Snorm,
            PixelFormat::Rg8Uint => MTLPixelFormat::RG8Uint,
            PixelFormat::Rg8Sint => MTLPixelFormat::RG8Sint,
            PixelFormat::Rg16Unorm => MTLPixelFormat::RG16Unorm,
            PixelFormat::Rg16Snorm => MTLPixelFormat::RG16Snorm,
            PixelFormat::Rg16Uint => MTLPixelFormat::RG16Uint,
            PixelFormat::Rg16Sint => MTLPixelFormat::RG16Sint,
            PixelFormat::Rg16Float => MTLPixelFormat::RG16Float,
            // PixelFormat::Rg32Unorm => MTLPixelFormat::RG32Unorm,
            // PixelFormat::Rg32Snorm => MTLPixelFormat::RG32Snorm,
            PixelFormat::Rg32Uint => MTLPixelFormat::RG32Uint,
            PixelFormat::Rg32Sint => MTLPixelFormat::RG32Sint,
            PixelFormat::Rg32Float => MTLPixelFormat::RG32Float,
            // PixelFormat::Rgb8Unorm => MTLPixelFormat::RGB8Unorm,
            // PixelFormat::Rgb8Srgb => MTLPixelFormat::RGB8Unorm_sRGB,
            // PixelFormat::Rgb8Snorm => MTLPixelFormat::RGB8Snorm,
            // PixelFormat::Rgb8Uint => MTLPixelFormat::RGB8Uint,
            // PixelFormat::Rgb8Sint => MTLPixelFormat::RGB8Sint,
            // PixelFormat::Rgb16Unorm => MTLPixelFormat::RGB16Unorm,
            // PixelFormat::Rgb16Snorm => MTLPixelFormat::RGB16Snorm,
            // PixelFormat::Rgb16Uint => MTLPixelFormat::RGB16Uint,
            // PixelFormat::Rgb16Sint => MTLPixelFormat::RGB16Sint,
            // PixelFormat::Rgb16Float => MTLPixelFormat::RGB16Float,
            // PixelFormat::Rgb32Unorm => MTLPixelFormat::RGB32Unorm,
            // PixelFormat::Rgb32Snorm => MTLPixelFormat::RGB32Snorm,
            // PixelFormat::Rgb32Uint => MTLPixelFormat::RGB32Uint,
            // PixelFormat::Rgb32Sint => MTLPixelFormat::RGB32Sint,
            // PixelFormat::Rgb32Float => MTLPixelFormat::RGB32Float,
            PixelFormat::Rgba8Unorm => MTLPixelFormat::RGBA8Unorm,
            PixelFormat::Rgba8Srgb => MTLPixelFormat::RGBA8Unorm_sRGB,
            PixelFormat::Rgba8Snorm => MTLPixelFormat::RGBA8Snorm,
            PixelFormat::Rgba8Uint => MTLPixelFormat::RGBA8Uint,
            PixelFormat::Rgba8Sint => MTLPixelFormat::RGBA8Sint,
            PixelFormat::Rgba16Unorm => MTLPixelFormat::RGBA16Unorm,
            PixelFormat::Rgba16Snorm => MTLPixelFormat::RGBA16Snorm,
            PixelFormat::Rgba16Uint => MTLPixelFormat::RGBA16Uint,
            PixelFormat::Rgba16Sint => MTLPixelFormat::RGBA16Sint,
            PixelFormat::Rgba16Float => MTLPixelFormat::RGBA16Float,
            // PixelFormat::Rgba32Unorm => MTLPixelFormat::RGBA32Unorm,
            // PixelFormat::Rgba32Snorm => MTLPixelFormat::RGBA32Snorm,
            PixelFormat::Rgba32Uint => MTLPixelFormat::RGBA32Uint,
            PixelFormat::Rgba32Sint => MTLPixelFormat::RGBA32Sint,
            PixelFormat::Rgba32Float => MTLPixelFormat::RGBA32Float,
            // PixelFormat::Bgr8Unorm => MTLPixelFormat::BGR8Unorm,
            // PixelFormat::Bgr8Srgb => MTLPixelFormat::BGR8Unorm_sRGB,
            // PixelFormat::Bgr8Snorm => MTLPixelFormat::BGR8Snorm,
            // PixelFormat::Bgr8Uint => MTLPixelFormat::BGR8Uint,
            // PixelFormat::Bgr8Sint => MTLPixelFormat::BGR8Sint,
            PixelFormat::Bgra8Unorm => MTLPixelFormat::BGRA8Unorm,
            PixelFormat::Bgra8Srgb => MTLPixelFormat::BGRA8Unorm_sRGB,
            // PixelFormat::Bgra8Snorm => MTLPixelFormat::BGRA8Snorm,
            // PixelFormat::Bgra8Uint => MTLPixelFormat::BGRA8Uint,
            // PixelFormat::Bgra8Sint => MTLPixelFormat::BGRA8Sint,
            PixelFormat::D16Unorm => MTLPixelFormat::Depth16Unorm,
            PixelFormat::D32Float => MTLPixelFormat::Depth32Float,
            PixelFormat::S8Uint => MTLPixelFormat::Stencil8,
            // PixelFormat::D16UnormS8Uint => MTLPixelFormat::Depth16Unorm_Stencil8,
            PixelFormat::D24UnormS8Uint => MTLPixelFormat::Depth24Unorm_Stencil8,
            PixelFormat::D32FloatS8Uint => MTLPixelFormat::Depth32Float_Stencil8,
            _ => return None,
        })
    }
}

impl TryFromMetal<MTLPixelFormat> for PixelFormat {
    #[inline]
    fn try_from_metal(t: MTLPixelFormat) -> Option<Self> {
        Some(match t {
            MTLPixelFormat::R8Unorm => PixelFormat::R8Unorm,
            MTLPixelFormat::R8Unorm_sRGB => PixelFormat::R8Srgb,
            MTLPixelFormat::R8Snorm => PixelFormat::R8Snorm,
            MTLPixelFormat::R8Uint => PixelFormat::R8Uint,
            MTLPixelFormat::R8Sint => PixelFormat::R8Sint,
            MTLPixelFormat::R16Unorm => PixelFormat::R16Unorm,
            MTLPixelFormat::R16Snorm => PixelFormat::R16Snorm,
            MTLPixelFormat::R16Uint => PixelFormat::R16Uint,
            MTLPixelFormat::R16Sint => PixelFormat::R16Sint,
            MTLPixelFormat::R16Float => PixelFormat::R16Float,
            // MTLPixelFormat::R32Unorm => PixelFormat::R32Unorm,
            // MTLPixelFormat::R32Snorm => PixelFormat::R32Snorm,
            MTLPixelFormat::R32Uint => PixelFormat::R32Uint,
            MTLPixelFormat::R32Sint => PixelFormat::R32Sint,
            MTLPixelFormat::R32Float => PixelFormat::R32Float,
            MTLPixelFormat::RG8Unorm => PixelFormat::Rg8Unorm,
            MTLPixelFormat::RG8Unorm_sRGB => PixelFormat::Rg8Srgb,
            MTLPixelFormat::RG8Snorm => PixelFormat::Rg8Snorm,
            MTLPixelFormat::RG8Uint => PixelFormat::Rg8Uint,
            MTLPixelFormat::RG8Sint => PixelFormat::Rg8Sint,
            MTLPixelFormat::RG16Unorm => PixelFormat::Rg16Unorm,
            MTLPixelFormat::RG16Snorm => PixelFormat::Rg16Snorm,
            MTLPixelFormat::RG16Uint => PixelFormat::Rg16Uint,
            MTLPixelFormat::RG16Sint => PixelFormat::Rg16Sint,
            MTLPixelFormat::RG16Float => PixelFormat::Rg16Float,
            // MTLPixelFormat::RG32Unorm => PixelFormat::Rg32Unorm,
            // MTLPixelFormat::RG32Snorm => PixelFormat::Rg32Snorm,
            MTLPixelFormat::RG32Uint => PixelFormat::Rg32Uint,
            MTLPixelFormat::RG32Sint => PixelFormat::Rg32Sint,
            MTLPixelFormat::RG32Float => PixelFormat::Rg32Float,
            // MTLPixelFormat::RGB8Unorm => PixelFormat::Rgb8Unorm,
            // MTLPixelFormat::RGB8Snorm => PixelFormat::Rgb8Snorm,
            // MTLPixelFormat::RGB8Uint => PixelFormat::Rgb8Uint,
            // MTLPixelFormat::RGB8Sint => PixelFormat::Rgb8Sint,
            // MTLPixelFormat::RGB16Unorm => PixelFormat::Rgb16Unorm,
            // MTLPixelFormat::RGB16Snorm => PixelFormat::Rgb16Snorm,
            // MTLPixelFormat::RGB16Uint => PixelFormat::Rgb16Uint,
            // MTLPixelFormat::RGB16Sint => PixelFormat::Rgb16Sint,
            // MTLPixelFormat::RGB16Float => PixelFormat::Rgb16Float,
            // MTLPixelFormat::RGB32Unorm => PixelFormat::Rgb32Unorm,
            // MTLPixelFormat::RGB32Snorm => PixelFormat::Rgb32Snorm,
            // MTLPixelFormat::RGB32Uint => PixelFormat::Rgb32Uint,
            // MTLPixelFormat::RGB32Sint => PixelFormat::Rgb32Sint,
            // MTLPixelFormat::RGB32Float => PixelFormat::Rgb32Float,
            MTLPixelFormat::RGBA8Unorm => PixelFormat::Rgba8Unorm,
            MTLPixelFormat::RGBA8Unorm_sRGB => PixelFormat::Rgba8Srgb,
            MTLPixelFormat::RGBA8Snorm => PixelFormat::Rgba8Snorm,
            MTLPixelFormat::RGBA8Uint => PixelFormat::Rgba8Uint,
            MTLPixelFormat::RGBA8Sint => PixelFormat::Rgba8Sint,
            MTLPixelFormat::RGBA16Unorm => PixelFormat::Rgba16Unorm,
            MTLPixelFormat::RGBA16Snorm => PixelFormat::Rgba16Snorm,
            MTLPixelFormat::RGBA16Uint => PixelFormat::Rgba16Uint,
            MTLPixelFormat::RGBA16Sint => PixelFormat::Rgba16Sint,
            MTLPixelFormat::RGBA16Float => PixelFormat::Rgba16Float,
            // MTLPixelFormat::RGBA32Unorm => PixelFormat::Rgba32Unorm,
            // MTLPixelFormat::RGBA32Snorm => PixelFormat::Rgba32Snorm,
            MTLPixelFormat::RGBA32Uint => PixelFormat::Rgba32Uint,
            MTLPixelFormat::RGBA32Sint => PixelFormat::Rgba32Sint,
            MTLPixelFormat::RGBA32Float => PixelFormat::Rgba32Float,
            // MTLPixelFormat::BGR8Unorm => PixelFormat::Bgr8Unorm,
            // MTLPixelFormat::BGR8Unorm_sRGB => PixelFormat::Bgr8Srgb,
            // MTLPixelFormat::BGR8Snorm => PixelFormat::Bgr8Snorm,
            // MTLPixelFormat::BGR8Uint => PixelFormat::Bgr8Uint,
            // MTLPixelFormat::BGR8Sint => PixelFormat::Bgr8Sint,
            MTLPixelFormat::BGRA8Unorm => PixelFormat::Bgra8Unorm,
            MTLPixelFormat::BGRA8Unorm_sRGB => PixelFormat::Bgra8Srgb,
            // MTLPixelFormat::BGRA8Snorm => PixelFormat::Bgra8Snorm,
            // MTLPixelFormat::BGRA8Uint => PixelFormat::Bgra8Uint,
            // MTLPixelFormat::BGRA8Sint => PixelFormat::Bgra8Sint,
            MTLPixelFormat::Depth16Unorm => PixelFormat::D16Unorm,
            MTLPixelFormat::Depth32Float => PixelFormat::D32Float,
            MTLPixelFormat::Stencil8 => PixelFormat::S8Uint,
            // MTLPixelFormat::Depth16Unorm_Stencil8 => PixelFormat::D16UnormS8Uint,
            MTLPixelFormat::Depth24Unorm_Stencil8 => PixelFormat::D24UnormS8Uint,
            MTLPixelFormat::Depth32Float_Stencil8 => PixelFormat::D32FloatS8Uint,
            _ => return None,
        })
    }
}

impl TryMetalFrom<VertexFormat> for MTLVertexFormat {
    #[inline]
    fn try_metal_from(format: VertexFormat) -> Option<Self> {
        Some(match format {
            VertexFormat::Uint8 => MTLVertexFormat::UChar,
            VertexFormat::Uint16 => MTLVertexFormat::UShort,
            VertexFormat::Uint32 => MTLVertexFormat::UInt,
            VertexFormat::Sint8 => MTLVertexFormat::Char,
            VertexFormat::Sint16 => MTLVertexFormat::Short,
            VertexFormat::Sint32 => MTLVertexFormat::Int,
            VertexFormat::Unorm8 => MTLVertexFormat::UCharNormalized,
            VertexFormat::Unorm16 => MTLVertexFormat::UShortNormalized,
            // VertexFormat::Unorm32 => MTLVertexFormat::UIntNormalized,
            VertexFormat::Snorm8 => MTLVertexFormat::CharNormalized,
            VertexFormat::Snorm16 => MTLVertexFormat::ShortNormalized,
            // VertexFormat::Snorm32 => MTLVertexFormat::IntNormalized,
            VertexFormat::Float16 => MTLVertexFormat::Half,
            VertexFormat::Float32 => MTLVertexFormat::Float,
            VertexFormat::Uint8x2 => MTLVertexFormat::UChar2,
            VertexFormat::Uint16x2 => MTLVertexFormat::UShort2,
            VertexFormat::Uint32x2 => MTLVertexFormat::UInt2,
            VertexFormat::Sint8x2 => MTLVertexFormat::Char2,
            VertexFormat::Sint16x2 => MTLVertexFormat::Short2,
            VertexFormat::Sint32x2 => MTLVertexFormat::Int2,
            VertexFormat::Unorm8x2 => MTLVertexFormat::UChar2Normalized,
            VertexFormat::Unorm16x2 => MTLVertexFormat::UShort2Normalized,
            // VertexFormat::Unorm32x2 => MTLVertexFormat::UInt2Normalized,
            VertexFormat::Snorm8x2 => MTLVertexFormat::Char2Normalized,
            VertexFormat::Snorm16x2 => MTLVertexFormat::Short2Normalized,
            // VertexFormat::Snorm32x2 => MTLVertexFormat::Int2Normalized,
            VertexFormat::Float16x2 => MTLVertexFormat::Half2,
            VertexFormat::Float32x2 => MTLVertexFormat::Float2,
            VertexFormat::Uint8x3 => MTLVertexFormat::UChar3,
            VertexFormat::Uint16x3 => MTLVertexFormat::UShort3,
            VertexFormat::Uint32x3 => MTLVertexFormat::UInt3,
            VertexFormat::Sint8x3 => MTLVertexFormat::Char3,
            VertexFormat::Sint16x3 => MTLVertexFormat::Short3,
            VertexFormat::Sint32x3 => MTLVertexFormat::Int3,
            VertexFormat::Unorm8x3 => MTLVertexFormat::UChar3Normalized,
            VertexFormat::Unorm16x3 => MTLVertexFormat::UShort3Normalized,
            // VertexFormat::Unorm32x3 => MTLVertexFormat::UInt3Normalized,
            VertexFormat::Snorm8x3 => MTLVertexFormat::Char3Normalized,
            VertexFormat::Snorm16x3 => MTLVertexFormat::Short3Normalized,
            // VertexFormat::Snorm32x3 => MTLVertexFormat::Int3Normalized,
            VertexFormat::Float16x3 => MTLVertexFormat::Half3,
            VertexFormat::Float32x3 => MTLVertexFormat::Float3,
            VertexFormat::Uint8x4 => MTLVertexFormat::UChar4,
            VertexFormat::Uint16x4 => MTLVertexFormat::UShort4,
            VertexFormat::Uint32x4 => MTLVertexFormat::UInt4,
            VertexFormat::Sint8x4 => MTLVertexFormat::Char4,
            VertexFormat::Sint16x4 => MTLVertexFormat::Short4,
            VertexFormat::Sint32x4 => MTLVertexFormat::Int4,
            VertexFormat::Unorm8x4 => MTLVertexFormat::UChar4Normalized,
            VertexFormat::Unorm16x4 => MTLVertexFormat::UShort4Normalized,
            // VertexFormat::Unorm32x4 => MTLVertexFormat::UInt4Normalized,
            VertexFormat::Snorm8x4 => MTLVertexFormat::Char4Normalized,
            VertexFormat::Snorm16x4 => MTLVertexFormat::Short4Normalized,
            // VertexFormat::Snorm32x4 => MTLVertexFormat::Int4Normalized,
            VertexFormat::Float16x4 => MTLVertexFormat::Half4,
            VertexFormat::Float32x4 => MTLVertexFormat::Float4,
            _ => return None,
        })
    }
}

impl TryMetalFrom<VertexFormat> for MTLAttributeFormat {
    #[inline]
    fn try_metal_from(format: VertexFormat) -> Option<Self> {
        Some(match format {
            VertexFormat::Uint8 => MTLAttributeFormat::UChar,
            VertexFormat::Uint16 => MTLAttributeFormat::UShort,
            VertexFormat::Uint32 => MTLAttributeFormat::UInt,
            VertexFormat::Sint8 => MTLAttributeFormat::Char,
            VertexFormat::Sint16 => MTLAttributeFormat::Short,
            VertexFormat::Sint32 => MTLAttributeFormat::Int,
            VertexFormat::Unorm8 => MTLAttributeFormat::UCharNormalized,
            VertexFormat::Unorm16 => MTLAttributeFormat::UShortNormalized,
            // VertexFormat::Unorm32 => MTLAttributeFormat::UIntNormalized,
            VertexFormat::Snorm8 => MTLAttributeFormat::CharNormalized,
            VertexFormat::Snorm16 => MTLAttributeFormat::ShortNormalized,
            // VertexFormat::Snorm32 => MTLAttributeFormat::IntNormalized,
            VertexFormat::Float16 => MTLAttributeFormat::Half,
            VertexFormat::Float32 => MTLAttributeFormat::Float,
            VertexFormat::Uint8x2 => MTLAttributeFormat::UChar2,
            VertexFormat::Uint16x2 => MTLAttributeFormat::UShort2,
            VertexFormat::Uint32x2 => MTLAttributeFormat::UInt2,
            VertexFormat::Sint8x2 => MTLAttributeFormat::Char2,
            VertexFormat::Sint16x2 => MTLAttributeFormat::Short2,
            VertexFormat::Sint32x2 => MTLAttributeFormat::Int2,
            VertexFormat::Unorm8x2 => MTLAttributeFormat::UChar2Normalized,
            VertexFormat::Unorm16x2 => MTLAttributeFormat::UShort2Normalized,
            // VertexFormat::Unorm32x2 => MTLAttributeFormat::UInt2Normalized,
            VertexFormat::Snorm8x2 => MTLAttributeFormat::Char2Normalized,
            VertexFormat::Snorm16x2 => MTLAttributeFormat::Short2Normalized,
            // VertexFormat::Snorm32x2 => MTLAttributeFormat::Int2Normalized,
            VertexFormat::Float16x2 => MTLAttributeFormat::Half2,
            VertexFormat::Float32x2 => MTLAttributeFormat::Float2,
            VertexFormat::Uint8x3 => MTLAttributeFormat::UChar3,
            VertexFormat::Uint16x3 => MTLAttributeFormat::UShort3,
            VertexFormat::Uint32x3 => MTLAttributeFormat::UInt3,
            VertexFormat::Sint8x3 => MTLAttributeFormat::Char3,
            VertexFormat::Sint16x3 => MTLAttributeFormat::Short3,
            VertexFormat::Sint32x3 => MTLAttributeFormat::Int3,
            VertexFormat::Unorm8x3 => MTLAttributeFormat::UChar3Normalized,
            VertexFormat::Unorm16x3 => MTLAttributeFormat::UShort3Normalized,
            // VertexFormat::Unorm32x3 => MTLAttributeFormat::UInt3Normalized,
            VertexFormat::Snorm8x3 => MTLAttributeFormat::Char3Normalized,
            VertexFormat::Snorm16x3 => MTLAttributeFormat::Short3Normalized,
            // VertexFormat::Snorm32x3 => MTLAttributeFormat::Int3Normalized,
            VertexFormat::Float16x3 => MTLAttributeFormat::Half3,
            VertexFormat::Float32x3 => MTLAttributeFormat::Float3,
            VertexFormat::Uint8x4 => MTLAttributeFormat::UChar4,
            VertexFormat::Uint16x4 => MTLAttributeFormat::UShort4,
            VertexFormat::Uint32x4 => MTLAttributeFormat::UInt4,
            VertexFormat::Sint8x4 => MTLAttributeFormat::Char4,
            VertexFormat::Sint16x4 => MTLAttributeFormat::Short4,
            VertexFormat::Sint32x4 => MTLAttributeFormat::Int4,
            VertexFormat::Unorm8x4 => MTLAttributeFormat::UChar4Normalized,
            VertexFormat::Unorm16x4 => MTLAttributeFormat::UShort4Normalized,
            // VertexFormat::Unorm32x4 => MTLAttributeFormat::UInt4Normalized,
            VertexFormat::Snorm8x4 => MTLAttributeFormat::Char4Normalized,
            VertexFormat::Snorm16x4 => MTLAttributeFormat::Short4Normalized,
            // VertexFormat::Snorm32x4 => MTLAttributeFormat::Int4Normalized,
            VertexFormat::Float16x4 => MTLAttributeFormat::Half4,
            VertexFormat::Float32x4 => MTLAttributeFormat::Float4,
            _ => return None,
        })
    }
}

impl MetalFrom<PrimitiveTopology> for MTLPrimitiveTopologyClass {
    #[inline]
    fn metal_from(t: PrimitiveTopology) -> Self {
        match t {
            PrimitiveTopology::Point => MTLPrimitiveTopologyClass::Point,
            PrimitiveTopology::Line => MTLPrimitiveTopologyClass::Line,
            PrimitiveTopology::Triangle => MTLPrimitiveTopologyClass::Triangle,
        }
    }
}

impl MetalFrom<PrimitiveTopology> for MTLPrimitiveType {
    #[inline]
    fn metal_from(t: PrimitiveTopology) -> Self {
        match t {
            PrimitiveTopology::Point => MTLPrimitiveType::Point,
            PrimitiveTopology::Line => MTLPrimitiveType::Line,
            PrimitiveTopology::Triangle => MTLPrimitiveType::Triangle,
        }
    }
}

impl MetalFrom<BlendOp> for MTLBlendOperation {
    #[inline]
    fn metal_from(t: BlendOp) -> Self {
        match t {
            BlendOp::Add => MTLBlendOperation::Add,
            BlendOp::Subtract => MTLBlendOperation::Subtract,
            BlendOp::ReverseSubtract => MTLBlendOperation::ReverseSubtract,
            BlendOp::Min => MTLBlendOperation::Min,
            BlendOp::Max => MTLBlendOperation::Max,
        }
    }
}

impl MetalFrom<BlendFactor> for MTLBlendFactor {
    #[inline]
    fn metal_from(t: BlendFactor) -> Self {
        match t {
            BlendFactor::Zero => MTLBlendFactor::Zero,
            BlendFactor::One => MTLBlendFactor::One,
            BlendFactor::SrcColor => MTLBlendFactor::SourceColor,
            BlendFactor::OneMinusSrcColor => MTLBlendFactor::OneMinusSourceColor,
            BlendFactor::SrcAlpha => MTLBlendFactor::SourceAlpha,
            BlendFactor::OneMinusSrcAlpha => MTLBlendFactor::OneMinusSourceAlpha,
            BlendFactor::DstColor => MTLBlendFactor::DestinationColor,
            BlendFactor::OneMinusDstColor => MTLBlendFactor::OneMinusDestinationColor,
            BlendFactor::DstAlpha => MTLBlendFactor::DestinationAlpha,
            BlendFactor::OneMinusDstAlpha => MTLBlendFactor::OneMinusDestinationAlpha,
            BlendFactor::SrcAlphaSaturated => MTLBlendFactor::SourceAlphaSaturated,
            // BlendFactor::BlendColor => MTLBlendFactor::BlendColor,
            // BlendFactor::OneMinusBlendColor => MTLBlendFactor::OneMinusBlendColor,
        }
    }
}

impl MetalFrom<CompareFunction> for MTLCompareFunction {
    #[inline]
    fn metal_from(t: CompareFunction) -> Self {
        match t {
            CompareFunction::Never => MTLCompareFunction::Never,
            CompareFunction::Less => MTLCompareFunction::Less,
            CompareFunction::Equal => MTLCompareFunction::Equal,
            CompareFunction::LessEqual => MTLCompareFunction::LessEqual,
            CompareFunction::Greater => MTLCompareFunction::Greater,
            CompareFunction::NotEqual => MTLCompareFunction::NotEqual,
            CompareFunction::GreaterEqual => MTLCompareFunction::GreaterEqual,
            CompareFunction::Always => MTLCompareFunction::Always,
        }
    }
}

impl MetalFrom<WriteMask> for MTLColorWriteMask {
    #[inline]
    fn metal_from(t: WriteMask) -> Self {
        let mut mask = MTLColorWriteMask::empty();
        if t.contains(WriteMask::RED) {
            mask |= MTLColorWriteMask::Red;
        }
        if t.contains(WriteMask::GREEN) {
            mask |= MTLColorWriteMask::Green;
        }
        if t.contains(WriteMask::BLUE) {
            mask |= MTLColorWriteMask::Blue;
        }
        if t.contains(WriteMask::ALPHA) {
            mask |= MTLColorWriteMask::Alpha;
        }
        mask
    }
}

impl MetalFrom<ImageUsage> for MTLTextureUsage {
    #[inline]
    fn metal_from(t: ImageUsage) -> Self {
        let mut mask = MTLTextureUsage::empty();
        if t.contains(ImageUsage::SAMPLED) {
            mask |= MTLTextureUsage::ShaderRead;
        }
        if t.contains(ImageUsage::STORAGE) {
            mask |= MTLTextureUsage::ShaderWrite;
        }
        if t.contains(ImageUsage::TARGET) {
            mask |= MTLTextureUsage::RenderTarget;
        }
        // if t.contains(ImageUsage::TRANSFER_SRC) {
        //     mask |= MTLTextureUsage::Unknown;
        // }
        // if t.contains(ImageUsage::TRANSFER_DST) {
        //     mask |= MTLTextureUsage::Unknown;
        // }
        mask
    }
}

impl FromMetal<MTLTextureUsage> for ImageUsage {
    #[inline]
    fn from_metal(t: MTLTextureUsage) -> Self {
        let mut mask = ImageUsage::empty();
        if t.contains(MTLTextureUsage::ShaderRead) {
            mask |= ImageUsage::SAMPLED;
        }
        if t.contains(MTLTextureUsage::ShaderWrite) {
            mask |= ImageUsage::STORAGE;
        }
        if t.contains(MTLTextureUsage::RenderTarget) {
            mask |= ImageUsage::TARGET;
        }

        mask |= ImageUsage::TRANSFER_SRC;
        mask |= ImageUsage::TRANSFER_DST;

        mask
    }
}

// impl MetalFrom<RenderStages> for MTLRenderStages {
//     fn metal_from(stages: RenderStages) -> Self {
//         let mut mask = MTLRenderStages::empty();
//         if stages.intersects(
//             RenderStages::DRAW_INDIRECT | RenderStages::VERTEX_INPUT | RenderStages::VERTEX_SHADER,
//         ) {
//             mask |= MTLRenderStages::Vertex;
//         }
//         if stages.contains(
//             RenderStages::EARLY_FRAGMENT_TEST
//                 | RenderStages::FRAGMENT_SHADER
//                 | RenderStages::LATE_FRAGMENT_TEST
//                 | RenderStages::COLOR_OUTPUT,
//         ) {
//             mask |= MTLRenderStages::Fragment;
//         }
//         mask
//     }
// }

impl MetalFrom<Filter> for MTLSamplerMinMagFilter {
    #[inline]
    fn metal_from(filter: Filter) -> Self {
        match filter {
            Filter::Nearest => MTLSamplerMinMagFilter::Nearest,
            Filter::Linear => MTLSamplerMinMagFilter::Linear,
        }
    }
}

impl MetalFrom<MipMapMode> for MTLSamplerMipFilter {
    #[inline]
    fn metal_from(mode: MipMapMode) -> Self {
        match mode {
            MipMapMode::Nearest => MTLSamplerMipFilter::Nearest,
            MipMapMode::Linear => MTLSamplerMipFilter::Linear,
        }
    }
}

impl MetalFrom<AddressMode> for MTLSamplerAddressMode {
    #[inline]
    fn metal_from(mode: AddressMode) -> Self {
        match mode {
            AddressMode::ClampToEdge => MTLSamplerAddressMode::ClampToEdge,
            AddressMode::Repeat => MTLSamplerAddressMode::Repeat,
            AddressMode::MirrorRepeat => MTLSamplerAddressMode::MirrorRepeat,
        }
    }
}
