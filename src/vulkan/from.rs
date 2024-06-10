use ash::vk;

use crate::{
    generic::{
        AddressMode, BlendFactor, BlendOp, BufferUsage, CompareFunction, ComponentSwizzle, Culling,
        Extent2, Extent3, FamilyCapabilities, Filter, FrontFace, ImageExtent, ImageUsage,
        MipMapMode, Offset2, Offset3, PipelineStage, PipelineStages, PixelFormat, QueueFlags,
        ShaderStage, ShaderStages, Swizzle, VertexFormat, WriteMask,
    },
    mat,
};

macro_rules! from_flags {
    ($from:ty => $to:ty, [$($from_flag:ident),* $(,)?], $flags:expr) => {
        from_flags!($from => $to, [$($from_flag => $from_flag,)*], $flags)
    };
    ($from:ty => $to:ty, [$($from_flag:ident => $to_flag:ident),* $(,)?], $flags:expr) => {{
        let mut dst = <$to>::empty();
        #[allow(unused_mut)]
        let mut src = $flags;
        $(
            if src.contains(<$from>::$from_flag) {
                dst |= <$to>::$to_flag;
            }
        )*
        dst
    }};
}

pub trait FromAsh<A> {
    fn from_ash(ash: A) -> Self;
}

pub trait AshInto<T> {
    fn ash_into(self) -> T;
}

impl<T, A> AshInto<T> for A
where
    T: FromAsh<A>,
{
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn ash_into(self) -> T {
        T::from_ash(self)
    }
}

pub trait AshFrom<T> {
    fn ash_from(generic: T) -> Self;
}

pub trait IntoAsh<A> {
    fn into_ash(self) -> A;
}

impl<A, T> IntoAsh<A> for T
where
    A: AshFrom<T>,
{
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn into_ash(self) -> A {
        A::ash_from(self)
    }
}

pub trait TryFromAsh<T>: Sized {
    fn try_from_ash(t: T) -> Option<Self>;
}

pub trait TryAshInto<T> {
    fn try_ash_into(self) -> Option<T>;
}

impl<T, U> TryAshInto<U> for T
where
    U: TryFromAsh<T>,
{
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn try_ash_into(self) -> Option<U> {
        U::try_from_ash(self)
    }
}

pub trait TryAshFrom<T>: Sized {
    fn try_ash_from(t: T) -> Option<Self>;
}

pub trait TryIntoAsh<T> {
    fn try_into_ash(self) -> Option<T>;
}

impl<T, U> TryIntoAsh<U> for T
where
    U: TryAshFrom<T>,
{
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn try_into_ash(self) -> Option<U> {
        U::try_ash_from(self)
    }
}

impl FromAsh<vk::QueueFamilyProperties> for FamilyCapabilities {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn from_ash(value: vk::QueueFamilyProperties) -> Self {
        FamilyCapabilities {
            queue_flags: value.queue_flags.ash_into(),
            queue_count: value.queue_count.try_into().unwrap_or(usize::MAX), // Saturate is OK.
        }
    }
}

impl FromAsh<vk::QueueFamilyProperties2<'_>> for FamilyCapabilities {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn from_ash(value: vk::QueueFamilyProperties2) -> Self {
        value.queue_family_properties.ash_into()
    }
}

impl FromAsh<vk::QueueFlags> for QueueFlags {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn from_ash(value: vk::QueueFlags) -> Self {
        // from_flags!(vk::QueueFlags => QueueFlags, [GRAPHICS, COMPUTE, TRANSFER], value)

        let mut result = QueueFlags::empty();
        if value.contains(vk::QueueFlags::GRAPHICS) {
            result |= QueueFlags::GRAPHICS | QueueFlags::TRANSFER;
        } else if value.contains(vk::QueueFlags::COMPUTE) {
            result |= QueueFlags::COMPUTE | QueueFlags::TRANSFER;
        } else if value.contains(vk::QueueFlags::TRANSFER) {
            result |= QueueFlags::TRANSFER;
        }
        result
    }
}

impl AshFrom<BufferUsage> for vk::BufferUsageFlags {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn ash_from(value: BufferUsage) -> Self {
        from_flags!(BufferUsage => vk::BufferUsageFlags, [
            TRANSFER_SRC => TRANSFER_SRC,
            TRANSFER_DST => TRANSFER_DST,
            UNIFORM => UNIFORM_BUFFER,
            STORAGE => STORAGE_BUFFER,
            INDEX => INDEX_BUFFER,
            VERTEX => VERTEX_BUFFER,
            INDIRECT => INDIRECT_BUFFER,
        ], value)
    }
}

impl AshFrom<ImageExtent> for vk::ImageType {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn ash_from(value: ImageExtent) -> Self {
        match value {
            ImageExtent::D1(_) => vk::ImageType::TYPE_1D,
            ImageExtent::D2(_) => vk::ImageType::TYPE_2D,
            ImageExtent::D3(_) => vk::ImageType::TYPE_3D,
        }
    }
}

impl AshFrom<ImageExtent> for vk::ImageViewType {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn ash_from(value: ImageExtent) -> Self {
        match value {
            ImageExtent::D1(_) => vk::ImageViewType::TYPE_1D,
            ImageExtent::D2(_) => vk::ImageViewType::TYPE_2D,
            ImageExtent::D3(_) => vk::ImageViewType::TYPE_3D,
        }
    }
}

impl TryAshFrom<PixelFormat> for vk::Format {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn try_ash_from(value: PixelFormat) -> Option<Self> {
        Some(match value {
            PixelFormat::R8Unorm => vk::Format::R8_UNORM,
            PixelFormat::R8Snorm => vk::Format::R8_SNORM,
            PixelFormat::R8Uint => vk::Format::R8_UINT,
            PixelFormat::R8Sint => vk::Format::R8_SINT,
            PixelFormat::R8Srgb => vk::Format::R8_SRGB,
            PixelFormat::R16Unorm => vk::Format::R16_UNORM,
            PixelFormat::R16Snorm => vk::Format::R16_SNORM,
            PixelFormat::R16Uint => vk::Format::R16_UINT,
            PixelFormat::R16Sint => vk::Format::R16_SINT,
            PixelFormat::R16Float => vk::Format::R16_SFLOAT,
            // PixelFormat::R32Unorm => vk::Format::R32_UNORM,
            // PixelFormat::R32Snorm => vk::Format::R32_SNORM,
            PixelFormat::R32Uint => vk::Format::R32_UINT,
            PixelFormat::R32Sint => vk::Format::R32_SINT,
            PixelFormat::R32Float => vk::Format::R32_SFLOAT,
            PixelFormat::Rg8Unorm => vk::Format::R8G8_UNORM,
            PixelFormat::Rg8Snorm => vk::Format::R8G8_SNORM,
            PixelFormat::Rg8Uint => vk::Format::R8G8_UINT,
            PixelFormat::Rg8Sint => vk::Format::R8G8_SINT,
            PixelFormat::Rg8Srgb => vk::Format::R8G8_SRGB,
            PixelFormat::Rg16Unorm => vk::Format::R16G16_UNORM,
            PixelFormat::Rg16Snorm => vk::Format::R16G16_SNORM,
            PixelFormat::Rg16Uint => vk::Format::R16G16_UINT,
            PixelFormat::Rg16Sint => vk::Format::R16G16_SINT,
            PixelFormat::Rg16Float => vk::Format::R16G16_SFLOAT,
            // PixelFormat::Rg32Unorm => vk::Format::R32G32_UNORM,
            // PixelFormat::Rg32Snorm => vk::Format::R32G32_SNORM,
            PixelFormat::Rg32Uint => vk::Format::R32G32_UINT,
            PixelFormat::Rg32Sint => vk::Format::R32G32_SINT,
            PixelFormat::Rg32Float => vk::Format::R32G32_SFLOAT,
            PixelFormat::Rgb8Unorm => vk::Format::R8G8B8_UNORM,
            PixelFormat::Rgb8Snorm => vk::Format::R8G8B8_SNORM,
            PixelFormat::Rgb8Uint => vk::Format::R8G8B8_UINT,
            PixelFormat::Rgb8Sint => vk::Format::R8G8B8_SINT,
            PixelFormat::Rgb8Srgb => vk::Format::R8G8B8_SRGB,
            PixelFormat::Rgb16Unorm => vk::Format::R16G16B16_UNORM,
            PixelFormat::Rgb16Snorm => vk::Format::R16G16B16_SNORM,
            PixelFormat::Rgb16Uint => vk::Format::R16G16B16_UINT,
            PixelFormat::Rgb16Sint => vk::Format::R16G16B16_SINT,
            PixelFormat::Rgb16Float => vk::Format::R16G16B16_SFLOAT,
            // PixelFormat::Rgb32Unorm => vk::Format::R32G32B32_UNORM,
            // PixelFormat::Rgb32Snorm => vk::Format::R32G32B32_SNORM,
            PixelFormat::Rgb32Uint => vk::Format::R32G32B32_UINT,
            PixelFormat::Rgb32Sint => vk::Format::R32G32B32_SINT,
            PixelFormat::Rgb32Float => vk::Format::R32G32B32_SFLOAT,
            PixelFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
            PixelFormat::Rgba8Snorm => vk::Format::R8G8B8A8_SNORM,
            PixelFormat::Rgba8Uint => vk::Format::R8G8B8A8_UINT,
            PixelFormat::Rgba8Sint => vk::Format::R8G8B8A8_SINT,
            PixelFormat::Rgba8Srgb => vk::Format::R8G8B8A8_SRGB,
            PixelFormat::Rgba16Unorm => vk::Format::R16G16B16A16_UNORM,
            PixelFormat::Rgba16Snorm => vk::Format::R16G16B16A16_SNORM,
            PixelFormat::Rgba16Uint => vk::Format::R16G16B16A16_UINT,
            PixelFormat::Rgba16Sint => vk::Format::R16G16B16A16_SINT,
            PixelFormat::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
            // PixelFormat::Rgba32Unorm => vk::Format::R32G32B32A32_UNORM,
            // PixelFormat::Rgba32Snorm => vk::Format::R32G32B32A32_SNORM,
            PixelFormat::Rgba32Uint => vk::Format::R32G32B32A32_UINT,
            PixelFormat::Rgba32Sint => vk::Format::R32G32B32A32_SINT,
            PixelFormat::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
            PixelFormat::Bgr8Unorm => vk::Format::B8G8R8_UNORM,
            PixelFormat::Bgr8Snorm => vk::Format::B8G8R8_SNORM,
            PixelFormat::Bgr8Uint => vk::Format::B8G8R8_UINT,
            PixelFormat::Bgr8Sint => vk::Format::B8G8R8_SINT,
            PixelFormat::Bgr8Srgb => vk::Format::B8G8R8_SRGB,
            PixelFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
            PixelFormat::Bgra8Snorm => vk::Format::B8G8R8A8_SNORM,
            PixelFormat::Bgra8Uint => vk::Format::B8G8R8A8_UINT,
            PixelFormat::Bgra8Sint => vk::Format::B8G8R8A8_SINT,
            PixelFormat::Bgra8Srgb => vk::Format::B8G8R8A8_SRGB,
            PixelFormat::D16Unorm => vk::Format::D16_UNORM,
            PixelFormat::D32Float => vk::Format::D32_SFLOAT,
            PixelFormat::S8Uint => vk::Format::S8_UINT,
            PixelFormat::D16UnormS8Uint => vk::Format::D16_UNORM_S8_UINT,
            PixelFormat::D24UnormS8Uint => vk::Format::D24_UNORM_S8_UINT,
            PixelFormat::D32FloatS8Uint => vk::Format::D32_SFLOAT_S8_UINT,
            _ => return None,
        })
    }
}

impl TryFromAsh<vk::Format> for PixelFormat {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn try_from_ash(value: vk::Format) -> Option<Self> {
        Some(match value {
            vk::Format::R8_UNORM => PixelFormat::R8Unorm,
            vk::Format::R8_SNORM => PixelFormat::R8Snorm,
            vk::Format::R8_UINT => PixelFormat::R8Uint,
            vk::Format::R8_SINT => PixelFormat::R8Sint,
            vk::Format::R8_SRGB => PixelFormat::R8Srgb,
            vk::Format::R16_UNORM => PixelFormat::R16Unorm,
            vk::Format::R16_SNORM => PixelFormat::R16Snorm,
            vk::Format::R16_UINT => PixelFormat::R16Uint,
            vk::Format::R16_SINT => PixelFormat::R16Sint,
            vk::Format::R16_SFLOAT => PixelFormat::R16Float,
            // vk::Format::R32_UNORM => PixelFormat::R32Unorm,
            // vk::Format::R32_SNORM => PixelFormat::R32Snorm,
            vk::Format::R32_UINT => PixelFormat::R32Uint,
            vk::Format::R32_SINT => PixelFormat::R32Sint,
            vk::Format::R32_SFLOAT => PixelFormat::R32Float,
            vk::Format::R8G8_UNORM => PixelFormat::Rg8Unorm,
            vk::Format::R8G8_SNORM => PixelFormat::Rg8Snorm,
            vk::Format::R8G8_UINT => PixelFormat::Rg8Uint,
            vk::Format::R8G8_SINT => PixelFormat::Rg8Sint,
            vk::Format::R8G8_SRGB => PixelFormat::Rg8Srgb,
            vk::Format::R16G16_UNORM => PixelFormat::Rg16Unorm,
            vk::Format::R16G16_SNORM => PixelFormat::Rg16Snorm,
            vk::Format::R16G16_UINT => PixelFormat::Rg16Uint,
            vk::Format::R16G16_SINT => PixelFormat::Rg16Sint,
            vk::Format::R16G16_SFLOAT => PixelFormat::Rg16Float,
            // vk::Format::R32G32_UNORM => PixelFormat::Rg32Unorm,
            // vk::Format::R32G32_SNORM => PixelFormat::Rg32Snorm,
            vk::Format::R32G32_UINT => PixelFormat::Rg32Uint,
            vk::Format::R32G32_SINT => PixelFormat::Rg32Sint,
            vk::Format::R32G32_SFLOAT => PixelFormat::Rg32Float,
            vk::Format::R8G8B8_UNORM => PixelFormat::Rgb8Unorm,
            vk::Format::R8G8B8_SNORM => PixelFormat::Rgb8Snorm,
            vk::Format::R8G8B8_UINT => PixelFormat::Rgb8Uint,
            vk::Format::R8G8B8_SINT => PixelFormat::Rgb8Sint,
            vk::Format::R8G8B8_SRGB => PixelFormat::Rgb8Srgb,
            vk::Format::R16G16B16_UNORM => PixelFormat::Rgb16Unorm,
            vk::Format::R16G16B16_SNORM => PixelFormat::Rgb16Snorm,
            vk::Format::R16G16B16_UINT => PixelFormat::Rgb16Uint,
            vk::Format::R16G16B16_SINT => PixelFormat::Rgb16Sint,
            vk::Format::R16G16B16_SFLOAT => PixelFormat::Rgb16Float,
            // vk::Format::R32G32B32_UNORM => PixelFormat::Rgb32Unorm,
            // vk::Format::R32G32B32_SNORM => PixelFormat::Rgb32Snorm,
            vk::Format::R32G32B32_UINT => PixelFormat::Rgb32Uint,
            vk::Format::R32G32B32_SINT => PixelFormat::Rgb32Sint,
            vk::Format::R32G32B32_SFLOAT => PixelFormat::Rgb32Float,
            vk::Format::R8G8B8A8_UNORM => PixelFormat::Rgba8Unorm,
            vk::Format::R8G8B8A8_SNORM => PixelFormat::Rgba8Snorm,
            vk::Format::R8G8B8A8_UINT => PixelFormat::Rgba8Uint,
            vk::Format::R8G8B8A8_SINT => PixelFormat::Rgba8Sint,
            vk::Format::R8G8B8A8_SRGB => PixelFormat::Rgba8Srgb,
            vk::Format::R16G16B16A16_UNORM => PixelFormat::Rgba16Unorm,
            vk::Format::R16G16B16A16_SNORM => PixelFormat::Rgba16Snorm,
            vk::Format::R16G16B16A16_UINT => PixelFormat::Rgba16Uint,
            vk::Format::R16G16B16A16_SINT => PixelFormat::Rgba16Sint,
            vk::Format::R16G16B16A16_SFLOAT => PixelFormat::Rgba16Float,
            // vk::Format::R32G32B32A32_UNORM => PixelFormat::Rgba32Unorm,
            // vk::Format::R32G32B32A32_SNORM => PixelFormat::Rgba32Snorm,
            vk::Format::R32G32B32A32_UINT => PixelFormat::Rgba32Uint,
            vk::Format::R32G32B32A32_SINT => PixelFormat::Rgba32Sint,
            vk::Format::R32G32B32A32_SFLOAT => PixelFormat::Rgba32Float,
            vk::Format::B8G8R8_UNORM => PixelFormat::Bgr8Unorm,
            vk::Format::B8G8R8_SNORM => PixelFormat::Bgr8Snorm,
            vk::Format::B8G8R8_UINT => PixelFormat::Bgr8Uint,
            vk::Format::B8G8R8_SINT => PixelFormat::Bgr8Sint,
            vk::Format::B8G8R8_SRGB => PixelFormat::Bgr8Srgb,
            vk::Format::B8G8R8A8_UNORM => PixelFormat::Bgra8Unorm,
            vk::Format::B8G8R8A8_SNORM => PixelFormat::Bgra8Snorm,
            vk::Format::B8G8R8A8_UINT => PixelFormat::Bgra8Uint,
            vk::Format::B8G8R8A8_SINT => PixelFormat::Bgra8Sint,
            vk::Format::B8G8R8A8_SRGB => PixelFormat::Bgra8Srgb,
            vk::Format::D16_UNORM => PixelFormat::D16Unorm,
            vk::Format::D32_SFLOAT => PixelFormat::D32Float,
            vk::Format::S8_UINT => PixelFormat::S8Uint,
            vk::Format::D16_UNORM_S8_UINT => PixelFormat::D16UnormS8Uint,
            vk::Format::D24_UNORM_S8_UINT => PixelFormat::D24UnormS8Uint,
            vk::Format::D32_SFLOAT_S8_UINT => PixelFormat::D32FloatS8Uint,
            _ => return None,
        })
    }
}

impl AshFrom<(ImageUsage, PixelFormat)> for vk::ImageUsageFlags {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn ash_from((usage, format): (ImageUsage, PixelFormat)) -> Self {
        let mut result = vk::ImageUsageFlags::empty();
        if usage.contains(ImageUsage::TRANSFER_SRC) {
            result |= vk::ImageUsageFlags::TRANSFER_SRC;
        }
        if usage.contains(ImageUsage::TRANSFER_DST) {
            result |= vk::ImageUsageFlags::TRANSFER_DST;
        }
        if usage.contains(ImageUsage::SAMPLED) {
            result |= vk::ImageUsageFlags::SAMPLED;
        }
        if usage.contains(ImageUsage::STORAGE) {
            result |= vk::ImageUsageFlags::STORAGE;
        }
        if usage.contains(ImageUsage::TARGET) {
            if format.is_color() {
                result |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
            } else {
                result |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
            }
        }
        result
    }
}

impl FromAsh<vk::ImageUsageFlags> for ImageUsage {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn from_ash(usage: vk::ImageUsageFlags) -> Self {
        let mut result = ImageUsage::empty();
        if usage.contains(vk::ImageUsageFlags::TRANSFER_SRC) {
            result |= ImageUsage::TRANSFER_SRC;
        }
        if usage.contains(vk::ImageUsageFlags::TRANSFER_DST) {
            result |= ImageUsage::TRANSFER_DST;
        }
        if usage.contains(vk::ImageUsageFlags::SAMPLED) {
            result |= ImageUsage::SAMPLED;
        }
        if usage.contains(vk::ImageUsageFlags::STORAGE) {
            result |= ImageUsage::STORAGE;
        }
        if usage.intersects(
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        ) {
            result |= ImageUsage::TARGET;
        }
        result
    }
}

impl TryAshFrom<VertexFormat> for vk::Format {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn try_ash_from(value: VertexFormat) -> Option<Self> {
        Some(match value {
            VertexFormat::Uint8 => vk::Format::R8_UINT,
            VertexFormat::Uint16 => vk::Format::R16_UINT,
            VertexFormat::Uint32 => vk::Format::R32_UINT,
            VertexFormat::Sint8 => vk::Format::R8_SINT,
            VertexFormat::Sint16 => vk::Format::R16_SINT,
            VertexFormat::Sint32 => vk::Format::R32_SINT,
            VertexFormat::Unorm8 => vk::Format::R8_UNORM,
            VertexFormat::Unorm16 => vk::Format::R16_UNORM,
            // VertexFormat::Unorm32 => vk::Format::R32_UNORM,
            VertexFormat::Snorm8 => vk::Format::R8_SNORM,
            VertexFormat::Snorm16 => vk::Format::R16_SNORM,
            // VertexFormat::Snorm32 => vk::Format::R32_SNORM,
            VertexFormat::Float16 => vk::Format::R16_SFLOAT,
            VertexFormat::Float32 => vk::Format::R32_SFLOAT,
            VertexFormat::Uint8x2 => vk::Format::R8G8_UINT,
            VertexFormat::Uint16x2 => vk::Format::R16G16_UINT,
            VertexFormat::Uint32x2 => vk::Format::R32G32_UINT,
            VertexFormat::Sint8x2 => vk::Format::R8G8_SINT,
            VertexFormat::Sint16x2 => vk::Format::R16G16_SINT,
            VertexFormat::Sint32x2 => vk::Format::R32G32_SINT,
            VertexFormat::Unorm8x2 => vk::Format::R8G8_UNORM,
            VertexFormat::Unorm16x2 => vk::Format::R16G16_UNORM,
            // VertexFormat::Unorm32x2 => vk::Format::R32G32_UNORM,
            VertexFormat::Snorm8x2 => vk::Format::R8G8_SNORM,
            VertexFormat::Snorm16x2 => vk::Format::R16G16_SNORM,
            // VertexFormat::Snorm32x2 => vk::Format::R32G32_SNORM,
            VertexFormat::Float16x2 => vk::Format::R16G16_SFLOAT,
            VertexFormat::Float32x2 => vk::Format::R32G32_SFLOAT,
            VertexFormat::Uint8x3 => vk::Format::R8G8B8_UINT,
            VertexFormat::Uint16x3 => vk::Format::R16G16B16_UINT,
            VertexFormat::Uint32x3 => vk::Format::R32G32B32_UINT,
            VertexFormat::Sint8x3 => vk::Format::R8G8B8_SINT,
            VertexFormat::Sint16x3 => vk::Format::R16G16B16_SINT,
            VertexFormat::Sint32x3 => vk::Format::R32G32B32_SINT,
            VertexFormat::Unorm8x3 => vk::Format::R8G8B8_UNORM,
            VertexFormat::Unorm16x3 => vk::Format::R16G16B16_UNORM,
            // VertexFormat::Unorm32x3 => vk::Format::R32G32B32_UNORM,
            VertexFormat::Snorm8x3 => vk::Format::R8G8B8_SNORM,
            VertexFormat::Snorm16x3 => vk::Format::R16G16B16_SNORM,
            // VertexFormat::Snorm32x3 => vk::Format::R32G32B32_SNORM,
            VertexFormat::Float16x3 => vk::Format::R16G16B16_SFLOAT,
            VertexFormat::Float32x3 => vk::Format::R32G32B32_SFLOAT,
            VertexFormat::Uint8x4 => vk::Format::R8G8B8A8_UINT,
            VertexFormat::Uint16x4 => vk::Format::R16G16B16A16_UINT,
            VertexFormat::Uint32x4 => vk::Format::R32G32B32A32_UINT,
            VertexFormat::Sint8x4 => vk::Format::R8G8B8A8_SINT,
            VertexFormat::Sint16x4 => vk::Format::R16G16B16A16_SINT,
            VertexFormat::Sint32x4 => vk::Format::R32G32B32A32_SINT,
            VertexFormat::Unorm8x4 => vk::Format::R8G8B8A8_UNORM,
            VertexFormat::Unorm16x4 => vk::Format::R16G16B16A16_UNORM,
            // VertexFormat::Unorm32x4 => vk::Format::R32G32B32A32_UNORM,
            VertexFormat::Snorm8x4 => vk::Format::R8G8B8A8_SNORM,
            VertexFormat::Snorm16x4 => vk::Format::R16G16B16A16_SNORM,
            // VertexFormat::Snorm32x4 => vk::Format::R32G32B32A32_SNORM,
            VertexFormat::Float16x4 => vk::Format::R16G16B16A16_SFLOAT,
            VertexFormat::Float32x4 => vk::Format::R32G32B32A32_SFLOAT,
            _ => return None,
        })
    }
}

impl AshFrom<CompareFunction> for vk::CompareOp {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn ash_from(compare: CompareFunction) -> Self {
        match compare {
            CompareFunction::Never => vk::CompareOp::NEVER,
            CompareFunction::Less => vk::CompareOp::LESS,
            CompareFunction::Equal => vk::CompareOp::EQUAL,
            CompareFunction::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
            CompareFunction::Greater => vk::CompareOp::GREATER,
            CompareFunction::NotEqual => vk::CompareOp::NOT_EQUAL,
            CompareFunction::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
            CompareFunction::Always => vk::CompareOp::ALWAYS,
        }
    }
}

impl AshFrom<BlendFactor> for vk::BlendFactor {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn ash_from(factor: BlendFactor) -> Self {
        match factor {
            BlendFactor::Zero => vk::BlendFactor::ZERO,
            BlendFactor::One => vk::BlendFactor::ONE,
            BlendFactor::SrcColor => vk::BlendFactor::SRC_COLOR,
            BlendFactor::OneMinusSrcColor => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
            BlendFactor::SrcAlpha => vk::BlendFactor::SRC_ALPHA,
            BlendFactor::OneMinusSrcAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            BlendFactor::DstColor => vk::BlendFactor::DST_COLOR,
            BlendFactor::OneMinusDstColor => vk::BlendFactor::ONE_MINUS_DST_COLOR,
            BlendFactor::DstAlpha => vk::BlendFactor::DST_ALPHA,
            BlendFactor::OneMinusDstAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
            BlendFactor::SrcAlphaSaturated => vk::BlendFactor::SRC_ALPHA_SATURATE,
            BlendFactor::BlendColor => vk::BlendFactor::CONSTANT_COLOR,
            BlendFactor::OneMinusBlendColor => vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR,
        }
    }
}

impl AshFrom<BlendOp> for vk::BlendOp {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn ash_from(op: BlendOp) -> Self {
        match op {
            BlendOp::Add => vk::BlendOp::ADD,
            BlendOp::Subtract => vk::BlendOp::SUBTRACT,
            BlendOp::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
            BlendOp::Min => vk::BlendOp::MIN,
            BlendOp::Max => vk::BlendOp::MAX,
        }
    }
}

impl AshFrom<WriteMask> for vk::ColorComponentFlags {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn ash_from(mask: WriteMask) -> Self {
        let mut flags = vk::ColorComponentFlags::empty();
        if mask.contains(WriteMask::RED) {
            flags |= vk::ColorComponentFlags::R;
        }
        if mask.contains(WriteMask::GREEN) {
            flags |= vk::ColorComponentFlags::G;
        }
        if mask.contains(WriteMask::BLUE) {
            flags |= vk::ColorComponentFlags::B;
        }
        if mask.contains(WriteMask::ALPHA) {
            flags |= vk::ColorComponentFlags::A;
        }
        flags
    }
}

impl AshFrom<Filter> for ash::vk::Filter {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn ash_from(filter: Filter) -> Self {
        match filter {
            Filter::Nearest => ash::vk::Filter::NEAREST,
            Filter::Linear => ash::vk::Filter::LINEAR,
        }
    }
}

impl AshFrom<MipMapMode> for ash::vk::SamplerMipmapMode {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn ash_from(mode: MipMapMode) -> Self {
        match mode {
            MipMapMode::Nearest => ash::vk::SamplerMipmapMode::NEAREST,
            MipMapMode::Linear => ash::vk::SamplerMipmapMode::LINEAR,
        }
    }
}

impl AshFrom<AddressMode> for ash::vk::SamplerAddressMode {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn ash_from(mode: AddressMode) -> Self {
        match mode {
            AddressMode::Repeat => ash::vk::SamplerAddressMode::REPEAT,
            AddressMode::MirrorRepeat => ash::vk::SamplerAddressMode::MIRRORED_REPEAT,
            AddressMode::ClampToEdge => ash::vk::SamplerAddressMode::CLAMP_TO_EDGE,
        }
    }
}

impl AshFrom<ShaderStage> for ash::vk::ShaderStageFlags {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn ash_from(stage: ShaderStage) -> Self {
        match stage {
            ShaderStage::Vertex => ash::vk::ShaderStageFlags::VERTEX,
            ShaderStage::Fragment => ash::vk::ShaderStageFlags::FRAGMENT,
            ShaderStage::Compute => ash::vk::ShaderStageFlags::COMPUTE,
        }
    }
}

impl AshFrom<ShaderStages> for ash::vk::ShaderStageFlags {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn ash_from(stages: ShaderStages) -> Self {
        from_flags!(ShaderStages => vk::ShaderStageFlags, [
            VERTEX => VERTEX,
            FRAGMENT => FRAGMENT,
            COMPUTE => COMPUTE,
        ], stages)
    }
}

impl AshFrom<PipelineStage> for ash::vk::PipelineStageFlags {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn ash_from(stage: PipelineStage) -> Self {
        match stage {
            PipelineStage::DrawIndirect => ash::vk::PipelineStageFlags::DRAW_INDIRECT,
            PipelineStage::VertexInput => ash::vk::PipelineStageFlags::VERTEX_INPUT,
            PipelineStage::VertexShader => ash::vk::PipelineStageFlags::VERTEX_SHADER,
            PipelineStage::EarlyFragmentTest => ash::vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            PipelineStage::FragmentShader => ash::vk::PipelineStageFlags::FRAGMENT_SHADER,
            PipelineStage::LateFragmentTest => ash::vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            PipelineStage::ColorOutput => ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            PipelineStage::ComputeShader => ash::vk::PipelineStageFlags::COMPUTE_SHADER,
            PipelineStage::Transfer => ash::vk::PipelineStageFlags::TRANSFER,
        }
    }
}

impl AshFrom<PipelineStages> for ash::vk::PipelineStageFlags {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn ash_from(stages: PipelineStages) -> Self {
        from_flags!(PipelineStages => vk::PipelineStageFlags, [
            DRAW_INDIRECT => DRAW_INDIRECT,
            VERTEX_INPUT => VERTEX_INPUT,
            VERTEX_SHADER => VERTEX_SHADER,
            EARLY_FRAGMENT_TEST => EARLY_FRAGMENT_TESTS,
            FRAGMENT_SHADER => FRAGMENT_SHADER,
            LATE_FRAGMENT_TEST => LATE_FRAGMENT_TESTS,
            COLOR_OUTPUT => COLOR_ATTACHMENT_OUTPUT,
            COMPUTE_SHADER => COMPUTE_SHADER,
            TRANSFER => TRANSFER,
        ], stages)
    }
}

impl AshFrom<FrontFace> for ash::vk::FrontFace {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn ash_from(face: FrontFace) -> Self {
        match face {
            FrontFace::Clockwise => ash::vk::FrontFace::CLOCKWISE,
            FrontFace::CounterClockwise => ash::vk::FrontFace::COUNTER_CLOCKWISE,
        }
    }
}

impl AshFrom<Culling> for ash::vk::CullModeFlags {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn ash_from(generic: Culling) -> Self {
        match generic {
            Culling::None => ash::vk::CullModeFlags::NONE,
            Culling::Front => ash::vk::CullModeFlags::FRONT,
            Culling::Back => ash::vk::CullModeFlags::BACK,
        }
    }
}

impl AshFrom<Extent2> for ash::vk::Extent2D {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn ash_from(generic: Extent2) -> Self {
        ash::vk::Extent2D {
            width: generic.width(),
            height: generic.height(),
        }
    }
}

impl AshFrom<Extent3> for ash::vk::Extent3D {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn ash_from(generic: Extent3) -> Self {
        ash::vk::Extent3D {
            width: generic.width(),
            height: generic.height(),
            depth: generic.depth(),
        }
    }
}

impl AshFrom<ImageExtent> for ash::vk::Extent3D {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn ash_from(generic: ImageExtent) -> Self {
        match generic {
            ImageExtent::D1(extent) => ash::vk::Extent3D {
                width: extent.width(),
                height: 1,
                depth: 1,
            },
            ImageExtent::D2(extent) => ash::vk::Extent3D {
                width: extent.width(),
                height: extent.height(),
                depth: 1,
            },
            ImageExtent::D3(extent) => ash::vk::Extent3D {
                width: extent.width(),
                height: extent.height(),
                depth: extent.depth(),
            },
        }
    }
}

impl AshFrom<Offset2> for ash::vk::Offset2D {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn ash_from(generic: Offset2) -> Self {
        ash::vk::Offset2D {
            x: generic.x(),
            y: generic.y(),
        }
    }
}

impl AshFrom<Offset3> for ash::vk::Offset3D {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn ash_from(generic: Offset3) -> Self {
        ash::vk::Offset3D {
            x: generic.x(),
            y: generic.y(),
            z: generic.z(),
        }
    }
}

impl AshFrom<ComponentSwizzle> for ash::vk::ComponentSwizzle {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn ash_from(component: ComponentSwizzle) -> Self {
        match component {
            ComponentSwizzle::Identity => ash::vk::ComponentSwizzle::IDENTITY,
            ComponentSwizzle::Zero => ash::vk::ComponentSwizzle::ZERO,
            ComponentSwizzle::One => ash::vk::ComponentSwizzle::ONE,
            ComponentSwizzle::R => ash::vk::ComponentSwizzle::R,
            ComponentSwizzle::G => ash::vk::ComponentSwizzle::G,
            ComponentSwizzle::B => ash::vk::ComponentSwizzle::B,
            ComponentSwizzle::A => ash::vk::ComponentSwizzle::A,
        }
    }
}

impl AshFrom<Swizzle> for ash::vk::ComponentMapping {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn ash_from(swizzle: Swizzle) -> Self {
        ash::vk::ComponentMapping {
            r: swizzle.r.into_ash(),
            g: swizzle.g.into_ash(),
            b: swizzle.b.into_ash(),
            a: swizzle.a.into_ash(),
        }
    }
}
