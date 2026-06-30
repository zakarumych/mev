use crate::{
    generic::{
        AddressMode, BlendFactor, BlendOp, BufferUsage, CompareFunction, ComponentSwizzle, Culling,
        Extent2, Extent3, FamilyCapabilities, Filter, FrontFace, ImageExtent, ImageUsage,
        MipMapMode, Offset2, Offset3, PipelineStage, PipelineStages, PixelFormat, QueueFlags,
        ShaderStage, ShaderStages, Swizzle, VertexFormat, WriteMask,
    },
    mat, ArgumentKind, Blend, BlendDesc, ClearColor, Extent1, PrimitiveTopology, VertexStepMode,
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

pub trait FromWgpu<A> {
    fn from_wgpu(wgpu: A) -> Self;
}

pub trait WgpuInto<T> {
    fn wgpu_into(self) -> T;
}

impl<T, A> WgpuInto<T> for A
where
    T: FromWgpu<A>,
{
    #[inline]
    fn wgpu_into(self) -> T {
        T::from_wgpu(self)
    }
}

pub trait WgpuFrom<T> {
    fn wgpu_from(mev: T) -> Self;
}

pub trait IntoWgpu<A> {
    fn into_wgpu(self) -> A;
}

impl<A, T> IntoWgpu<A> for T
where
    A: WgpuFrom<T>,
{
    #[inline]
    fn into_wgpu(self) -> A {
        A::wgpu_from(self)
    }
}

pub trait TryFromWgpu<T>: Sized {
    fn try_from_wgpu(t: T) -> Option<Self>;
}

pub trait TryWgpuInto<T> {
    fn try_wgpu_into(self) -> Option<T>;
}

impl<T, U> TryWgpuInto<U> for T
where
    U: TryFromWgpu<T>,
{
    #[inline]
    fn try_wgpu_into(self) -> Option<U> {
        U::try_from_wgpu(self)
    }
}

pub trait TryWgpuFrom<T>: Sized {
    fn try_wgpu_from(t: T) -> Option<Self>;
}

pub trait TryIntoWgpu<T> {
    fn try_into_wgpu(self) -> Option<T>;
}

impl<T, U> TryIntoWgpu<U> for T
where
    U: TryWgpuFrom<T>,
{
    #[inline]
    fn try_into_wgpu(self) -> Option<U> {
        U::try_wgpu_from(self)
    }
}

impl FromWgpu<wgpu::BufferUsages> for BufferUsage {
    fn from_wgpu(wgpu: wgpu::BufferUsages) -> Self {
        let mut flags = BufferUsage::empty();

        if wgpu.contains(wgpu::BufferUsages::MAP_READ) {
            flags |= BufferUsage::HOST_READ;
        }

        if wgpu.contains(wgpu::BufferUsages::MAP_WRITE) {
            flags |= BufferUsage::HOST_WRITE;
        }

        if wgpu.contains(wgpu::BufferUsages::COPY_SRC) {
            flags |= BufferUsage::TRANSFER_SRC;
        }

        if wgpu.contains(wgpu::BufferUsages::COPY_DST) {
            flags |= BufferUsage::TRANSFER_DST;
        }

        if wgpu.contains(wgpu::BufferUsages::INDEX) {
            flags |= BufferUsage::INDEX;
        }

        if wgpu.contains(wgpu::BufferUsages::VERTEX) {
            flags |= BufferUsage::VERTEX;
        }

        if wgpu.contains(wgpu::BufferUsages::UNIFORM) {
            flags |= BufferUsage::UNIFORM;
        }

        if wgpu.contains(wgpu::BufferUsages::STORAGE) {
            flags |= BufferUsage::STORAGE;
        }

        if wgpu.contains(wgpu::BufferUsages::INDIRECT) {
            flags |= BufferUsage::INDIRECT;
        }

        flags
    }
}

impl WgpuFrom<BufferUsage> for wgpu::BufferUsages {
    fn wgpu_from(mev: BufferUsage) -> Self {
        let mut flags = wgpu::BufferUsages::empty();

        if mev.contains(BufferUsage::HOST_READ) {
            flags |= wgpu::BufferUsages::MAP_READ;
        }

        if mev.contains(BufferUsage::HOST_WRITE) {
            flags |= wgpu::BufferUsages::MAP_WRITE;
        }

        if mev.contains(BufferUsage::TRANSFER_SRC) {
            flags |= wgpu::BufferUsages::COPY_SRC;
        }

        if mev.contains(BufferUsage::TRANSFER_DST) {
            flags |= wgpu::BufferUsages::COPY_DST;
        }

        if mev.contains(BufferUsage::INDEX) {
            flags |= wgpu::BufferUsages::INDEX;
        }

        if mev.contains(BufferUsage::VERTEX) {
            flags |= wgpu::BufferUsages::VERTEX;
        }

        if mev.contains(BufferUsage::UNIFORM) {
            flags |= wgpu::BufferUsages::UNIFORM;
        }

        if mev.contains(BufferUsage::STORAGE) {
            flags |= wgpu::BufferUsages::STORAGE;
        }

        if mev.contains(BufferUsage::INDIRECT) {
            flags |= wgpu::BufferUsages::INDIRECT;
        }

        flags
    }
}

impl FromWgpu<wgpu::TextureUsages> for ImageUsage {
    fn from_wgpu(wgpu: wgpu::TextureUsages) -> Self {
        let mut flags = ImageUsage::empty();

        if wgpu.contains(wgpu::TextureUsages::COPY_SRC) {
            flags |= ImageUsage::TRANSFER_SRC;
        }

        if wgpu.contains(wgpu::TextureUsages::COPY_DST) {
            flags |= ImageUsage::TRANSFER_DST;
        }

        if wgpu.contains(wgpu::TextureUsages::TEXTURE_BINDING) {
            flags |= ImageUsage::SAMPLED;
        }

        if wgpu.contains(wgpu::TextureUsages::STORAGE_BINDING) {
            flags |= ImageUsage::STORAGE;
        }

        if wgpu.contains(wgpu::TextureUsages::RENDER_ATTACHMENT) {
            flags |= ImageUsage::TARGET;
        }

        flags
    }
}

impl WgpuFrom<ImageUsage> for wgpu::TextureUsages {
    fn wgpu_from(mev: ImageUsage) -> Self {
        let mut flags = wgpu::TextureUsages::empty();
        if mev.contains(ImageUsage::TRANSFER_SRC) {
            flags |= wgpu::TextureUsages::COPY_SRC;
        }
        if mev.contains(ImageUsage::TRANSFER_DST) {
            flags |= wgpu::TextureUsages::COPY_DST;
        }
        if mev.contains(ImageUsage::SAMPLED) {
            flags |= wgpu::TextureUsages::TEXTURE_BINDING;
        }
        if mev.contains(ImageUsage::STORAGE) {
            flags |= wgpu::TextureUsages::STORAGE_BINDING;
        }
        if mev.contains(ImageUsage::TARGET) {
            flags |= wgpu::TextureUsages::RENDER_ATTACHMENT;
        }
        flags
    }
}

impl WgpuFrom<PixelFormat> for wgpu::TextureFormat {
    fn wgpu_from(mev: PixelFormat) -> Self {
        use wgpu::TextureFormat::*;
        match mev {
            PixelFormat::R8Unorm => R8Unorm,
            PixelFormat::R8Snorm => R8Snorm,
            PixelFormat::R8Uint => R8Uint,
            PixelFormat::R8Sint => R8Sint,
            PixelFormat::R16Unorm => R16Unorm,
            PixelFormat::R16Snorm => R16Snorm,
            PixelFormat::R16Uint => R16Uint,
            PixelFormat::R16Sint => R16Sint,
            PixelFormat::R16Float => R16Float,
            PixelFormat::R32Uint => R32Uint,
            PixelFormat::R32Sint => R32Sint,
            PixelFormat::R32Float => R32Float,
            PixelFormat::Rg8Unorm => Rg8Unorm,
            PixelFormat::Rg8Snorm => Rg8Snorm,
            PixelFormat::Rg8Uint => Rg8Uint,
            PixelFormat::Rg8Sint => Rg8Sint,
            PixelFormat::Rg16Unorm => Rg16Unorm,
            PixelFormat::Rg16Snorm => Rg16Snorm,
            PixelFormat::Rg16Uint => Rg16Uint,
            PixelFormat::Rg16Sint => Rg16Sint,
            PixelFormat::Rg16Float => Rg16Float,
            PixelFormat::Rg32Uint => Rg32Uint,
            PixelFormat::Rg32Sint => Rg32Sint,
            PixelFormat::Rg32Float => Rg32Float,
            PixelFormat::Rgba8Unorm => Rgba8Unorm,
            PixelFormat::Rgba8Snorm => Rgba8Snorm,
            PixelFormat::Rgba8Uint => Rgba8Uint,
            PixelFormat::Rgba8Sint => Rgba8Sint,
            PixelFormat::Rgba8Srgb => Rgba8UnormSrgb,
            PixelFormat::Rgba16Unorm => Rgba16Unorm,
            PixelFormat::Rgba16Snorm => Rgba16Snorm,
            PixelFormat::Rgba16Uint => Rgba16Uint,
            PixelFormat::Rgba16Sint => Rgba16Sint,
            PixelFormat::Rgba16Float => Rgba16Float,
            PixelFormat::Rgba32Uint => Rgba32Uint,
            PixelFormat::Rgba32Sint => Rgba32Sint,
            PixelFormat::Rgba32Float => Rgba32Float,
            PixelFormat::Bgra8Unorm => Bgra8Unorm,
            PixelFormat::Bgra8Srgb => Bgra8UnormSrgb,
            PixelFormat::D16Unorm => Depth16Unorm,
            PixelFormat::D32Float => Depth32Float,
            PixelFormat::S8Uint => Stencil8,
            PixelFormat::D24UnormS8Uint => Depth24PlusStencil8,
            PixelFormat::D32FloatS8Uint => Depth32FloatStencil8,
            PixelFormat::Bc1RgbUnorm => Bc1RgbaUnorm,
            PixelFormat::Bc1RgbSrgb => Bc1RgbaUnormSrgb,
            PixelFormat::Bc1RgbaUnorm => Bc1RgbaUnorm,
            PixelFormat::Bc1RgbaSrgb => Bc1RgbaUnormSrgb,
            PixelFormat::Bc2Unorm => Bc2RgbaUnorm,
            PixelFormat::Bc2Srgb => Bc2RgbaUnormSrgb,
            PixelFormat::Bc3Unorm => Bc3RgbaUnorm,
            PixelFormat::Bc3Srgb => Bc3RgbaUnormSrgb,
            PixelFormat::Bc4Unorm => Bc4RUnorm,
            PixelFormat::Bc4Snorm => Bc4RSnorm,
            PixelFormat::Bc5Unorm => Bc5RgUnorm,
            PixelFormat::Bc5Snorm => Bc5RgSnorm,
            _ => panic!("Unsupported mev::PixelFormat: {:?}", mev),
        }
    }
}

impl FromWgpu<wgpu::TextureFormat> for PixelFormat {
    fn from_wgpu(wgpu: wgpu::TextureFormat) -> Self {
        use wgpu::TextureFormat::*;

        match wgpu {
            R8Unorm => PixelFormat::R8Unorm,
            R8Snorm => PixelFormat::R8Snorm,
            R8Uint => PixelFormat::R8Uint,
            R8Sint => PixelFormat::R8Sint,
            R16Unorm => PixelFormat::R16Unorm,
            R16Snorm => PixelFormat::R16Snorm,
            R16Uint => PixelFormat::R16Uint,
            R16Sint => PixelFormat::R16Sint,
            R16Float => PixelFormat::R16Float,
            R32Uint => PixelFormat::R32Uint,
            R32Sint => PixelFormat::R32Sint,
            R32Float => PixelFormat::R32Float,
            Rg8Unorm => PixelFormat::Rg8Unorm,
            Rg8Snorm => PixelFormat::Rg8Snorm,
            Rg8Uint => PixelFormat::Rg8Uint,
            Rg8Sint => PixelFormat::Rg8Sint,
            Rg16Unorm => PixelFormat::Rg16Unorm,
            Rg16Snorm => PixelFormat::Rg16Snorm,
            Rg16Uint => PixelFormat::Rg16Uint,
            Rg16Sint => PixelFormat::Rg16Sint,
            Rg16Float => PixelFormat::Rg16Float,
            Rg32Uint => PixelFormat::Rg32Uint,
            Rg32Sint => PixelFormat::Rg32Sint,
            Rg32Float => PixelFormat::Rg32Float,
            Rgba8Unorm => PixelFormat::Rgba8Unorm,
            Rgba8Snorm => PixelFormat::Rgba8Snorm,
            Rgba8Uint => PixelFormat::Rgba8Uint,
            Rgba8Sint => PixelFormat::Rgba8Sint,
            Rgba8UnormSrgb => PixelFormat::Rgba8Srgb,
            Rgba16Unorm => PixelFormat::Rgba16Unorm,
            Rgba16Snorm => PixelFormat::Rgba16Snorm,
            Rgba16Uint => PixelFormat::Rgba16Uint,
            Rgba16Sint => PixelFormat::Rgba16Sint,
            Rgba16Float => PixelFormat::Rgba16Float,
            Rgba32Uint => PixelFormat::Rgba32Uint,
            Rgba32Sint => PixelFormat::Rgba32Sint,
            Rgba32Float => PixelFormat::Rgba32Float,
            Bgra8Unorm => PixelFormat::Bgra8Unorm,
            Bgra8UnormSrgb => PixelFormat::Bgra8Srgb,
            Depth16Unorm => PixelFormat::D16Unorm,
            Depth32Float => PixelFormat::D32Float,
            Stencil8 => PixelFormat::S8Uint,
            Depth24PlusStencil8 => PixelFormat::D24UnormS8Uint,
            Depth32FloatStencil8 => PixelFormat::D32FloatS8Uint,
            Bc1RgbaUnorm => PixelFormat::Bc1RgbUnorm,
            Bc1RgbaUnormSrgb => PixelFormat::Bc1RgbSrgb,
            Bc1RgbaUnorm => PixelFormat::Bc1RgbaUnorm,
            Bc1RgbaUnormSrgb => PixelFormat::Bc1RgbaSrgb,
            Bc2RgbaUnorm => PixelFormat::Bc2Unorm,
            Bc2RgbaUnormSrgb => PixelFormat::Bc2Srgb,
            Bc3RgbaUnorm => PixelFormat::Bc3Unorm,
            Bc3RgbaUnormSrgb => PixelFormat::Bc3Srgb,
            Bc4RUnorm => PixelFormat::Bc4Unorm,
            Bc4RSnorm => PixelFormat::Bc4Snorm,
            Bc5RgUnorm => PixelFormat::Bc5Unorm,
            Bc5RgSnorm => PixelFormat::Bc5Snorm,
            _ => panic!("Unsupported wgpu::TextureFormat: {:?}", wgpu),
        }
    }
}

impl WgpuFrom<VertexFormat> for wgpu::VertexFormat {
    fn wgpu_from(mev: VertexFormat) -> Self {
        match mev {
            VertexFormat::Uint8 => wgpu::VertexFormat::Uint8,
            VertexFormat::Uint16 => wgpu::VertexFormat::Uint16,
            VertexFormat::Uint32 => wgpu::VertexFormat::Uint32,
            VertexFormat::Sint8 => wgpu::VertexFormat::Sint8,
            VertexFormat::Sint16 => wgpu::VertexFormat::Sint16,
            VertexFormat::Sint32 => wgpu::VertexFormat::Sint32,
            VertexFormat::Unorm8 => wgpu::VertexFormat::Unorm8,
            VertexFormat::Unorm16 => wgpu::VertexFormat::Unorm16,
            VertexFormat::Snorm8 => wgpu::VertexFormat::Snorm8,
            VertexFormat::Snorm16 => wgpu::VertexFormat::Snorm16,
            VertexFormat::Float16 => wgpu::VertexFormat::Float16,
            VertexFormat::Float32 => wgpu::VertexFormat::Float32,

            VertexFormat::Uint8x2 => wgpu::VertexFormat::Uint8x2,
            VertexFormat::Uint16x2 => wgpu::VertexFormat::Uint16x2,
            VertexFormat::Uint32x2 => wgpu::VertexFormat::Uint32x2,
            VertexFormat::Sint8x2 => wgpu::VertexFormat::Sint8x2,
            VertexFormat::Sint16x2 => wgpu::VertexFormat::Sint16x2,
            VertexFormat::Sint32x2 => wgpu::VertexFormat::Sint32x2,
            VertexFormat::Unorm8x2 => wgpu::VertexFormat::Unorm8x2,
            VertexFormat::Unorm16x2 => wgpu::VertexFormat::Unorm16x2,
            VertexFormat::Snorm8x2 => wgpu::VertexFormat::Snorm8x2,
            VertexFormat::Snorm16x2 => wgpu::VertexFormat::Snorm16x2,
            VertexFormat::Float16x2 => wgpu::VertexFormat::Float16x2,
            VertexFormat::Float32x2 => wgpu::VertexFormat::Float32x2,

            VertexFormat::Uint32x3 => wgpu::VertexFormat::Uint32x3,
            VertexFormat::Sint32x3 => wgpu::VertexFormat::Sint32x3,
            VertexFormat::Float32x3 => wgpu::VertexFormat::Float32x3,

            VertexFormat::Uint8x4 => wgpu::VertexFormat::Uint8x4,
            VertexFormat::Uint16x4 => wgpu::VertexFormat::Uint16x4,
            VertexFormat::Uint32x4 => wgpu::VertexFormat::Uint32x4,
            VertexFormat::Sint8x4 => wgpu::VertexFormat::Sint8x4,
            VertexFormat::Sint16x4 => wgpu::VertexFormat::Sint16x4,
            VertexFormat::Sint32x4 => wgpu::VertexFormat::Sint32x4,
            VertexFormat::Unorm8x4 => wgpu::VertexFormat::Unorm8x4,
            VertexFormat::Unorm16x4 => wgpu::VertexFormat::Unorm16x4,
            VertexFormat::Snorm8x4 => wgpu::VertexFormat::Snorm8x4,
            VertexFormat::Snorm16x4 => wgpu::VertexFormat::Snorm16x4,
            VertexFormat::Float16x4 => wgpu::VertexFormat::Float16x4,
            VertexFormat::Float32x4 => wgpu::VertexFormat::Float32x4,

            _ => panic!("Unsupported mev::VertexFormat: {:?}", mev),
        }
    }
}

impl WgpuFrom<ImageExtent> for wgpu::Extent3d {
    fn wgpu_from(mev: ImageExtent) -> Self {
        match mev {
            ImageExtent::D1(extent) => wgpu::Extent3d {
                width: extent.width(),
                height: 1,
                depth_or_array_layers: 1,
            },
            ImageExtent::D2(extent) => wgpu::Extent3d {
                width: extent.width(),
                height: extent.height(),
                depth_or_array_layers: 1,
            },
            ImageExtent::D3(extent) => wgpu::Extent3d {
                width: extent.width(),
                height: extent.height(),
                depth_or_array_layers: extent.depth(),
            },
        }
    }
}

impl FromWgpu<(wgpu::TextureDimension, wgpu::Extent3d)> for ImageExtent {
    fn from_wgpu((dim, extent): (wgpu::TextureDimension, wgpu::Extent3d)) -> Self {
        match dim {
            wgpu::TextureDimension::D1 => ImageExtent::D1(Extent1::new(extent.width)),
            wgpu::TextureDimension::D2 => {
                ImageExtent::D2(Extent2::new(extent.width, extent.height))
            }
            wgpu::TextureDimension::D3 => ImageExtent::D3(Extent3::new(
                extent.width,
                extent.height,
                extent.depth_or_array_layers,
            )),
        }
    }
}

impl WgpuFrom<ShaderStages> for wgpu::ShaderStages {
    fn wgpu_from(mev: ShaderStages) -> Self {
        let mut flags = wgpu::ShaderStages::empty();

        if mev.contains(ShaderStages::VERTEX) {
            flags |= wgpu::ShaderStages::VERTEX;
        }

        if mev.contains(ShaderStages::FRAGMENT) {
            flags |= wgpu::ShaderStages::FRAGMENT;
        }

        if mev.contains(ShaderStages::COMPUTE) {
            flags |= wgpu::ShaderStages::COMPUTE;
        }

        flags
    }
}

impl WgpuFrom<ArgumentKind> for wgpu::BindingType {
    fn wgpu_from(mev: ArgumentKind) -> Self {
        match mev {
            ArgumentKind::UniformBuffer => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            ArgumentKind::StorageBuffer => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            ArgumentKind::SampledImage => wgpu::BindingType::Texture {
                multisampled: false,
                view_dimension: wgpu::TextureViewDimension::D2,
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
            },
            ArgumentKind::StorageImage => wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::ReadWrite,
                format: wgpu::TextureFormat::Rgba8Unorm,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            ArgumentKind::Sampler => {
                wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)
            }
        }
    }
}

impl WgpuFrom<ClearColor> for wgpu::Color {
    fn wgpu_from(mev: ClearColor) -> Self {
        wgpu::Color {
            r: mev.0.into(),
            g: mev.1.into(),
            b: mev.2.into(),
            a: mev.3.into(),
        }
    }
}

impl WgpuFrom<VertexStepMode> for wgpu::VertexStepMode {
    fn wgpu_from(mev: VertexStepMode) -> Self {
        match mev {
            VertexStepMode::Vertex => wgpu::VertexStepMode::Vertex,
            VertexStepMode::Instance { rate: 1 } => wgpu::VertexStepMode::Instance,
            _ => panic!("Unsupported mev::VertexStepMode: {:?}", mev),
        }
    }
}

impl WgpuFrom<BlendOp> for wgpu::BlendOperation {
    fn wgpu_from(mev: BlendOp) -> Self {
        match mev {
            BlendOp::Add => wgpu::BlendOperation::Add,
            BlendOp::Subtract => wgpu::BlendOperation::Subtract,
            BlendOp::ReverseSubtract => wgpu::BlendOperation::ReverseSubtract,
            BlendOp::Min => wgpu::BlendOperation::Min,
            BlendOp::Max => wgpu::BlendOperation::Max,
        }
    }
}

impl WgpuFrom<BlendFactor> for wgpu::BlendFactor {
    fn wgpu_from(mev: BlendFactor) -> Self {
        match mev {
            BlendFactor::Zero => wgpu::BlendFactor::Zero,
            BlendFactor::One => wgpu::BlendFactor::One,
            BlendFactor::SrcColor => wgpu::BlendFactor::Src,
            BlendFactor::OneMinusSrcColor => wgpu::BlendFactor::OneMinusSrc,
            BlendFactor::DstColor => wgpu::BlendFactor::Dst,
            BlendFactor::OneMinusDstColor => wgpu::BlendFactor::OneMinusDst,
            BlendFactor::SrcAlpha => wgpu::BlendFactor::SrcAlpha,
            BlendFactor::OneMinusSrcAlpha => wgpu::BlendFactor::OneMinusSrcAlpha,
            BlendFactor::DstAlpha => wgpu::BlendFactor::DstAlpha,
            BlendFactor::OneMinusDstAlpha => wgpu::BlendFactor::OneMinusDstAlpha,
            BlendFactor::SrcAlphaSaturated => wgpu::BlendFactor::SrcAlphaSaturated,
        }
    }
}

impl WgpuFrom<Blend> for wgpu::BlendComponent {
    fn wgpu_from(mev: Blend) -> Self {
        wgpu::BlendComponent {
            src_factor: mev.src.into_wgpu(),
            dst_factor: mev.dst.into_wgpu(),
            operation: mev.op.into_wgpu(),
        }
    }
}

impl WgpuFrom<WriteMask> for wgpu::ColorWrites {
    fn wgpu_from(mev: WriteMask) -> Self {
        let mut writes = wgpu::ColorWrites::empty();

        if mev.contains(WriteMask::RED) {
            writes |= wgpu::ColorWrites::RED;
        }
        if mev.contains(WriteMask::GREEN) {
            writes |= wgpu::ColorWrites::GREEN;
        }
        if mev.contains(WriteMask::BLUE) {
            writes |= wgpu::ColorWrites::BLUE;
        }
        if mev.contains(WriteMask::ALPHA) {
            writes |= wgpu::ColorWrites::ALPHA;
        }
        writes
    }
}

impl WgpuFrom<CompareFunction> for wgpu::CompareFunction {
    fn wgpu_from(mev: CompareFunction) -> Self {
        match mev {
            CompareFunction::Never => wgpu::CompareFunction::Never,
            CompareFunction::Less => wgpu::CompareFunction::Less,
            CompareFunction::Equal => wgpu::CompareFunction::Equal,
            CompareFunction::LessEqual => wgpu::CompareFunction::LessEqual,
            CompareFunction::Greater => wgpu::CompareFunction::Greater,
            CompareFunction::NotEqual => wgpu::CompareFunction::NotEqual,
            CompareFunction::GreaterEqual => wgpu::CompareFunction::GreaterEqual,
            CompareFunction::Always => wgpu::CompareFunction::Always,
        }
    }
}

impl WgpuFrom<PrimitiveTopology> for wgpu::PrimitiveTopology {
    fn wgpu_from(mev: PrimitiveTopology) -> Self {
        match mev {
            PrimitiveTopology::Point => wgpu::PrimitiveTopology::PointList,
            PrimitiveTopology::Line => wgpu::PrimitiveTopology::LineList,
            PrimitiveTopology::Triangle => wgpu::PrimitiveTopology::TriangleList,
        }
    }
}

impl WgpuFrom<FrontFace> for wgpu::FrontFace {
    fn wgpu_from(mev: FrontFace) -> Self {
        match mev {
            FrontFace::Clockwise => wgpu::FrontFace::Cw,
            FrontFace::CounterClockwise => wgpu::FrontFace::Ccw,
        }
    }
}

impl WgpuFrom<Culling> for Option<wgpu::Face> {
    fn wgpu_from(mev: Culling) -> Self {
        match mev {
            Culling::None => None,
            Culling::Front => Some(wgpu::Face::Front),
            Culling::Back => Some(wgpu::Face::Back),
        }
    }
}

impl WgpuFrom<AddressMode> for wgpu::AddressMode {
    fn wgpu_from(mev: AddressMode) -> Self {
        match mev {
            AddressMode::Repeat => wgpu::AddressMode::Repeat,
            AddressMode::MirrorRepeat => wgpu::AddressMode::MirrorRepeat,
            AddressMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        }
    }
}

impl WgpuFrom<Filter> for wgpu::FilterMode {
    fn wgpu_from(mev: Filter) -> Self {
        match mev {
            Filter::Nearest => wgpu::FilterMode::Nearest,
            Filter::Linear => wgpu::FilterMode::Linear,
        }
    }
}

impl WgpuFrom<MipMapMode> for wgpu::MipmapFilterMode {
    fn wgpu_from(mev: MipMapMode) -> Self {
        match mev {
            MipMapMode::Nearest => wgpu::MipmapFilterMode::Nearest,
            MipMapMode::Linear => wgpu::MipmapFilterMode::Linear,
        }
    }
}
