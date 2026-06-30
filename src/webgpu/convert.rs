use crate::generic::{
    AddressMode, ArgumentKind, BlendFactor, BlendOp, BufferUsage, CompareFunction, Culling,
    Filter, FrontFace, ImageUsage, MipMapMode, PixelFormat, PrimitiveTopology, ShaderStages,
    StoreOp, VertexFormat, VertexStepMode, WriteMask,
};

pub fn pixel_format(f: PixelFormat) -> wgpu::TextureFormat {
    use wgpu::TextureFormat::*;
    match f {
        PixelFormat::R8Unorm => R8Unorm,
        PixelFormat::R8Snorm => R8Snorm,
        PixelFormat::R8Uint => R8Uint,
        PixelFormat::R8Sint => R8Sint,
        PixelFormat::R8Srgb => panic!("R8Srgb not supported in wgpu"),
        PixelFormat::R16Unorm => R16Unorm,
        PixelFormat::R16Snorm => R16Snorm,
        PixelFormat::R16Uint => R16Uint,
        PixelFormat::R16Sint => R16Sint,
        PixelFormat::R16Float => R16Float,
        PixelFormat::R32Unorm => panic!("R32Unorm not supported in wgpu"),
        PixelFormat::R32Snorm => panic!("R32Snorm not supported in wgpu"),
        PixelFormat::R32Uint => R32Uint,
        PixelFormat::R32Sint => R32Sint,
        PixelFormat::R32Float => R32Float,
        PixelFormat::Rg8Unorm => Rg8Unorm,
        PixelFormat::Rg8Snorm => Rg8Snorm,
        PixelFormat::Rg8Uint => Rg8Uint,
        PixelFormat::Rg8Sint => Rg8Sint,
        PixelFormat::Rg8Srgb => panic!("Rg8Srgb not supported in wgpu"),
        PixelFormat::Rg16Unorm => Rg16Unorm,
        PixelFormat::Rg16Snorm => Rg16Snorm,
        PixelFormat::Rg16Uint => Rg16Uint,
        PixelFormat::Rg16Sint => Rg16Sint,
        PixelFormat::Rg16Float => Rg16Float,
        PixelFormat::Rg32Unorm => panic!("Rg32Unorm not supported in wgpu"),
        PixelFormat::Rg32Snorm => panic!("Rg32Snorm not supported in wgpu"),
        PixelFormat::Rg32Uint => Rg32Uint,
        PixelFormat::Rg32Sint => Rg32Sint,
        PixelFormat::Rg32Float => Rg32Float,
        PixelFormat::Rgb8Unorm => panic!("Rgb8Unorm not supported in wgpu"),
        PixelFormat::Rgb8Snorm => panic!("Rgb8Snorm not supported in wgpu"),
        PixelFormat::Rgb8Uint => panic!("Rgb8Uint not supported in wgpu"),
        PixelFormat::Rgb8Sint => panic!("Rgb8Sint not supported in wgpu"),
        PixelFormat::Rgb8Srgb => panic!("Rgb8Srgb not supported in wgpu"),
        PixelFormat::Rgb16Unorm => panic!("Rgb16Unorm not supported in wgpu"),
        PixelFormat::Rgb16Snorm => panic!("Rgb16Snorm not supported in wgpu"),
        PixelFormat::Rgb16Uint => panic!("Rgb16Uint not supported in wgpu"),
        PixelFormat::Rgb16Sint => panic!("Rgb16Sint not supported in wgpu"),
        PixelFormat::Rgb16Float => panic!("Rgb16Float not supported in wgpu"),
        PixelFormat::Rgb32Unorm => panic!("Rgb32Unorm not supported in wgpu"),
        PixelFormat::Rgb32Snorm => panic!("Rgb32Snorm not supported in wgpu"),
        PixelFormat::Rgb32Uint => panic!("Rgb32Uint not supported in wgpu"),
        PixelFormat::Rgb32Sint => panic!("Rgb32Sint not supported in wgpu"),
        PixelFormat::Rgb32Float => panic!("Rgb32Float not supported in wgpu"),
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
        PixelFormat::Rgba32Unorm => panic!("Rgba32Unorm not supported in wgpu"),
        PixelFormat::Rgba32Snorm => panic!("Rgba32Snorm not supported in wgpu"),
        PixelFormat::Rgba32Uint => Rgba32Uint,
        PixelFormat::Rgba32Sint => Rgba32Sint,
        PixelFormat::Rgba32Float => Rgba32Float,
        PixelFormat::Bgr8Unorm => panic!("Bgr8Unorm not supported in wgpu"),
        PixelFormat::Bgr8Snorm => panic!("Bgr8Snorm not supported in wgpu"),
        PixelFormat::Bgr8Uint => panic!("Bgr8Uint not supported in wgpu"),
        PixelFormat::Bgr8Sint => panic!("Bgr8Sint not supported in wgpu"),
        PixelFormat::Bgr8Srgb => panic!("Bgr8Srgb not supported in wgpu"),
        PixelFormat::Bgra8Unorm => Bgra8Unorm,
        PixelFormat::Bgra8Snorm => panic!("Bgra8Snorm not supported in wgpu"),
        PixelFormat::Bgra8Uint => panic!("Bgra8Uint not supported in wgpu"),
        PixelFormat::Bgra8Sint => panic!("Bgra8Sint not supported in wgpu"),
        PixelFormat::Bgra8Srgb => Bgra8UnormSrgb,
        PixelFormat::D16Unorm => Depth16Unorm,
        PixelFormat::D32Float => Depth32Float,
        PixelFormat::S8Uint => Stencil8,
        PixelFormat::D16UnormS8Uint => panic!("D16UnormS8Uint not supported in wgpu"),
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
    }
}

pub fn buffer_usages(u: BufferUsage) -> wgpu::BufferUsages {
    let mut out = wgpu::BufferUsages::empty();
    if u.contains(BufferUsage::TRANSFER_SRC) {
        out |= wgpu::BufferUsages::COPY_SRC;
    }
    if u.contains(BufferUsage::TRANSFER_DST) {
        out |= wgpu::BufferUsages::COPY_DST;
    }
    if u.contains(BufferUsage::UNIFORM) {
        out |= wgpu::BufferUsages::UNIFORM;
    }
    if u.contains(BufferUsage::STORAGE) {
        out |= wgpu::BufferUsages::STORAGE;
    }
    if u.contains(BufferUsage::INDEX) {
        out |= wgpu::BufferUsages::INDEX;
    }
    if u.contains(BufferUsage::VERTEX) {
        out |= wgpu::BufferUsages::VERTEX;
    }
    if u.contains(BufferUsage::INDIRECT) {
        out |= wgpu::BufferUsages::INDIRECT;
    }
    if u.contains(BufferUsage::HOST_READ) {
        out |= wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
    }
    if u.contains(BufferUsage::HOST_WRITE) {
        out |= wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC;
    }
    out
}

pub fn image_usages(u: ImageUsage) -> wgpu::TextureUsages {
    let mut out = wgpu::TextureUsages::empty();
    if u.contains(ImageUsage::TRANSFER_SRC) {
        out |= wgpu::TextureUsages::COPY_SRC;
    }
    if u.contains(ImageUsage::TRANSFER_DST) {
        out |= wgpu::TextureUsages::COPY_DST;
    }
    if u.contains(ImageUsage::SAMPLED) {
        out |= wgpu::TextureUsages::TEXTURE_BINDING;
    }
    if u.contains(ImageUsage::STORAGE) {
        out |= wgpu::TextureUsages::STORAGE_BINDING;
    }
    if u.contains(ImageUsage::TARGET) {
        out |= wgpu::TextureUsages::RENDER_ATTACHMENT;
    }
    out
}

pub fn vertex_format(f: VertexFormat) -> wgpu::VertexFormat {
    use wgpu::VertexFormat::*;
    match f {
        VertexFormat::Uint8x2 => Uint8x2,
        VertexFormat::Uint8x4 => Uint8x4,
        VertexFormat::Sint8x2 => Sint8x2,
        VertexFormat::Sint8x4 => Sint8x4,
        VertexFormat::Unorm8x2 => Unorm8x2,
        VertexFormat::Unorm8x4 => Unorm8x4,
        VertexFormat::Snorm8x2 => Snorm8x2,
        VertexFormat::Snorm8x4 => Snorm8x4,
        VertexFormat::Uint16x2 => Uint16x2,
        VertexFormat::Uint16x4 => Uint16x4,
        VertexFormat::Sint16x2 => Sint16x2,
        VertexFormat::Sint16x4 => Sint16x4,
        VertexFormat::Unorm16x2 => Unorm16x2,
        VertexFormat::Unorm16x4 => Unorm16x4,
        VertexFormat::Snorm16x2 => Snorm16x2,
        VertexFormat::Snorm16x4 => Snorm16x4,
        VertexFormat::Float16x2 => Float16x2,
        VertexFormat::Float16x4 => Float16x4,
        VertexFormat::Float32 => Float32,
        VertexFormat::Float32x2 => Float32x2,
        VertexFormat::Float32x3 => Float32x3,
        VertexFormat::Float32x4 => Float32x4,
        VertexFormat::Uint32 => Uint32,
        VertexFormat::Uint32x2 => Uint32x2,
        VertexFormat::Uint32x3 => Uint32x3,
        VertexFormat::Uint32x4 => Uint32x4,
        VertexFormat::Sint32 => Sint32,
        VertexFormat::Sint32x2 => Sint32x2,
        VertexFormat::Sint32x3 => Sint32x3,
        VertexFormat::Sint32x4 => Sint32x4,
    }
}

pub fn filter_mode(f: Filter) -> wgpu::FilterMode {
    match f {
        Filter::Nearest => wgpu::FilterMode::Nearest,
        Filter::Linear => wgpu::FilterMode::Linear,
    }
}

pub fn mipmap_filter(m: MipMapMode) -> wgpu::FilterMode {
    match m {
        MipMapMode::Nearest => wgpu::FilterMode::Nearest,
        MipMapMode::Linear => wgpu::FilterMode::Linear,
    }
}

pub fn address_mode(a: AddressMode) -> wgpu::AddressMode {
    match a {
        AddressMode::Repeat => wgpu::AddressMode::Repeat,
        AddressMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
        AddressMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        AddressMode::ClampToBorder => wgpu::AddressMode::ClampToBorder,
    }
}

pub fn compare_function(c: CompareFunction) -> wgpu::CompareFunction {
    match c {
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

pub fn blend_factor(f: BlendFactor) -> wgpu::BlendFactor {
    match f {
        BlendFactor::Zero => wgpu::BlendFactor::Zero,
        BlendFactor::One => wgpu::BlendFactor::One,
        BlendFactor::SrcColor => wgpu::BlendFactor::Src,
        BlendFactor::OneMinusSrcColor => wgpu::BlendFactor::OneMinusSrc,
        BlendFactor::SrcAlpha => wgpu::BlendFactor::SrcAlpha,
        BlendFactor::OneMinusSrcAlpha => wgpu::BlendFactor::OneMinusSrcAlpha,
        BlendFactor::DstColor => wgpu::BlendFactor::Dst,
        BlendFactor::OneMinusDstColor => wgpu::BlendFactor::OneMinusDst,
        BlendFactor::DstAlpha => wgpu::BlendFactor::DstAlpha,
        BlendFactor::OneMinusDstAlpha => wgpu::BlendFactor::OneMinusDstAlpha,
        BlendFactor::SrcAlphaSaturated => wgpu::BlendFactor::SrcAlphaSaturated,
    }
}

pub fn blend_op(o: BlendOp) -> wgpu::BlendOperation {
    match o {
        BlendOp::Add => wgpu::BlendOperation::Add,
        BlendOp::Subtract => wgpu::BlendOperation::Subtract,
        BlendOp::ReverseSubtract => wgpu::BlendOperation::ReverseSubtract,
        BlendOp::Min => wgpu::BlendOperation::Min,
        BlendOp::Max => wgpu::BlendOperation::Max,
    }
}

pub fn write_mask(m: WriteMask) -> wgpu::ColorWrites {
    let mut out = wgpu::ColorWrites::empty();
    if m.contains(WriteMask::RED) {
        out |= wgpu::ColorWrites::RED;
    }
    if m.contains(WriteMask::GREEN) {
        out |= wgpu::ColorWrites::GREEN;
    }
    if m.contains(WriteMask::BLUE) {
        out |= wgpu::ColorWrites::BLUE;
    }
    if m.contains(WriteMask::ALPHA) {
        out |= wgpu::ColorWrites::ALPHA;
    }
    out
}

pub fn primitive_topology(t: PrimitiveTopology) -> wgpu::PrimitiveTopology {
    match t {
        PrimitiveTopology::Point => wgpu::PrimitiveTopology::PointList,
        PrimitiveTopology::Line => wgpu::PrimitiveTopology::LineList,
        PrimitiveTopology::Triangle => wgpu::PrimitiveTopology::TriangleList,
    }
}

pub fn front_face(f: FrontFace) -> wgpu::FrontFace {
    match f {
        FrontFace::Clockwise => wgpu::FrontFace::Cw,
        FrontFace::CounterClockwise => wgpu::FrontFace::Ccw,
    }
}

pub fn cull_mode(c: Culling) -> Option<wgpu::Face> {
    match c {
        Culling::None => None,
        Culling::Front => Some(wgpu::Face::Front),
        Culling::Back => Some(wgpu::Face::Back),
    }
}

pub fn shader_stages(s: ShaderStages) -> wgpu::ShaderStages {
    let mut out = wgpu::ShaderStages::empty();
    if s.contains(ShaderStages::VERTEX) {
        out |= wgpu::ShaderStages::VERTEX;
    }
    if s.contains(ShaderStages::FRAGMENT) {
        out |= wgpu::ShaderStages::FRAGMENT;
    }
    if s.contains(ShaderStages::COMPUTE) {
        out |= wgpu::ShaderStages::COMPUTE;
    }
    out
}

pub fn binding_type(kind: ArgumentKind) -> wgpu::BindingType {
    match kind {
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
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        ArgumentKind::StorageImage => wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::ReadWrite,
            format: wgpu::TextureFormat::Rgba8Unorm,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        ArgumentKind::Sampler => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
    }
}

pub fn vertex_step_mode(m: VertexStepMode) -> wgpu::VertexStepMode {
    match m {
        VertexStepMode::Vertex => wgpu::VertexStepMode::Vertex,
        VertexStepMode::Instance { .. } => wgpu::VertexStepMode::Instance,
        VertexStepMode::Constant => wgpu::VertexStepMode::Vertex,
    }
}

pub fn store_op(s: StoreOp) -> wgpu::StoreOp {
    match s {
        StoreOp::Store => wgpu::StoreOp::Store,
        StoreOp::DontCare => wgpu::StoreOp::Discard,
    }
}
