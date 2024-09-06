use core::fmt;
use std::error::Error;

use crate::backend::CreatePipelineErrorKind;

use super::{arguments::ArgumentGroupLayout, PixelFormat, Shader, VertexFormat};

/// Describes single vertex attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VertexAttributeDesc {
    /// Vertex attribute format.
    pub format: VertexFormat,

    /// Index of the buffer that contains vertex data.
    pub buffer_index: u32,

    /// Offset from the beginning of the vertex data in buffer.
    pub offset: u32,
}

/// Step mode for vertex buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VertexStepMode {
    /// Advance every vertex.
    /// Repeat for each instance.
    Vertex,

    /// Advance every `rate` instances.
    Instance { rate: u32 },

    /// No advancement.
    Constant,
}

impl Default for VertexStepMode {
    fn default() -> Self {
        VertexStepMode::Vertex
    }
}

/// Describes vertex buffer layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VertexLayoutDesc {
    /// Stride in bytes between vertices in the vertex buffer.
    pub stride: u32,

    /// Specifies when to advance to the next vertex (by `stride` bytes).
    pub step_mode: VertexStepMode,
}

/// Describes primitive topology.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PrimitiveTopology {
    /// Vertex buffer contains list of points that will be rasterized.
    Point,

    /// Vertex buffer contains list of lines that will be rasterized.
    Line,

    /// Vertex buffer contains list of triangles that will be rasterized.
    #[default]
    Triangle,
}

/// Describes color render target.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ColorTargetDesc {
    /// Pixel format of the color target.
    ///
    /// It must be a color format.
    pub format: PixelFormat,

    /// Blending options for the color target.
    pub blend: Option<BlendDesc>,
}

/// Describes blending options for color render target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlendDesc {
    /// Masks which channels to write.
    pub mask: WriteMask,

    /// Blending option for color channels.
    pub color: Blend,

    /// Blending option for alpha channel.
    pub alpha: Blend,
}

// By default, do basic alpha blending.
impl Default for BlendDesc {
    fn default() -> Self {
        BlendDesc {
            mask: WriteMask::all(),
            color: Blend {
                op: BlendOp::Add,
                src: BlendFactor::One,
                dst: BlendFactor::OneMinusSrcAlpha,
            },
            alpha: Blend {
                op: BlendOp::Add,
                src: BlendFactor::One,
                dst: BlendFactor::OneMinusSrcAlpha,
            },
        }
    }
}

/// Describes blending option.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Blend {
    /// Blending operation.
    pub op: BlendOp,

    /// Source blending factor.
    /// Source color is multiplied by this factor before blending operation.
    pub src: BlendFactor,

    /// Destination blending factor.
    /// Destination color is multiplied by this factor before blending operation.
    pub dst: BlendFactor,
}

bitflags::bitflags! {
    /// Mask for color blend write.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct WriteMask: u8 {
        const RED = 0x1;
        const GREEN = 0x2;
        const BLUE = 0x4;
        const ALPHA = 0x8;
    }
}

/// Blending factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlendFactor {
    /// 0. color is ignored.
    Zero,

    /// 1. color is used as is.
    One,

    /// Multiply by source color.
    SrcColor,

    /// Multiply by 1 - source color.
    OneMinusSrcColor,

    /// Multiply by source alpha.
    SrcAlpha,

    /// Multiply by 1 - source alpha.
    OneMinusSrcAlpha,

    /// Multiply by destination color.
    DstColor,

    /// Multiply by 1 - destination color.
    OneMinusDstColor,

    /// Multiply by destination alpha.
    DstAlpha,

    /// Multiply by 1 - destination alpha.
    OneMinusDstAlpha,

    /// Multiply by minimum of source and 1 - destination alpha.
    SrcAlphaSaturated,
    // /// Multiply by constant color.
    // BlendColor,

    // /// Multiply by 1 - constant color.
    // OneMinusBlendColor,
}

/// Blending operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlendOp {
    /// Add two values after factor multiplication.
    Add,

    /// Subtract second value from the first after factor multiplication.
    Subtract,

    /// Subtract first value from the second after factor multiplication.
    ReverseSubtract,

    /// Minimum of two values after factor multiplication.
    Min,

    /// Maximum of two values after factor multiplication.
    Max,
}

/// Describes depth-stencil render target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DepthStencilDesc {
    /// Pixel format of the depth-stencil target.
    ///
    /// It must be a depth, stencil or depth-stencil format.
    pub format: PixelFormat,

    /// Flag to enable write operation into the target.
    pub write_enabled: bool,

    /// Comparison function for depth test.
    pub compare: CompareFunction,
}

/// Comparison function for depth test.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompareFunction {
    /// Never pass.
    Never,

    /// Pass if source is less than destination.
    Less,

    /// Pass if source is equal to destination.
    Equal,

    /// Pass if source is less than or equal to destination.
    LessEqual,

    /// Pass if source is greater than destination.
    Greater,

    /// Pass if source is not equal to destination.
    NotEqual,

    /// Pass if source is greater than or equal to destination.
    GreaterEqual,

    /// Always pass.
    Always,
}

/// Front face winding order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FrontFace {
    /// Clockwise winding order.
    #[default]
    Clockwise,

    /// Counter-clockwise winding order.
    CounterClockwise,
}

/// Face culling mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Culling {
    /// No culling.
    None,

    /// Cull front faces.
    Front,

    /// Cull back faces.
    #[default]
    Back,
}

/// Describes render pipeline.
pub struct RenderPipelineDesc<'a> {
    /// Name of the pipeline.
    pub name: &'a str,

    /// Vertex shader.
    pub vertex_shader: Shader<'a>,

    /// Vertex attributes.
    pub vertex_attributes: Vec<VertexAttributeDesc>,

    /// Vertex buffer layouts.
    pub vertex_layouts: Vec<VertexLayoutDesc>,

    /// Primitive topology.
    pub primitive_topology: PrimitiveTopology,

    /// Rasterization options.
    pub raster: Option<RasterDesc<'a>>,

    /// Size of the shader constants in bytes.
    pub constants: usize,

    /// Arguments used by shaders.
    pub arguments: &'a [ArgumentGroupLayout<'a>],
}

/// Describes rasterization options.
pub struct RasterDesc<'a> {
    /// Fragment shader.
    pub fragment_shader: Option<Shader<'a>>,

    /// Color render targets.
    pub color_targets: Vec<ColorTargetDesc>,

    /// Depth-stencil target.
    pub depth_stencil: Option<DepthStencilDesc>,

    /// Front face winding order.
    pub front_face: FrontFace,

    /// Face culling mode.
    pub culling: Culling,
}

/// Error during render pipeline creation.
#[derive(Debug)]
pub struct CreatePipelineError(pub(crate) CreatePipelineErrorKind);

impl fmt::Display for CreatePipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl Error for CreatePipelineError {}
