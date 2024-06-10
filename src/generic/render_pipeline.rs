use core::fmt;
use std::error::Error;

use crate::backend::CreatePipelineErrorKind;

use super::{arguments::ArgumentGroupLayout, PixelFormat, Shader, VertexFormat};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VertexAttributeDesc {
    pub format: VertexFormat,
    pub offset: u32,
    pub buffer_index: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VertexStepMode {
    Vertex,
    Instance { rate: u32 },
    Constant,
}

impl Default for VertexStepMode {
    fn default() -> Self {
        VertexStepMode::Vertex
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VertexLayoutDesc {
    pub stride: u32,
    pub step_mode: VertexStepMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrimitiveTopology {
    Point,
    Line,
    Triangle,
}

impl Default for PrimitiveTopology {
    fn default() -> Self {
        PrimitiveTopology::Triangle
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ColorTargetDesc {
    pub format: PixelFormat,
    pub blend: Option<BlendDesc>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlendDesc {
    pub mask: WriteMask,
    pub color: Blend,
    pub alpha: Blend,
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Blend {
    pub op: BlendOp,
    pub src: BlendFactor,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlendFactor {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstColor,
    OneMinusDstColor,
    DstAlpha,
    OneMinusDstAlpha,
    SrcAlphaSaturated,
    BlendColor,
    OneMinusBlendColor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DepthStencilDesc {
    pub format: PixelFormat,
    pub write_enabled: bool,
    pub compare: CompareFunction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompareFunction {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FrontFace {
    Clockwise,
    CounterClockwise,
}

impl Default for FrontFace {
    #[inline(always)]
    fn default() -> Self {
        FrontFace::Clockwise
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Culling {
    None,
    Front,
    Back,
}

impl Default for Culling {
    #[inline(always)]
    fn default() -> Self {
        Culling::Back
    }
}

pub struct RenderPipelineDesc<'a> {
    pub name: &'a str,
    pub vertex_shader: Shader<'a>,
    pub vertex_attributes: Vec<VertexAttributeDesc>,
    pub vertex_layouts: Vec<VertexLayoutDesc>,
    pub primitive_topology: PrimitiveTopology,
    pub raster: Option<RasterDesc<'a>>,
    pub constants: usize,
    pub arguments: &'a [ArgumentGroupLayout<'a>],
}

pub struct RasterDesc<'a> {
    pub fragment_shader: Option<Shader<'a>>,
    pub color_targets: Vec<ColorTargetDesc>,
    pub depth_stencil: Option<DepthStencilDesc>,
    pub front_face: FrontFace,
    pub culling: Culling,
}

#[derive(Debug)]
pub struct CreatePipelineError(pub(crate) CreatePipelineErrorKind);

impl fmt::Display for CreatePipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl Error for CreatePipelineError {}
