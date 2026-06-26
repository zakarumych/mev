use core::fmt;
use std::error::Error;

use crate::{backend::Image, DataType, ScalarType, VectorSize, VertexAttributes};

use super::{arguments::ArgumentGroupLayout, PixelFormat, Shader, VertexFormat};

/// Load operation for an attachment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoadOp<T> {
    /// Load the attachment from memory.
    Load,

    /// Clear the attachment with the given color.
    /// Sets all pixels to specified color before rendering.
    ///
    /// This is safe and fast option is old content is not needed.
    Clear(T),

    /// Do not load the attachment.
    /// Pixel color will be undefined before rendering.
    ///
    /// This is fastest option.
    /// But usable only if all pixels will be written during rendering,
    /// or undefined color in not written pixels is acceptable.
    DontCare,
}

/// Store operation for an attachment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StoreOp {
    /// Store pixels to memory.
    Store,

    /// Do not store pixels to memory.
    ///
    /// Avoids writing to memory if the content is not needed.
    DontCare,
}

/// Clear value for color attachment.
///
/// It is simple RGBA, only channels present in the image format are used.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClearColor(pub f32, pub f32, pub f32, pub f32);

impl ClearColor {
    pub const BLACK: Self = ClearColor(0.0, 0.0, 0.0, 1.0);
    pub const TRANSPARENT: Self = ClearColor(0.0, 0.0, 0.0, 0.0);
    pub const WHITE: Self = ClearColor(1.0, 1.0, 1.0, 1.0);
    pub const RED: Self = ClearColor(1.0, 0.0, 0.0, 1.0);
    pub const GREEN: Self = ClearColor(0.0, 1.0, 0.0, 1.0);
    pub const BLUE: Self = ClearColor(0.0, 0.0, 1.0, 1.0);
    pub const YELLOW: Self = ClearColor(1.0, 1.0, 0.0, 1.0);
    pub const CYAN: Self = ClearColor(0.0, 1.0, 1.0, 1.0);
    pub const MAGENTA: Self = ClearColor(1.0, 0.0, 1.0, 1.0);
    pub const GRAY: Self = ClearColor(0.5, 0.5, 0.5, 1.0);
    pub const DARK_GRAY: Self = ClearColor(0.25, 0.25, 0.25, 1.0);
    pub const LIGHT_GRAY: Self = ClearColor(0.75, 0.75, 0.75, 1.0);
}

/// Clear value for depth-stencil attachment.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct ClearDepthStencil {
    /// Clear value for depth component.
    pub depth: f32,

    /// Clear value for stencil component.
    pub stencil: u32,
}

/// Description of an attachment in a render pass.
#[derive(Clone, Copy)]
pub struct AttachmentDesc<'a, T> {
    /// Image to use as attachment.
    pub image: &'a Image,

    /// Load operation for the attachment.
    pub load: LoadOp<T>,

    /// Store operation for the attachment.
    pub store: StoreOp,
}

impl<'a, T> AttachmentDesc<'a, T> {
    /// Create a new attachment description from image reference.
    pub fn new(image: &'a Image) -> Self {
        AttachmentDesc {
            image,
            load: LoadOp::Load,
            store: StoreOp::Store,
        }
    }

    /// Set load operation to do not load.
    pub fn no_load(mut self) -> Self {
        self.load = LoadOp::DontCare;
        self
    }

    /// Set load operation to clear with specified color.
    pub fn clear(mut self, color: T) -> Self {
        self.load = LoadOp::Clear(color);
        self
    }

    /// Set store operation to do not store.
    pub fn no_store(mut self) -> Self {
        self.store = StoreOp::DontCare;
        self
    }

    /// Set specified load operation.
    pub fn load_op(mut self, op: LoadOp<T>) -> Self {
        self.load = op;
        self
    }

    /// Set specified store operation.
    pub fn store_op(mut self, op: StoreOp) -> Self {
        self.store = op;
        self
    }
}

impl<'a, T> From<&'a Image> for AttachmentDesc<'a, T> {
    fn from(image: &'a Image) -> Self {
        AttachmentDesc::new(image)
    }
}

/// Description of a render pass.
#[derive(Clone, Copy, Default)]
pub struct RenderPassDesc<'a> {
    /// Name of the render pass.
    pub name: &'a str,

    /// Color attachments of the render pass.
    pub color_attachments: &'a [AttachmentDesc<'a, ClearColor>],

    /// Depth-stencil attachment of the render pass.
    pub depth_stencil_attachment: Option<AttachmentDesc<'a, ClearDepthStencil>>,
}

impl<'a> RenderPassDesc<'a> {
    /// Create a new render pass description
    pub const fn new() -> Self {
        RenderPassDesc {
            name: "",
            color_attachments: &[],
            depth_stencil_attachment: None,
        }
    }

    /// Set name of the render pass.
    pub fn name(mut self, name: &'a str) -> Self {
        self.name = name;
        self
    }

    /// Set color attachments of the render pass.
    pub fn color_attachments(mut self, attachments: &'a [AttachmentDesc<'a, ClearColor>]) -> Self {
        self.color_attachments = attachments;
        self
    }

    /// Set depth-stencil attachment of the render pass.
    pub fn depth_stencil_attachment(
        mut self,
        attachment: AttachmentDesc<'a, ClearDepthStencil>,
    ) -> Self {
        self.depth_stencil_attachment = Some(attachment);
        self
    }
}

/// Describes single vertex attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VertexAttributeDesc {
    /// Index of the buffer that contains vertex data.
    pub buffer_index: u32,

    /// Vertex attribute format.
    pub format: VertexFormat,

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

    /// Vertex buffer layouts.
    pub vertex_layouts: Vec<VertexLayoutDesc>,

    /// Vertex attributes.
    pub vertex_attributes: Vec<VertexAttributeDesc>,

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
pub enum PipelineError {
    /// Invalid shader entry point.
    InvalidShaderEntry,

    /// Generic failure with a message.
    Failure(String),
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PipelineError::InvalidShaderEntry => write!(f, "Invalid shader entry point"),
            PipelineError::Failure(msg) => write!(f, "Failure: {}", msg),
        }
    }
}

impl Error for PipelineError {}

/// Trait to be derived for structures that contain vertex attributes.
///
/// The derived implementation will provide
/// 1. a static description of the vertex layout
/// 2. a structure containing iterators of `VertexAttributeDesc` for each field in the structure.
///    Iterators are needed because some types map to multiple vertex attributes (e.g. `mat4x4<f32>` maps to 4 `vec4<f32>` attributes).
///
/// When deriving this trait step mode for `LAYOUT` is controlled by attributes.
/// `VertexStepMode::Vertex` is default,
/// `#[mev(instance = N)]` sets `VertexStepMode::Instance { rate: N }`,
/// and `#[mev(constant)]` sets `VertexStepMode::Constant`.
///
/// Fields that exist in the structure but are not used as vertex attributes can be marked with `#[mev(skip)]`.
///
/// All unskipped fields must implement `VertexAttributes` trait, which is a fixed set of types that can be used as vertex attributes.
/// Those types are scalars - i8, u8, i16, u16, i32, u32, f32,
/// vectors - vec2, vec3, vec4 of those scalars,
/// and matrices - mat2x2, mat3x3, mat4x4 etc of those scalars.
///
/// Note that structure does not have to be `#[repr(C)]` or `#[repr(packed)]`
/// because the derived implementation will calculate offsets of each field and use them in the vertex attribute descriptions.
pub trait VertexBinding {
    const LAYOUT: VertexLayoutDesc;

    /// Structure containing iterators of `VertexAttributeDesc` for each field in `Self`.
    type AttributeDescs;

    fn descs(buffer_index: u32) -> Self::AttributeDescs;
}
