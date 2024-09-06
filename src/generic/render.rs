use crate::backend::Image;

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
