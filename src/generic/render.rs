use crate::backend::Image;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoadOp<T> {
    Load,
    Clear(T),
    DontCare,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StoreOp {
    Store,
    DontCare,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClearColor(pub f32, pub f32, pub f32, pub f32);

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct ClearDepthStencil {
    pub depth: f32,
    pub stencil: u32,
}

#[derive(Clone, Copy)]
pub struct AttachmentDesc<'a, T> {
    pub image: &'a Image,
    pub load: LoadOp<T>,
    pub store: StoreOp,
}

impl<'a, T> AttachmentDesc<'a, T> {
    pub fn new(image: &'a Image) -> Self {
        AttachmentDesc {
            image,
            load: LoadOp::Load,
            store: StoreOp::Store,
        }
    }

    pub fn no_load(mut self) -> Self {
        self.load = LoadOp::DontCare;
        self
    }

    pub fn clear(mut self, color: T) -> Self {
        self.load = LoadOp::Clear(color);
        self
    }

    pub fn no_store(mut self) -> Self {
        self.store = StoreOp::DontCare;
        self
    }

    pub fn load_op(mut self, op: LoadOp<T>) -> Self {
        self.load = op;
        self
    }

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

#[derive(Clone, Copy, Default)]
pub struct RenderPassDesc<'a> {
    pub name: &'a str,
    pub color_attachments: &'a [AttachmentDesc<'a, ClearColor>],
    pub depth_stencil_attachment: Option<AttachmentDesc<'a, ClearDepthStencil>>,
}

impl<'a> RenderPassDesc<'a> {
    pub const fn new() -> Self {
        RenderPassDesc {
            name: "",
            color_attachments: &[],
            depth_stencil_attachment: None,
        }
    }

    pub fn name(mut self, name: &'a str) -> Self {
        self.name = name;
        self
    }

    pub fn color_attachments(mut self, attachments: &'a [AttachmentDesc<'a, ClearColor>]) -> Self {
        self.color_attachments = attachments;
        self
    }

    pub fn depth_stencil_attachment(
        mut self,
        attachment: AttachmentDesc<'a, ClearDepthStencil>,
    ) -> Self {
        self.depth_stencil_attachment = Some(attachment);
        self
    }
}
