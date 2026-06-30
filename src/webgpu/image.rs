use std::{fmt, hash, rc::Rc};

use crate::{
    generic::{ImageDesc, ImageExtent, ImageUsage, PixelFormat, ViewDesc},
    ArgumentKind, Automatic, Sampled, Storage,
};

use super::{
    arguments::ArgumentsField,
    from::{IntoWgpu, WgpuInto},
    Device,
};

pub(super) struct ImageInner {
    texture: Rc<wgpu::Texture>,
    view: wgpu::TextureView,
    format: PixelFormat,
    extent: ImageExtent,
    layers: u32,
    levels: u32,
    usage: ImageUsage,
}

#[derive(Clone)]
pub struct Image {
    view: wgpu::TextureView,
    base_layer: u32,
    layers: u32,
    base_level: u32,
    levels: u32,
}

impl Image {
    pub(super) fn new(view: wgpu::TextureView, layers: u32, levels: u32) -> Self {
        Image {
            view,
            base_layer: 0,
            layers,
            base_level: 0,
            levels,
        }
    }

    pub(super) fn wgpu_view(&self) -> &wgpu::TextureView {
        &self.view
    }

    pub(super) fn base_layer(&self) -> u32 {
        self.base_layer
    }

    pub(super) fn base_level(&self) -> u32 {
        self.base_level
    }
}

impl fmt::Debug for Image {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Image").field("view", &self.view).finish()
    }
}

impl PartialEq for Image {
    fn eq(&self, other: &Self) -> bool {
        self.view == other.view
    }
}

impl Eq for Image {}

impl hash::Hash for Image {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.view.hash(state);
    }
}

impl crate::traits::Resource for Image {}

#[hidden_trait::expose]
impl crate::traits::Image for Image {
    fn format(&self) -> PixelFormat {
        self.view.texture().format().wgpu_into()
    }

    fn extent(&self) -> ImageExtent {
        (self.view.texture().dimension(), self.view.texture().size()).wgpu_into()
    }

    fn layers(&self) -> u32 {
        self.layers
    }

    fn levels(&self) -> u32 {
        self.levels
    }

    fn usage(&self) -> ImageUsage {
        self.view.texture().usage().wgpu_into()
    }

    fn view(&self, device: &Device, desc: ViewDesc) -> Image {
        let view_desc = wgpu::TextureViewDescriptor {
            label: None,
            format: Some(desc.format.into_wgpu()),
            dimension: None,
            usage: None,
            aspect: wgpu::TextureAspect::All,
            base_mip_level: desc.base_level + self.base_level,
            mip_level_count: Some(desc.levels),
            base_array_layer: desc.base_layer + self.base_layer,
            array_layer_count: Some(desc.layers),
        };

        let view = self.view.texture().create_view(&view_desc);

        Image {
            view,
            base_layer: desc.base_layer + self.base_layer,
            layers: desc.layers,
            base_level: desc.base_level + self.base_level,
            levels: desc.levels,
        }
    }

    fn detached(&self) -> bool {
        false
    }
}

impl ArgumentsField<Automatic> for Image {
    const KIND: ArgumentKind = crate::generic::ArgumentKind::SampledImage;
    const SIZE: usize = 1;

    fn as_binding_resource(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::TextureView(self.wgpu_view())
    }
}

impl ArgumentsField<Sampled> for Image {
    const KIND: ArgumentKind = crate::generic::ArgumentKind::SampledImage;
    const SIZE: usize = 1;

    fn as_binding_resource(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::TextureView(self.wgpu_view())
    }
}

impl ArgumentsField<Storage> for Image {
    const KIND: ArgumentKind = crate::generic::ArgumentKind::StorageImage;
    const SIZE: usize = 1;

    fn as_binding_resource(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::TextureView(self.wgpu_view())
    }
}
