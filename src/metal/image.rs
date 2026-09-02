use std::{
    hash::{Hash, Hasher},
    num::NonZero,
};

use mev_proc::hidden_trait;
use objc2::{
    msg_send,
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_foundation::{NSRange, NSUInteger};
use objc2_metal::{
    MTLComputeCommandEncoder, MTLRenderCommandEncoder, MTLTexture, MTLTextureSwizzle,
    MTLTextureSwizzleChannels, MTLTextureType,
};

use crate::{
    ImageUsage,
    generic::{
        ArgumentKind, Automatic, ComponentMap, Extent1, Extent2, Extent3, ImageArrayN, ImageExtent,
        ImageN, OutOfMemory, PixelFormat, Sampled, Storage, Swizzle, ViewDesc,
    },
};

use super::{
    Device,
    arguments::ArgumentsField,
    from::{MetalInto, TryIntoMetal, TryMetalInto},
};

#[derive(Clone, Debug)]
pub struct Image {
    texture: Retained<ProtocolObject<dyn MTLTexture>>,
}

impl PartialEq for Image {
    fn eq(&self, other: &Self) -> bool {
        self.texture == other.texture
    }
}

impl Eq for Image {}

impl Hash for Image {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Retained::as_ptr(&self.texture).hash(state);
    }
}

unsafe impl Send for Image {}
unsafe impl Sync for Image {}

impl Image {
    pub(super) fn new(texture: Retained<ProtocolObject<dyn MTLTexture>>) -> Self {
        Image { texture }
    }

    pub(super) fn metal(&self) -> &ProtocolObject<dyn MTLTexture> {
        self.texture.as_ref()
    }
}

impl crate::traits::Resource for Image {}

#[hidden_trait]
impl crate::traits::Image for Image {
    fn format(&self) -> PixelFormat {
        self.texture.pixelFormat().expect_metal_into()
    }

    fn extent(&self) -> ImageExtent {
        match self.texture.textureType() {
            MTLTextureType::Type1D | MTLTextureType::Type1DArray => {
                let width = self.texture.width();
                ImageExtent::D1(Extent1::new(width as u32))
            }
            MTLTextureType::Type2D | MTLTextureType::Type2DArray => {
                let width = self.texture.width();
                let height = self.texture.height();
                ImageExtent::D2(Extent2::new(width as u32, height as u32))
            }
            MTLTextureType::Type2DMultisample => unimplemented!(),
            MTLTextureType::Type2DMultisampleArray => unimplemented!(),
            MTLTextureType::TypeCube => unimplemented!(),
            MTLTextureType::TypeCubeArray => unimplemented!(),
            MTLTextureType::Type3D => {
                let width = self.texture.width();
                let height = self.texture.height();
                let depth = self.texture.depth();
                ImageExtent::D3(Extent3::new(width as u32, height as u32, depth as u32))
            }
            MTLTextureType::TypeTextureBuffer => {
                unreachable!("TypeTextureBuffer is not a valid texture type for an image")
            }
            _ => unreachable!("Unknown texture type"),
        }
    }

    fn layers(&self) -> u32 {
        self.texture.arrayLength() as u32
    }

    fn levels(&self) -> u32 {
        self.texture.mipmapLevelCount() as u32
    }

    fn usage(&self) -> ImageUsage {
        self.texture.usage().metal_into()
    }

    fn view(&self, _device: &Device, desc: ViewDesc) -> Image {
        let pixel_format = desc.format.expect_into_metal();
        let root_texture = self
            .texture
            .parentTexture()
            .unwrap_or_else(|| self.texture.clone());

        let level_count = (desc.levels.end - desc.levels.start) as NSUInteger;
        let layer_count = (desc.layers.end - desc.layers.start) as NSUInteger;

        if desc.swizzle == Swizzle::IDENTITY {
            if desc.layers.start == 0 && desc.levels.start == 0 {
                let texture = root_texture
                    .newTextureViewWithPixelFormat(desc.format.expect_into_metal())
                    .expect("Failed to create texture view");
                Image { texture }
            } else {
                let base_layer = self.texture.parentRelativeSlice() as u32 + desc.layers.start;
                let base_level = self.texture.parentRelativeLevel() as u32 + desc.levels.start;

                let texture = unsafe {
                    root_texture.newTextureViewWithPixelFormat_textureType_levels_slices(
                        pixel_format,
                        self.texture.textureType(),
                        NSRange::new(base_level as NSUInteger, level_count),
                        NSRange::new(base_layer as NSUInteger, layer_count),
                    )
                }
                .expect("Failed to create texture view");
                Image { texture }
            }
        } else {
            let base_layer = self.texture.parentRelativeSlice() as u32 + desc.layers.start;
            let base_level = self.texture.parentRelativeLevel() as u32 + desc.levels.start;
            let swizzle = self.texture.swizzle();

            let new_swizzle = swizzle_mul(swizzle, desc.swizzle);

            let texture = unsafe {
                root_texture.newTextureViewWithPixelFormat_textureType_levels_slices_swizzle(
                    pixel_format,
                    self.texture.textureType(),
                    NSRange::new(base_level as NSUInteger, level_count),
                    NSRange::new(base_layer as NSUInteger, layer_count),
                    new_swizzle,
                )
            }
            .expect("Failed to create texture view");

            Image { texture }
        }
    }

    fn detached(&self) -> bool {
        let count: NSUInteger = unsafe { msg_send![Retained::as_ptr(&self.texture), retainCount] };
        count == 1
    }
}

fn swizzle_mul(channels: MTLTextureSwizzleChannels, rhs: Swizzle) -> MTLTextureSwizzleChannels {
    use ComponentMap::*;

    let mul = |rhs: ComponentMap| -> MTLTextureSwizzle {
        match rhs {
            Zero => MTLTextureSwizzle::Zero,
            One => MTLTextureSwizzle::One,
            R => channels.red,
            G => channels.green,
            B => channels.blue,
            A => channels.alpha,
        }
    };

    MTLTextureSwizzleChannels {
        red: mul(rhs.r),
        green: mul(rhs.g),
        blue: mul(rhs.b),
        alpha: mul(rhs.a),
    }
}

impl ArgumentsField<Automatic> for Image {
    const KIND: ArgumentKind = ArgumentKind::SampledImage;
    const SIZE: Option<NonZero<usize>> = None;

    #[inline(always)]
    fn bind_vertex(&self, slot: u32, encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>) {
        unsafe {
            encoder.setVertexTexture_atIndex(Some(&self.texture), slot as NSUInteger);
        }
    }

    #[inline(always)]
    fn bind_fragment(&self, slot: u32, encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>) {
        unsafe {
            encoder.setFragmentTexture_atIndex(Some(&self.texture), slot as NSUInteger);
        }
    }

    #[inline(always)]
    fn bind_compute(&self, slot: u32, encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
        unsafe {
            encoder.setTexture_atIndex(Some(&self.texture), slot as NSUInteger);
        }
    }
}

impl ArgumentsField<Sampled> for Image {
    const KIND: ArgumentKind = ArgumentKind::SampledImage;
    const SIZE: Option<NonZero<usize>> = None;

    #[inline(always)]
    fn bind_vertex(&self, slot: u32, encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>) {
        unsafe {
            encoder.setVertexTexture_atIndex(Some(&self.texture), slot as NSUInteger);
        }
    }

    #[inline(always)]
    fn bind_fragment(&self, slot: u32, encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>) {
        unsafe {
            encoder.setFragmentTexture_atIndex(Some(&self.texture), slot as NSUInteger);
        }
    }

    #[inline(always)]
    fn bind_compute(&self, slot: u32, encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
        unsafe {
            encoder.setTexture_atIndex(Some(&self.texture), slot as NSUInteger);
        }
    }
}

impl ArgumentsField<Storage> for Image {
    const KIND: ArgumentKind = ArgumentKind::StorageImage;
    const SIZE: Option<NonZero<usize>> = None;

    #[inline(always)]
    fn bind_vertex(&self, slot: u32, encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>) {
        unsafe {
            encoder.setVertexTexture_atIndex(Some(&self.texture), slot as NSUInteger);
        }
    }

    #[inline(always)]
    fn bind_fragment(&self, slot: u32, encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>) {
        unsafe {
            encoder.setFragmentTexture_atIndex(Some(&self.texture), slot as NSUInteger);
        }
    }

    #[inline(always)]
    fn bind_compute(&self, slot: u32, encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
        unsafe {
            encoder.setTexture_atIndex(Some(&self.texture), slot as NSUInteger);
        }
    }
}

macro_rules! forward_image_arguments_field {
    ($ty:ident, $marker:ident) => {
        impl<const D: usize> ArgumentsField<$marker> for $ty<D> {
            const KIND: ArgumentKind = <Image as ArgumentsField<$marker>>::KIND;
            const SIZE: Option<NonZero<usize>> = <Image as ArgumentsField<$marker>>::SIZE;

            #[inline(always)]
            fn bind_vertex(
                &self,
                slot: u32,
                encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
            ) {
                <Image as ArgumentsField<$marker>>::bind_vertex(self.as_image(), slot, encoder)
            }

            #[inline(always)]
            fn bind_fragment(
                &self,
                slot: u32,
                encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
            ) {
                <Image as ArgumentsField<$marker>>::bind_fragment(self.as_image(), slot, encoder)
            }

            #[inline(always)]
            fn bind_compute(
                &self,
                slot: u32,
                encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
            ) {
                <Image as ArgumentsField<$marker>>::bind_compute(self.as_image(), slot, encoder)
            }
        }
    };
}

forward_image_arguments_field!(ImageN, Automatic);
forward_image_arguments_field!(ImageN, Sampled);
forward_image_arguments_field!(ImageN, Storage);
forward_image_arguments_field!(ImageArrayN, Automatic);
forward_image_arguments_field!(ImageArrayN, Sampled);
forward_image_arguments_field!(ImageArrayN, Storage);
