use std::{
    error::Error,
    fmt,
    ops::{Mul, Range},
};

use super::{format::PixelFormat, Extent1, Extent2, Extent3};

/// Image component swizzle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ComponentSwizzle {
    Identity,
    Zero,
    One,
    R,
    G,
    B,
    A,
}

impl Default for ComponentSwizzle {
    fn default() -> Self {
        ComponentSwizzle::Identity
    }
}

/// Image swizzle for each component.
/// 
/// It is used to remap components of an image in image views.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Swizzle {
    pub r: ComponentSwizzle,
    pub g: ComponentSwizzle,
    pub b: ComponentSwizzle,
    pub a: ComponentSwizzle,
}

impl Swizzle {
    /// Identity swizzle.
    /// i.e. use as-is.
    pub const IDENTITY: Self = Swizzle {
        r: ComponentSwizzle::Identity,
        g: ComponentSwizzle::Identity,
        b: ComponentSwizzle::Identity,
        a: ComponentSwizzle::Identity,
    };

    /// Map all components to R component.
    pub const RRRR: Self = Swizzle {
        r: ComponentSwizzle::R,
        g: ComponentSwizzle::R,
        b: ComponentSwizzle::R,
        a: ComponentSwizzle::R,
    };

    /// Map components R, G and B to one and Alpha to R.
    pub const _111R: Self = Swizzle {
        r: ComponentSwizzle::One,
        g: ComponentSwizzle::One,
        b: ComponentSwizzle::One,
        a: ComponentSwizzle::R,
    };
}

/// Combine two swizzles.
impl Mul for Swizzle {
    type Output = Self;

    #[cfg_attr(feature = "inline-more", inline(always))]
    fn mul(self, rhs: Self) -> Self {
        use ComponentSwizzle::*;

        let mul = |rhs: ComponentSwizzle, i| match rhs {
            Identity => i,
            Zero => Zero,
            One => One,
            R => self.r,
            G => self.g,
            B => self.b,
            A => self.a,
        };

        let r = mul(rhs.r, self.r);
        let g = mul(rhs.g, self.g);
        let b = mul(rhs.b, self.b);
        let a = mul(rhs.a, self.a);

        Swizzle { r, g, b, a }
    }
}

/// Extent of the image.
/// 
/// It can be 1D, 2D or 3D.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageExtent {
    D1(Extent1),
    D2(Extent2),
    D3(Extent3),
}

impl ImageExtent {
    /// Returns the width of the image.
    #[inline(always)]
    pub fn width(&self) -> u32 {
        match self {
            ImageExtent::D1(e) => e.width(),
            ImageExtent::D2(e) => e.width(),
            ImageExtent::D3(e) => e.width(),
        }
    }

    /// Returns the height of the image.
    /// 
    /// If the image is 1D, it returns 1.
    #[inline(always)]
    pub fn height(&self) -> u32 {
        match self {
            ImageExtent::D1(e) => 1,
            ImageExtent::D2(e) => e.height(),
            ImageExtent::D3(e) => e.height(),
        }
    }

    /// Returns the depth of the image.
    /// 
    /// If the image is 1D or 2D, it returns 1.
    #[inline(always)]
    pub fn depth(&self) -> u32 {
        match self {
            ImageExtent::D1(e) => 1,
            ImageExtent::D2(e) => 1,
            ImageExtent::D3(e) => e.depth(),
        }
    }

    /// Convert into `Extent1`, expecting 1D image.
    /// 
    /// # Panics
    /// 
    /// Panics if the image is not 1D.
    #[inline(always)]
    pub fn expect_1d(self) -> Extent1<u32> {
        match self {
            ImageExtent::D1(e) => e,
            _ => panic!("Expected 1D image extent"),
        }
    }

    /// Convert into `Extent2`, expecting 2D image.
    /// 
    /// # Panics
    /// 
    /// Panics if the image is not 2D.
    #[inline(always)]
    pub fn expect_2d(self) -> Extent2<u32> {
        match self {
            ImageExtent::D2(e) => e,
            _ => panic!("Expected 2D image extent"),
        }
    }

    /// Convert into `Extent3`, expecting 3D image.
    /// 
    /// # Panics
    /// 
    /// Panics if the image is not 3D.
    #[inline(always)]
    pub fn expect_3d(self) -> Extent3<u32> {
        match self {
            ImageExtent::D3(e) => e,
            _ => panic!("Expected 3D image extent"),
        }
    }

    /// Convert into `Extent1` from any image extent.
    /// 
    /// Ignores height if the image is 2D or 3D.
    /// Ignores depth if the image is 3D.
    #[inline(always)]
    pub fn into_1d(self) -> Extent1<u32> {
        match self {
            ImageExtent::D1(e) => e,
            ImageExtent::D2(e) => e.to_1d(),
            ImageExtent::D3(e) => e.to_1d(),
        }
    }

    /// Convert into `Extent2` from any image extent.
    /// 
    /// Uses 1 for height if the image is 1D.
    /// Ignores depth is the image is 3D.
    #[inline(always)]
    pub fn into_2d(self) -> Extent2<u32> {
        match self {
            ImageExtent::D1(e) => e.to_2d(),
            ImageExtent::D2(e) => e,
            ImageExtent::D3(e) => e.to_2d(),
        }
    }

    /// Convert into `Extent3` from any image extent.
    /// 
    /// Uses 1 for height if the image is 1D.
    /// Uses 1 for depth if the image is 1D or 2D.
    #[inline(always)]
    pub fn into_3d(self) -> Extent3<u32> {
        match self {
            ImageExtent::D1(e) => e.to_3d(),
            ImageExtent::D2(e) => e.to_3d(),
            ImageExtent::D3(e) => e,
        }
    }
}

impl PartialEq<Extent1> for ImageExtent {
    fn eq(&self, other: &Extent1) -> bool {
        match self {
            ImageExtent::D1(e) => *e == *other,
            ImageExtent::D2(e) => *e == other.to_2d(),
            ImageExtent::D3(e) => *e == other.to_3d(),
        }
    }
}

impl PartialEq<Extent2> for ImageExtent {
    fn eq(&self, other: &Extent2) -> bool {
        match self {
            ImageExtent::D1(e) => e.to_2d() == *other,
            ImageExtent::D2(e) => *e == *other,
            ImageExtent::D3(e) => *e == other.to_3d(),
        }
    }
}

impl PartialEq<Extent3> for ImageExtent {
    fn eq(&self, other: &Extent3) -> bool {
        match self {
            ImageExtent::D1(e) => e.to_3d() == *other,
            ImageExtent::D2(e) => e.to_3d() == *other,
            ImageExtent::D3(e) => *e == *other,
        }
    }
}

impl From<Extent1> for ImageExtent {
    #[inline(always)]
    fn from(extent: Extent1) -> Self {
        ImageExtent::D1(extent)
    }
}

impl From<Extent2> for ImageExtent {
    #[inline(always)]
    fn from(extent: Extent2) -> Self {
        ImageExtent::D2(extent)
    }
}

impl From<Extent3> for ImageExtent {
    #[inline(always)]
    fn from(extent: Extent3) -> Self {
        ImageExtent::D3(extent)
    }
}

bitflags::bitflags! {
    /// Image usage flags.
    /// 
    /// Image can only be used according to usage flags specified during creation.
    /// When creating a buffer, choose all flags that apply.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct ImageUsage: u32 {
        /// Image can be used as a source for transfer operations.
        /// i.e. it will be copied from.
        const TRANSFER_SRC = 0x0000_0001;

        /// Image can be used as a destination for transfer operations.
        /// i.e. it will be copied to.
        const TRANSFER_DST = 0x0000_0002;

        /// Image can be used as a sampled image in shader arguments.
        const SAMPLED = 0x0000_0004;

        /// Image can be used as a storage image in shader arguments.
        const STORAGE = 0x0000_0008;

        /// Image can be used as a target for rendering.
        const TARGET = 0x0000_0010;
    }
}

/// Description used for image creation.
pub struct ImageDesc<'a> {
    /// Image extent.
    pub extent: ImageExtent,

    /// Image pixel format.
    pub format: PixelFormat,

    /// Image usage flags.
    pub usage: ImageUsage,

    /// Image layers count.
    pub layers: u32,

    /// Image mip levels count.
    pub levels: u32,

    /// Image debug name.
    pub name: &'a str,
}

impl<'a> ImageDesc<'a> {
    /// Create a new image description.
    pub const fn new(extent: ImageExtent, format: PixelFormat, usage: ImageUsage) -> Self {
        ImageDesc {
            extent,
            format,
            usage,
            layers: 1,
            levels: 1,
            name: "",
        }
    }

    /// Create a new 1D image description.
    pub const fn new_d1(width: u32, format: PixelFormat, usage: ImageUsage) -> Self {
        ImageDesc::new(ImageExtent::D1(Extent1::new(width)), format, usage)
    }

    /// Create a new 2D image description.
    pub const fn new_d2(width: u32, height: u32, format: PixelFormat, usage: ImageUsage) -> Self {
        ImageDesc::new(ImageExtent::D2(Extent2::new(width, height)), format, usage)
    }

    /// Create a new 3D image description.
    pub const fn new_d3(
        width: u32,
        height: u32,
        depth: u32,
        format: PixelFormat,
        usage: ImageUsage,
    ) -> Self {
        ImageDesc::new(
            ImageExtent::D3(Extent3::new(width, height, depth)),
            format,
            usage,
        )
    }

    /// Set image layers count.
    pub fn layers(mut self, layers: u32) -> Self {
        self.layers = layers;
        self
    }

    /// Set image mip levels count.
    pub fn levels(mut self, levels: u32) -> Self {
        self.levels = levels;
        self
    }

    /// Create a new 1D texture description.
    pub const fn new_d1_texture(width: u32, format: PixelFormat) -> Self {
        ImageDesc::new_d1(
            width,
            format,
            ImageUsage::union(ImageUsage::SAMPLED, ImageUsage::TRANSFER_DST),
        )
    }

    /// Create a new 2D texture description.
    pub const fn new_d2_texture(width: u32, height: u32, format: PixelFormat) -> Self {
        ImageDesc::new_d2(
            width,
            height,
            format,
            ImageUsage::union(ImageUsage::SAMPLED, ImageUsage::TRANSFER_DST),
        )
    }

    /// Create a new 3D texture description.
    pub const fn new_d3_texture(width: u32, height: u32, depth: u32, format: PixelFormat) -> Self {
        ImageDesc::new_d3(
            width,
            height,
            depth,
            format,
            ImageUsage::union(ImageUsage::SAMPLED, ImageUsage::TRANSFER_DST),
        )
    }

    /// Create a new 1D render target description.
    pub const fn new_d1_rt(width: u32, format: PixelFormat) -> Self {
        ImageDesc::new_d1(
            width,
            format,
            ImageUsage::union(ImageUsage::SAMPLED, ImageUsage::TARGET),
        )
    }

    /// Create a new 2D render target description.
    pub const fn new_d2_rt(width: u32, height: u32, format: PixelFormat) -> Self {
        ImageDesc::new_d2(
            width,
            height,
            format,
            ImageUsage::union(ImageUsage::SAMPLED, ImageUsage::TARGET),
        )
    }

    /// Create a new 3D render target description.
    pub const fn new_d3_rt(width: u32, height: u32, depth: u32, format: PixelFormat) -> Self {
        ImageDesc::new_d3(
            width,
            height,
            depth,
            format,
            ImageUsage::union(ImageUsage::SAMPLED, ImageUsage::TARGET),
        )
    }

    /// Set image debug name.
    pub const fn with_name(mut self, name: &'a str) -> Self {
        self.name = name;
        self
    }
}

/// Description used for image view creation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ViewDesc {
    /// View pixel format.
    /// It can be different from the image format, but must be compatible.
    pub format: PixelFormat,

    /// Base layer of the view.
    /// 0's layer of the view would correspond to this layer of the parent image.
    pub base_layer: u32,

    /// Number of layers in the view.
    pub layers: u32,

    /// Base mip level of the view.
    /// 0's level of the view would correspond to this level of the parent image.
    pub base_level: u32,

    /// Number of mip levels in the view.
    pub levels: u32,

    /// Image component swizzle.
    pub swizzle: Swizzle,
}

impl ViewDesc {
    /// Create a new image view description.
    pub fn new(format: PixelFormat) -> Self {
        ViewDesc {
            format,
            base_layer: 0,
            layers: 1,
            base_level: 0,
            levels: 1,
            swizzle: Swizzle::IDENTITY,
        }
    }

    /// Set layers range of the view.
    pub fn layers(self, range: Range<u32>) -> Self {
        Self {
            layers: range.end - range.start,
            base_layer: range.start,
            ..self
        }
    }

    /// Set mip levels range of the view.
    pub fn levels(self, range: Range<u32>) -> Self {
        Self {
            levels: range.end - range.start,
            base_level: range.start,
            ..self
        }
    }

    /// Set image component swizzle.
    pub fn swizzle(self, swizzle: Swizzle) -> Self {
        Self { swizzle, ..self }
    }
}
