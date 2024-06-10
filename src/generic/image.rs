use std::{
    error::Error,
    fmt,
    ops::{Mul, Range},
};

use super::{format::PixelFormat, Extent1, Extent2, Extent3};

/// Image component swizzle
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

/// Image swizzle
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Swizzle {
    pub r: ComponentSwizzle,
    pub g: ComponentSwizzle,
    pub b: ComponentSwizzle,
    pub a: ComponentSwizzle,
}

impl Swizzle {
    pub const IDENTITY: Self = Swizzle {
        r: ComponentSwizzle::Identity,
        g: ComponentSwizzle::Identity,
        b: ComponentSwizzle::Identity,
        a: ComponentSwizzle::Identity,
    };

    pub const RRRR: Self = Swizzle {
        r: ComponentSwizzle::R,
        g: ComponentSwizzle::R,
        b: ComponentSwizzle::R,
        a: ComponentSwizzle::R,
    };

    pub const _111R: Self = Swizzle {
        r: ComponentSwizzle::One,
        g: ComponentSwizzle::One,
        b: ComponentSwizzle::One,
        a: ComponentSwizzle::R,
    };
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageExtent {
    D1(Extent1),
    D2(Extent2),
    D3(Extent3),
}

impl ImageExtent {
    #[inline(always)]
    pub fn width(&self) -> u32 {
        match self {
            ImageExtent::D1(e) => e.width(),
            ImageExtent::D2(e) => e.width(),
            ImageExtent::D3(e) => e.width(),
        }
    }

    #[inline(always)]
    pub fn height(&self) -> u32 {
        match self {
            ImageExtent::D1(e) => 1,
            ImageExtent::D2(e) => e.height(),
            ImageExtent::D3(e) => e.height(),
        }
    }

    #[inline(always)]
    pub fn depth(&self) -> u32 {
        match self {
            ImageExtent::D1(e) => 1,
            ImageExtent::D2(e) => 1,
            ImageExtent::D3(e) => e.depth(),
        }
    }

    #[inline(always)]
    pub fn expect_1d(self) -> Extent1<u32> {
        match self {
            ImageExtent::D1(e) => e,
            _ => panic!("Expected 1D image extent"),
        }
    }

    #[inline(always)]
    pub fn expect_2d(self) -> Extent2<u32> {
        match self {
            ImageExtent::D2(e) => e,
            _ => panic!("Expected 2D image extent"),
        }
    }

    #[inline(always)]
    pub fn expect_3d(self) -> Extent3<u32> {
        match self {
            ImageExtent::D3(e) => e,
            _ => panic!("Expected 3D image extent"),
        }
    }

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
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct ImageUsage: u32 {
        const TRANSFER_SRC = 0x0000_0001;
        const TRANSFER_DST = 0x0000_0002;
        const SAMPLED = 0x0000_0004;
        const STORAGE = 0x0000_0008;
        const TARGET = 0x0000_0010;
    }
}

pub struct ImageDesc<'a> {
    pub dimensions: ImageExtent,
    pub format: PixelFormat,
    pub usage: ImageUsage,
    pub layers: u32,
    pub levels: u32,
    pub name: &'a str,
}

impl<'a> ImageDesc<'a> {
    pub const fn new(dimensions: ImageExtent, format: PixelFormat, usage: ImageUsage) -> Self {
        ImageDesc {
            dimensions,
            format,
            usage,
            layers: 1,
            levels: 1,
            name: "",
        }
    }

    pub const fn new_d1(width: u32, format: PixelFormat, usage: ImageUsage) -> Self {
        ImageDesc::new(ImageExtent::D1(Extent1::new(width)), format, usage)
    }

    pub const fn new_d2(width: u32, height: u32, format: PixelFormat, usage: ImageUsage) -> Self {
        ImageDesc::new(ImageExtent::D2(Extent2::new(width, height)), format, usage)
    }

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

    pub fn layers(mut self, layers: u32) -> Self {
        self.layers = layers;
        self
    }

    pub fn levels(mut self, levels: u32) -> Self {
        self.levels = levels;
        self
    }

    pub const fn new_d1_texture(width: u32, format: PixelFormat) -> Self {
        ImageDesc::new_d1(
            width,
            format,
            ImageUsage::union(ImageUsage::SAMPLED, ImageUsage::TRANSFER_DST),
        )
    }

    pub const fn new_d2_texture(width: u32, height: u32, format: PixelFormat) -> Self {
        ImageDesc::new_d2(
            width,
            height,
            format,
            ImageUsage::union(ImageUsage::SAMPLED, ImageUsage::TRANSFER_DST),
        )
    }

    pub const fn new_d3_texture(width: u32, height: u32, depth: u32, format: PixelFormat) -> Self {
        ImageDesc::new_d3(
            width,
            height,
            depth,
            format,
            ImageUsage::union(ImageUsage::SAMPLED, ImageUsage::TRANSFER_DST),
        )
    }

    pub const fn new_d1_rtt(width: u32, format: PixelFormat) -> Self {
        ImageDesc::new_d1(
            width,
            format,
            ImageUsage::union(ImageUsage::SAMPLED, ImageUsage::TARGET),
        )
    }

    pub const fn new_d2_rtt(width: u32, height: u32, format: PixelFormat) -> Self {
        ImageDesc::new_d2(
            width,
            height,
            format,
            ImageUsage::union(ImageUsage::SAMPLED, ImageUsage::TARGET),
        )
    }

    pub const fn new_d3_rtt(width: u32, height: u32, depth: u32, format: PixelFormat) -> Self {
        ImageDesc::new_d3(
            width,
            height,
            depth,
            format,
            ImageUsage::union(ImageUsage::SAMPLED, ImageUsage::TARGET),
        )
    }

    pub const fn with_name(mut self, name: &'a str) -> Self {
        self.name = name;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ViewDesc {
    pub format: PixelFormat,
    pub base_layer: u32,
    pub layers: u32,
    pub base_level: u32,
    pub levels: u32,
    pub swizzle: Swizzle,
}

impl ViewDesc {
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

    pub fn layers(self, range: Range<u32>) -> Self {
        Self {
            layers: range.end - range.start,
            base_layer: range.start,
            ..self
        }
    }

    pub fn levels(self, range: Range<u32>) -> Self {
        Self {
            levels: range.end - range.start,
            base_level: range.start,
            ..self
        }
    }

    pub fn swizzle(self, swizzle: Swizzle) -> Self {
        Self { swizzle, ..self }
    }
}
