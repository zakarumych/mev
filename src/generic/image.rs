use std::{borrow::Cow, error::Error, fmt, ops::Mul, range::Range};

use super::{Extent1, Extent2, Extent3, format::PixelFormat};

use crate::{Extent, HasX, HasY, HasZ, Offset, backend::Image};

/// Image component swizzle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ComponentMap {
    Zero,
    One,
    R,
    G,
    B,
    A,
}

/// Image swizzle for each component.
///
/// It is used to remap components of an image in image views.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Swizzle {
    pub r: ComponentMap,
    pub g: ComponentMap,
    pub b: ComponentMap,
    pub a: ComponentMap,
}

impl Default for Swizzle {
    fn default() -> Self {
        Swizzle::IDENTITY
    }
}

impl Swizzle {
    /// Identity swizzle.
    /// i.e. use as-is.
    pub const IDENTITY: Self = Swizzle {
        r: ComponentMap::R,
        g: ComponentMap::G,
        b: ComponentMap::B,
        a: ComponentMap::A,
    };

    /// Map all components to R component.
    pub const RRRR: Self = Swizzle {
        r: ComponentMap::R,
        g: ComponentMap::R,
        b: ComponentMap::R,
        a: ComponentMap::R,
    };

    /// Map components R, G and B to one and Alpha to R.
    pub const _111R: Self = Swizzle {
        r: ComponentMap::One,
        g: ComponentMap::One,
        b: ComponentMap::One,
        a: ComponentMap::R,
    };
}

/// Combine two swizzles.
impl Mul for Swizzle {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        use ComponentMap::*;

        let mul = |rhs: ComponentMap, i| match rhs {
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
    #[inline(always)]
    pub const fn dimensions(&self) -> usize {
        match self {
            ImageExtent::D1(_) => 1,
            ImageExtent::D2(_) => 2,
            ImageExtent::D3(_) => 3,
        }
    }

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

    /// Extent at mip `level`: standard mip-chain halving, `max(1, dim >>
    /// level)` per axis.
    #[inline(always)]
    pub(crate) fn at_level(&self, level: u32) -> ImageExtent {
        match *self {
            ImageExtent::D1(e) => ImageExtent::D1(Extent1::new((e.width() >> level).max(1))),
            ImageExtent::D2(e) => ImageExtent::D2(Extent2::new(
                (e.width() >> level).max(1),
                (e.height() >> level).max(1),
            )),
            ImageExtent::D3(e) => ImageExtent::D3(Extent3::new(
                (e.width() >> level).max(1),
                (e.height() >> level).max(1),
                (e.depth() >> level).max(1),
            )),
        }
    }

    /// Convert into `Extent1`, expecting 1D image.
    ///
    /// # Panics
    ///
    /// Panics if the image is not 1D.
    #[inline(always)]
    pub fn expect_1d(self) -> Extent1 {
        match self {
            ImageExtent::D1(e) => e,
            _ => dimension_mismatch(),
        }
    }

    /// Convert into `Extent2`, expecting 2D image.
    ///
    /// # Panics
    ///
    /// Panics if the image is not 2D.
    #[inline(always)]
    pub fn expect_2d(self) -> Extent2 {
        match self {
            ImageExtent::D2(e) => e,
            _ => dimension_mismatch(),
        }
    }

    /// Convert into `Extent3`, expecting 3D image.
    ///
    /// # Panics
    ///
    /// Panics if the image is not 3D.
    #[inline(always)]
    pub fn expect_3d(self) -> Extent3 {
        match self {
            ImageExtent::D3(e) => e,
            _ => dimension_mismatch(),
        }
    }

    /// Convert into `Extent<D>`, expecting ND image.
    ///
    /// # Panics
    ///
    /// Panics if the image is not ND.
    #[inline(always)]
    pub fn expect_nd<const D: usize>(self) -> Extent<D> {
        match self {
            ImageExtent::D1(e) if D == 1 => Extent::from_1d(e),
            ImageExtent::D2(e) if D == 2 => Extent::from_2d(e),
            ImageExtent::D3(e) if D == 3 => Extent::from_3d(e),
            _ => dimension_mismatch(),
        }
    }

    /// Convert into `Extent1` from any image extent.
    ///
    /// Ignores height if the image is 2D or 3D.
    /// Ignores depth if the image is 3D.
    #[inline(always)]
    pub fn into_1d(self) -> Extent1 {
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
    pub fn into_2d(self) -> Extent2 {
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
    pub fn into_3d(self) -> Extent3 {
        match self {
            ImageExtent::D1(e) => e.to_3d(),
            ImageExtent::D2(e) => e.to_3d(),
            ImageExtent::D3(e) => e,
        }
    }

    #[inline(always)]
    pub fn into_nd<const D: usize>(&self) -> Extent<D> {
        match self {
            ImageExtent::D1(e) => Extent::from_1d(*e),
            ImageExtent::D2(e) => Extent::from_2d(*e),
            ImageExtent::D3(e) => Extent::from_3d(*e),
        }
    }

    #[inline(always)]
    pub fn is_1d(&self) -> bool {
        matches!(self, ImageExtent::D1(_))
    }

    #[inline(always)]
    pub fn is_2d(&self) -> bool {
        matches!(self, ImageExtent::D2(_))
    }

    #[inline(always)]
    pub fn is_3d(&self) -> bool {
        matches!(self, ImageExtent::D3(_))
    }

    #[inline(always)]
    pub fn is_nd<const D: usize>(&self) -> bool {
        match self {
            ImageExtent::D1(_) => D == 1,
            ImageExtent::D2(_) => D == 2,
            ImageExtent::D3(_) => D == 3,
        }
    }

    #[inline(always)]
    pub fn as_1d(&self) -> Option<Extent1> {
        match self {
            ImageExtent::D1(e) => Some(*e),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn as_2d(&self) -> Option<Extent2> {
        match self {
            ImageExtent::D2(e) => Some(*e),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn as_3d(&self) -> Option<Extent3> {
        match self {
            ImageExtent::D3(e) => Some(*e),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn as_nd<const D: usize>(&self) -> Option<Extent<D>> {
        match self {
            ImageExtent::D1(e) if D == 1 => Some(Extent::from_1d(*e)),
            ImageExtent::D2(e) if D == 2 => Some(Extent::from_2d(*e)),
            ImageExtent::D3(e) if D == 3 => Some(Extent::from_3d(*e)),
            _ => None,
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

/// Types that can be used as the `dims` field of an [`ImageDesc`], selecting
/// both what shape of image gets created and what typed wrapper
/// [`Device::new_image`](crate::backend::Device::new_image) returns for it.
pub trait ImageDims {
    /// The image handle type produced from these dims.
    type Image;

    /// The extent this describes.
    fn extent(&self) -> ImageExtent;

    /// The layer count this describes.
    fn layers(&self) -> u32;

    /// Called before any GPU allocation happens. Default no-op; overridden
    /// to panic on an invalid combination (e.g. array dims with `layers <=
    /// 1`) so the check fires before wasting a real allocation, not after.
    fn validate(&self) {}

    /// Wraps a freshly-created erased [`Image`] into `Self::Image`. Callers
    /// guarantee the image actually matches this dims value (same
    /// extent/layers used to create it), so implementations may construct
    /// unchecked.
    fn wrap(image: Image) -> Self::Image;
}

impl ImageDims for ImageExtent {
    type Image = Image;

    fn extent(&self) -> ImageExtent {
        *self
    }

    fn layers(&self) -> u32 {
        1
    }

    fn wrap(image: Image) -> Image {
        image
    }
}

impl ImageDims for (ImageExtent, u32) {
    type Image = Image;

    fn extent(&self) -> ImageExtent {
        self.0
    }

    fn layers(&self) -> u32 {
        self.1
    }

    fn wrap(image: Image) -> Image {
        image
    }
}

impl<const D: usize> ImageDims for Extent<D, u32> {
    type Image = ImageN<D>;

    fn extent(&self) -> ImageExtent {
        match D {
            1 => ImageExtent::D1(self.to_1d()),
            2 => ImageExtent::D2(self.to_2d()),
            3 => ImageExtent::D3(self.to_3d()),
            _ => dimension_mismatch(),
        }
    }

    fn layers(&self) -> u32 {
        1
    }

    fn wrap(image: Image) -> ImageN<D> {
        ImageN::new_unchecked(image)
    }
}

impl<const D: usize> ImageDims for (Extent<D, u32>, u32) {
    type Image = ImageArrayN<D>;

    fn extent(&self) -> ImageExtent {
        <Extent<D, u32> as ImageDims>::extent(&self.0)
    }

    fn layers(&self) -> u32 {
        self.1
    }

    fn validate(&self) {
        assert_array_layers_fit(self.1);
    }

    fn wrap(image: Image) -> ImageArrayN<D> {
        ImageArrayN::new_unchecked(image)
    }
}

#[inline(always)]
pub(crate) fn assert_array_layers_fit(layers: u32) {
    if layers <= 1 {
        array_layers_unfit();
    }
}

#[cold]
#[inline(never)]
fn array_layers_unfit() -> ! {
    panic!("Array image must have more than 1 layer; use the non-array constructor instead");
}

/// Description used for image creation.
pub struct ImageDesc<'a, T: ImageDims = ImageExtent> {
    /// Image dims: encodes extent, layer count, and which typed image
    /// handle [`Device::new_image`](crate::backend::Device::new_image)
    /// returns.
    pub dims: T,

    /// Image pixel format.
    pub format: PixelFormat,

    /// Image mip levels count.
    pub levels: u32,

    /// Image usage flags.
    pub usage: ImageUsage,

    /// Image debug name.
    pub name: &'a str,
}

impl<'a, T: ImageDims> ImageDesc<'a, T> {
    /// Create a new image description.
    pub fn new(dims: T, format: PixelFormat, usage: ImageUsage) -> Self {
        ImageDesc {
            dims,
            format,
            usage,
            levels: 1,
            name: "",
        }
    }

    /// Set image mip levels count.
    pub fn levels(mut self, levels: u32) -> Self {
        self.levels = levels;
        self
    }

    /// Set image debug name.
    pub fn with_name(mut self, name: &'a str) -> Self {
        self.name = name;
        self
    }
}

impl<'a, const D: usize> ImageDesc<'a, Extent<D, u32>> {
    /// Set image layers count, turning this into an array image
    /// description.
    pub fn layers(self, layers: u32) -> ImageDesc<'a, (Extent<D, u32>, u32)> {
        ImageDesc {
            dims: (self.dims, layers),
            format: self.format,
            levels: self.levels,
            usage: self.usage,
            name: self.name,
        }
    }
}

impl<'a> ImageDesc<'a, ImageExtent> {
    /// Set image layers count, turning this into an array image
    /// description.
    pub fn layers(self, layers: u32) -> ImageDesc<'a, (ImageExtent, u32)> {
        ImageDesc {
            dims: (self.dims, layers),
            format: self.format,
            levels: self.levels,
            usage: self.usage,
            name: self.name,
        }
    }
}

impl<'a> ImageDesc<'a, Extent1<u32>> {
    /// Create a new 1D image description.
    pub fn new_d1(width: u32, format: PixelFormat, usage: ImageUsage) -> Self {
        ImageDesc::new(Extent1::new(width), format, usage)
    }

    /// Create a new 1D texture description.
    pub fn new_d1_texture(width: u32, format: PixelFormat) -> Self {
        ImageDesc::new_d1(
            width,
            format,
            ImageUsage::union(ImageUsage::SAMPLED, ImageUsage::TRANSFER_DST),
        )
    }

    /// Create a new 1D render target description.
    pub fn new_d1_rt(width: u32, format: PixelFormat) -> Self {
        ImageDesc::new_d1(
            width,
            format,
            ImageUsage::union(ImageUsage::SAMPLED, ImageUsage::TARGET),
        )
    }
}

impl<'a> ImageDesc<'a, Extent2<u32>> {
    /// Create a new 2D image description.
    pub fn new_d2(width: u32, height: u32, format: PixelFormat, usage: ImageUsage) -> Self {
        ImageDesc::new(Extent2::new(width, height), format, usage)
    }

    /// Create a new 2D texture description.
    pub fn new_d2_texture(width: u32, height: u32, format: PixelFormat) -> Self {
        ImageDesc::new_d2(
            width,
            height,
            format,
            ImageUsage::union(ImageUsage::SAMPLED, ImageUsage::TRANSFER_DST),
        )
    }

    /// Create a new 2D render target description.
    pub fn new_d2_rt(width: u32, height: u32, format: PixelFormat) -> Self {
        ImageDesc::new_d2(
            width,
            height,
            format,
            ImageUsage::union(ImageUsage::SAMPLED, ImageUsage::TARGET),
        )
    }
}

impl<'a> ImageDesc<'a, Extent3<u32>> {
    /// Create a new 3D image description.
    pub fn new_d3(
        width: u32,
        height: u32,
        depth: u32,
        format: PixelFormat,
        usage: ImageUsage,
    ) -> Self {
        ImageDesc::new(Extent3::new(width, height, depth), format, usage)
    }

    /// Create a new 3D texture description.
    pub fn new_d3_texture(width: u32, height: u32, depth: u32, format: PixelFormat) -> Self {
        ImageDesc::new_d3(
            width,
            height,
            depth,
            format,
            ImageUsage::union(ImageUsage::SAMPLED, ImageUsage::TRANSFER_DST),
        )
    }

    /// Create a new 3D render target description.
    pub fn new_d3_rt(width: u32, height: u32, depth: u32, format: PixelFormat) -> Self {
        ImageDesc::new_d3(
            width,
            height,
            depth,
            format,
            ImageUsage::union(ImageUsage::SAMPLED, ImageUsage::TARGET),
        )
    }
}

/// Description used for image view creation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ViewDesc {
    /// View pixel format.
    /// It can be different from the image format, but must be compatible.
    pub format: PixelFormat,

    /// Layers in the view.
    pub layers: Range<u32>,

    /// MIP Levels in the view.
    pub levels: Range<u32>,

    /// Image component swizzle.
    pub swizzle: Swizzle,
}

impl ViewDesc {
    /// Create a new image view description.
    pub fn new(format: PixelFormat) -> Self {
        ViewDesc {
            format,
            layers: (0..1).into(),
            levels: (0..1).into(),
            swizzle: Swizzle::IDENTITY,
        }
    }

    /// Set layers range of the view.
    pub fn layers(self, range: Range<u32>) -> Self {
        Self {
            layers: range,
            ..self
        }
    }

    /// Restrict the view to a single layer.
    pub fn layer(self, index: u32) -> Self {
        Self {
            layers: (index..index + 1).into(),
            ..self
        }
    }

    /// Set mip levels range of the view.
    pub fn levels(self, range: Range<u32>) -> Self {
        Self {
            levels: range,
            ..self
        }
    }

    /// Set image component swizzle.
    pub fn swizzle(self, swizzle: Swizzle) -> Self {
        Self { swizzle, ..self }
    }
}

/// Trait for types that can be used to describe a region of an image.
pub trait ImageRegion<const D: usize> {
    fn offset(&self, image_extent: Extent<D>) -> Offset<D>;
    fn extent(&self, image_extent: Extent<D>) -> Extent<D>;
    fn layers(&self, image_layers: u32) -> Range<u32>;
}

impl<const D: usize> ImageRegion<D> for Extent<D> {
    #[inline(always)]
    fn offset(&self, image_extent: Extent<D>) -> Offset<D> {
        Offset::ZERO
    }

    #[inline(always)]
    fn extent(&self, image_extent: Extent<D>) -> Extent<D> {
        *self
    }

    #[inline(always)]
    fn layers(&self, image_layers: u32) -> Range<u32> {
        (0..image_layers).into()
    }
}

impl<const D: usize> ImageRegion<D> for (Offset<D>, Extent<D>) {
    #[inline(always)]
    fn offset(&self, image_extent: Extent<D>) -> Offset<D> {
        self.0
    }

    #[inline(always)]
    fn extent(&self, image_extent: Extent<D>) -> Extent<D> {
        self.1
    }

    #[inline(always)]
    fn layers(&self, image_layers: u32) -> Range<u32> {
        (0..image_layers).into()
    }
}

impl<const D: usize> ImageRegion<D> for (Extent<D>, Range<u32>) {
    #[inline(always)]
    fn offset(&self, image_extent: Extent<D>) -> Offset<D> {
        Offset::ZERO
    }

    #[inline(always)]
    fn extent(&self, image_extent: Extent<D>) -> Extent<D> {
        self.0
    }

    #[inline(always)]
    fn layers(&self, image_layers: u32) -> Range<u32> {
        assert!(
            self.1.start <= self.1.end,
            "ImageRegion: layers range is invalid"
        );
        assert!(
            self.1.end <= image_layers,
            "ImageRegion: layers range exceeds image layer count"
        );
        self.1
    }
}

impl<const D: usize> ImageRegion<D> for (Offset<D>, Extent<D>, Range<u32>) {
    #[inline(always)]
    fn offset(&self, image_extent: Extent<D>) -> Offset<D> {
        self.0
    }

    #[inline(always)]
    fn extent(&self, image_extent: Extent<D>) -> Extent<D> {
        self.1
    }

    #[inline(always)]
    fn layers(&self, image_layers: u32) -> Range<u32> {
        assert!(
            self.2.start <= self.2.end,
            "ImageRegion: layers range is invalid"
        );
        assert!(
            self.2.end <= image_layers,
            "ImageRegion: layers range exceeds image layer count"
        );
        self.2
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImageSlice<'a, const D: usize> {
    pub(crate) image: Cow<'a, Image>,
    pub(crate) offset: Offset<D>,
    pub(crate) extent: Extent<D>,
    pub(crate) layers: Range<u32>,
    pub(crate) level: u32,
}

impl<const D: usize> ImageSlice<'_, D> {
    pub fn image(&self) -> &Image {
        &self.image
    }

    pub fn level(&self) -> u32 {
        self.level
    }

    pub fn layers(&self) -> Range<u32> {
        self.layers
    }

    pub fn offset(&self) -> Offset<D> {
        self.offset
    }

    pub fn extent(&self) -> Extent<D> {
        self.extent
    }

    /// Panics if this slice's region doesn't fit within its image: offset
    /// must be non-negative, `offset + extent` must not exceed the image's
    /// real extent, and `layers` must be a valid, in-range subset of the
    /// image's real layer count.
    pub fn assert_fits_image(&self) {
        let image_extent = self.image.extent().expect_nd::<D>();

        for i in 0..D {
            if self.offset.0[i] < 0 {
                offset_unfit();
            }
            let end = self.offset.0[i] as i64 + self.extent.0[i] as i64;
            if end > image_extent.0[i] as i64 {
                extent_unfit();
            }
        }

        if self.layers.start > self.layers.end {
            layers_range_invalid();
        }
        if self.layers.end > self.image.layers() {
            layers_unfit();
        }
    }

    pub fn into_owned(self) -> ImageSlice<'static, D> {
        ImageSlice {
            image: Cow::Owned(self.image.into_owned()),
            level: self.level,
            layers: self.layers,
            offset: self.offset,
            extent: self.extent,
        }
    }

    pub fn with_level(mut self, level: u32) -> Self {
        assert_level_fits_to_image(level, self.image.levels());
        self.level = level;
        self
    }
}

pub type ImageSlice1D<'a> = ImageSlice<'a, 1>;
pub type ImageSlice2D<'a> = ImageSlice<'a, 2>;
pub type ImageSlice3D<'a> = ImageSlice<'a, 3>;

impl Image {
    pub fn slice<const D: usize>(&self, region: impl ImageRegion<D>) -> ImageSlice<'_, D> {
        let image_extent = self.extent().expect_nd::<D>();

        let offset = region.offset(image_extent);
        let extent = region.extent(image_extent);
        let layers = region.layers(self.layers());

        ImageSlice {
            image: Cow::Borrowed(self),
            offset,
            extent,
            layers,
            level: 0,
        }
    }

    /// Like `view`, but returns a typed, single-layer `ImageN<D>` instead of
    /// an erased `Image`. `desc` must describe exactly one layer (`ViewDesc`
    /// defaults to `layers: 1`; use `ViewDesc::layer` to pick which one out
    /// of a multi-layer image). Panics if `desc.layers != 1` or the
    /// resulting view isn't `D`-dimensional.
    pub fn view_layer<const D: usize>(
        &self,
        device: &crate::backend::Device,
        desc: ViewDesc,
    ) -> ImageN<D> {
        assert_eq!(
            desc.layers.end - desc.layers.start,
            1,
            "Image::view_layer: desc must describe exactly one layer"
        );
        let view = self.view(device, desc);
        ImageN::try_from(view).expect("Image::view_layer: resulting view is not N-dimensional")
    }
}

impl<'a, const D: usize> ImageSlice<'a, D> {
    pub fn slice(&self, region: impl ImageRegion<D>) -> ImageSlice<'_, D> {
        let offset = region.offset(self.extent());
        let extent = region.extent(self.extent());
        let layers = region.layers(self.layers.end - self.layers.start);

        ImageSlice {
            image: Cow::Borrowed(&self.image),
            offset: self.offset + offset,
            extent,
            layers: Range {
                start: self.layers.start + layers.start,
                end: self.layers.start + layers.end,
            },
            level: self.level,
        }
    }
}

pub trait AsImageSlice<const D: usize> {
    fn as_image_slice(&self) -> ImageSlice<'_, D>;
}

impl<const D: usize> AsImageSlice<D> for Image {
    fn as_image_slice(&self) -> ImageSlice<'_, D> {
        let extent = self.extent().expect_nd();

        ImageSlice {
            image: Cow::Borrowed(self),
            offset: Offset::ZERO,
            extent,
            layers: (0..self.layers()).into(),
            level: 0,
        }
    }
}

impl<'a, const D: usize> AsImageSlice<D> for ImageSlice<'a, D> {
    fn as_image_slice(&self) -> ImageSlice<'_, D> {
        ImageSlice {
            image: Cow::Borrowed(&self.image),
            offset: self.offset.to_nd(),
            extent: self.extent.to_nd(),
            layers: self.layers,
            level: self.level,
        }
    }
}

impl AsRef<Image> for Image {
    #[inline(always)]
    fn as_ref(&self) -> &Image {
        self
    }
}

/// N-dimensional array image with more than 1 layer.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImageArrayN<const D: usize>(Image);

impl<const D: usize> AsRef<ImageArrayN<D>> for ImageArrayN<D> {
    #[inline(always)]
    fn as_ref(&self) -> &ImageArrayN<D> {
        self
    }
}

impl<const D: usize> AsRef<Image> for ImageArrayN<D> {
    #[inline(always)]
    fn as_ref(&self) -> &Image {
        &self.0
    }
}

impl<const D: usize> ImageArrayN<D> {
    /// Constructs an `ImageArrayN` from an image that the caller guarantees
    /// is `D`-dimensional with more than 1 layer.
    ///
    /// This is the only way to construct an `ImageArrayN` without going
    /// through the fallible `TryFrom<Image>` impl.
    pub(crate) fn new_unchecked(image: Image) -> Self {
        ImageArrayN(image)
    }

    pub fn as_image(&self) -> &Image {
        &self.0
    }

    pub fn into_image(self) -> Image {
        self.0
    }

    /// Extent, typed. Infallible - `D` is guaranteed correct by construction.
    pub fn extent(&self) -> Extent<D> {
        self.0.extent().expect_nd::<D>()
    }

    pub fn format(&self) -> PixelFormat {
        self.0.format()
    }

    pub fn layers(&self) -> u32 {
        self.0.layers()
    }

    pub fn levels(&self) -> u32 {
        self.0.levels()
    }

    pub fn usage(&self) -> ImageUsage {
        self.0.usage()
    }

    pub fn view(&self, device: &crate::backend::Device, desc: ViewDesc) -> Image {
        self.0.view(device, desc)
    }

    /// Like `view`, but returns a typed, single-layer `ImageN<D>` view of
    /// this array image instead of an erased `Image`. `desc` must describe
    /// exactly one layer (`ViewDesc` defaults to `layers: 1`; use
    /// `ViewDesc::layer` to pick which one). `D` is already guaranteed to
    /// match (same as `self`'s own `D`), so only the layer count is
    /// checked. Panics if `desc.layers != 1`.
    pub fn view_layer(&self, device: &crate::backend::Device, desc: ViewDesc) -> ImageN<D> {
        assert_eq!(
            desc.layers.end - desc.layers.start,
            1,
            "ImageArrayN::view_layer: desc must describe exactly one layer"
        );
        let view = self.0.view(device, desc);
        ImageN::new_unchecked(view)
    }

    pub fn detached(&self) -> bool {
        self.0.detached()
    }

    pub fn slice(&self, region: impl ImageRegion<D>) -> ImageSlice<'_, D> {
        self.0.slice(region)
    }
}

pub type ImageArray1D = ImageArrayN<1>;
pub type ImageArray2D = ImageArrayN<2>;
pub type ImageArray3D = ImageArrayN<3>;

/// N-dimensional image with exactly 1 layer.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImageN<const D: usize>(ImageArrayN<D>);

impl<const D: usize> AsRef<ImageN<D>> for ImageN<D> {
    #[inline(always)]
    fn as_ref(&self) -> &ImageN<D> {
        self
    }
}

impl<const D: usize> AsRef<ImageArrayN<D>> for ImageN<D> {
    #[inline(always)]
    fn as_ref(&self) -> &ImageArrayN<D> {
        &self.0
    }
}

impl<const D: usize> AsRef<Image> for ImageN<D> {
    #[inline(always)]
    fn as_ref(&self) -> &Image {
        self.0.as_image()
    }
}

impl<const D: usize> ImageN<D> {
    /// Constructs an `ImageN` from an image that the caller guarantees is
    /// `D`-dimensional with exactly 1 layer.
    ///
    /// This is the only way to construct an `ImageN` without going through
    /// the fallible `TryFrom<Image>` impl.
    #[inline(always)]
    pub(crate) fn new_unchecked(image: Image) -> Self {
        ImageN(ImageArrayN::new_unchecked(image))
    }

    #[inline(always)]
    pub fn as_image(&self) -> &Image {
        self.0.as_ref()
    }

    #[inline(always)]
    pub fn into_image(self) -> Image {
        self.0.into_image()
    }

    /// Extent, typed. Infallible - `D` is guaranteed correct by construction.
    #[inline(always)]
    pub fn extent(&self) -> Extent<D> {
        self.0.extent()
    }

    #[inline(always)]
    pub fn format(&self) -> PixelFormat {
        self.0.format()
    }

    #[inline(always)]
    pub fn levels(&self) -> u32 {
        self.0.levels()
    }

    #[inline(always)]
    pub fn usage(&self) -> ImageUsage {
        self.0.usage()
    }

    #[inline(always)]
    pub fn view(&self, device: &crate::backend::Device, desc: ViewDesc) -> Image {
        self.0.view(device, desc)
    }

    #[inline(always)]
    pub fn detached(&self) -> bool {
        self.0.detached()
    }

    #[inline(always)]
    pub fn slice(&self, region: impl ImageRegion<D>) -> ImageSlice<'_, D> {
        self.0.slice(region)
    }
}

pub type Image1D = ImageN<1>;
pub type Image2D = ImageN<2>;
pub type Image3D = ImageN<3>;

/// Error returned when converting a dimension-erased `Image` into a
/// dimension-typed `ImageN`/`ImageArrayN` whose actual dimensionality (or,
/// for `ImageArrayN`, layer count) doesn't match what was requested.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageDimensionMismatch;

impl fmt::Display for ImageDimensionMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "image dimensions or layer count do not match the requested type"
        )
    }
}

impl Error for ImageDimensionMismatch {}

impl<const D: usize> TryFrom<Image> for ImageN<D> {
    type Error = ImageDimensionMismatch;

    #[inline(always)]
    fn try_from(image: Image) -> Result<Self, Self::Error> {
        if image.extent().dimensions() == D && image.layers() == 1 {
            Ok(ImageN(ImageArrayN(image)))
        } else {
            Err(ImageDimensionMismatch)
        }
    }
}

impl<const D: usize> TryFrom<Image> for ImageArrayN<D> {
    type Error = ImageDimensionMismatch;

    #[inline(always)]
    fn try_from(image: Image) -> Result<Self, Self::Error> {
        if image.extent().dimensions() == D && image.layers() > 1 {
            Ok(ImageArrayN(image))
        } else {
            Err(ImageDimensionMismatch)
        }
    }
}

impl<const D: usize> AsImageSlice<D> for ImageN<D> {
    #[inline(always)]
    fn as_image_slice(&self) -> ImageSlice<'_, D> {
        ImageSlice {
            image: Cow::Borrowed(self.as_image()),
            offset: Offset::ZERO,
            extent: self.extent(),
            layers: (0..1).into(),
            level: 0,
        }
    }
}

impl<const D: usize> AsImageSlice<D> for ImageArrayN<D> {
    #[inline(always)]
    fn as_image_slice(&self) -> ImageSlice<'_, D> {
        ImageSlice {
            image: Cow::Borrowed(self.as_image()),
            offset: Offset::ZERO,
            extent: self.extent(),
            layers: (0..self.0.layers()).into(),
            level: 0,
        }
    }
}

#[inline(always)]
fn assert_level_fits_to_image(level: u32, image_levels: u32) {
    if level >= image_levels {
        level_out_of_bounds();
    }
}

#[cold]
#[inline(never)]
fn layers_range_invalid() -> ! {
    panic!("Region layers range is invalid");
}

#[cold]
#[inline(never)]
fn layers_unfit() -> ! {
    panic!("Region layers do not fit in the image layers");
}

#[cold]
#[inline(never)]
fn offset_unfit() -> ! {
    panic!("Region offset does not fit in the image extent");
}

#[cold]
#[inline(never)]
fn level_out_of_bounds() -> ! {
    panic!("Region level is out of bounds");
}

#[cold]
#[inline(never)]
fn extent_unfit() -> ! {
    panic!("Region extent does not fit in the image extent");
}

#[inline(always)]
fn assert_dimensions_match<const D: usize>(extent: ImageExtent) {
    if extent.dimensions() != D {
        dimension_mismatch();
    }
}

#[cold]
#[inline(never)]
fn dimension_mismatch() -> ! {
    panic!("Image extent does not match the requested dimension");
}
