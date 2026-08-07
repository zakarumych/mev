use core::fmt;
use std::{
    hash::{Hash, Hasher},
    mem::{ManuallyDrop, size_of},
    num::NonZero,
    sync::Arc,
};

use ash::vk::{self, Handle};
use gpu_alloc::MemoryBlock;
use hashbrown::{HashMap, hash_map::Entry};
use parking_lot::Mutex;

use crate::generic::{
    ArgumentKind, Automatic, ImageArrayN, ImageExtent, ImageN, ImageUsage, OutOfMemory,
    PixelFormat, Sampled, Storage, Swizzle, ViewDesc,
};

use super::{
    arguments::ArgumentsField,
    device::{Device, DeviceMemory, DeviceOwned, WeakDevice},
    from::IntoAsh,
    refs::Refs,
};

enum Flavor {
    Device {
        block: ManuallyDrop<MemoryBlock<DeviceMemory>>,
        idx: usize,
    },
    Swapchain,
}

// Contains actual `vk::Image`
struct ImageData {
    owner: WeakDevice,
    format: PixelFormat,
    usage: ImageUsage,
    extent: ImageExtent,
    layers: u32,
    levels: u32,
    flavor: Flavor,
    views: Mutex<HashMap<ViewDesc, (vk::ImageView, usize)>>,
}

impl Drop for ImageData {
    fn drop(&mut self) {
        self.owner
            .drop_image_views(self.views.get_mut().values().map(|(_, idx)| *idx));

        if let Flavor::Device { block, idx } = &mut self.flavor {
            self.owner
                .drop_image(*idx, unsafe { ManuallyDrop::take(block) });
        }
    }
}

struct Inner {
    data: Arc<ImageData>,
    desc: ViewDesc,
    usage: ImageUsage,
    extent: ImageExtent,
    owner: WeakDevice,
}

#[derive(Clone)]
pub struct Image {
    handle: vk::Image,
    view: vk::ImageView,
    inner: Arc<Inner>,
}

impl PartialEq for Image {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
            && self.view == other.view
            && Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for Image {}

impl Hash for Image {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.view.hash(state);
        Arc::as_ptr(&self.inner).hash(state);
    }
}

impl fmt::Debug for Image {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Image")
            .field("handle", &self.handle)
            .field("view", &self.view)
            .finish()
    }
}

impl DeviceOwned for Image {
    #[inline]
    fn owner(&self) -> &WeakDevice {
        &self.inner.owner
    }
}

impl Image {
    fn build(
        owner: WeakDevice,
        handle: vk::Image,
        view: vk::ImageView,
        view_idx: usize,
        extent: impl Into<ImageExtent>,
        format: PixelFormat,
        usage: ImageUsage,
        layers: u32,
        levels: u32,
        flavor: Flavor,
    ) -> Self {
        let extent = extent.into();
        let desc = ViewDesc {
            format,
            layers: (0..layers).into(),
            levels: (0..levels).into(),
            swizzle: Swizzle::IDENTITY,
        };

        let mut views = HashMap::new();
        views.insert(desc, (view, view_idx));

        Image {
            handle,
            view,
            inner: Arc::new(Inner {
                data: Arc::new(ImageData {
                    owner: owner.clone(),
                    extent,
                    format,
                    usage,
                    layers,
                    levels,
                    flavor,
                    views: Mutex::new(views),
                }),
                desc,
                extent,
                usage,
                owner,
            }),
        }
    }

    pub(super) fn new(
        owner: WeakDevice,
        handle: vk::Image,
        view: vk::ImageView,
        view_idx: usize,
        extent: ImageExtent,
        format: PixelFormat,
        usage: ImageUsage,
        layers: u32,
        levels: u32,
        block: MemoryBlock<DeviceMemory>,
        idx: usize,
    ) -> Self {
        Image::build(
            owner,
            handle,
            view,
            view_idx,
            extent,
            format,
            usage,
            layers,
            levels,
            Flavor::Device {
                block: ManuallyDrop::new(block),
                idx,
            },
        )
    }

    pub(super) fn from_swapchain_image(
        owner: WeakDevice,
        handle: vk::Image,
        view: vk::ImageView,
        view_idx: usize,
        extent: impl Into<ImageExtent>,
        format: PixelFormat,
        usage: ImageUsage,
    ) -> Self {
        Image::build(
            owner,
            handle,
            view,
            view_idx,
            extent,
            format,
            usage,
            1,
            1,
            Flavor::Swapchain,
        )
    }

    /// Creates a null/invalid Image for use when device OOM occurs.
    pub(super) fn null(
        extent: ImageExtent,
        format: PixelFormat,
        usage: ImageUsage,
        layers: u32,
        levels: u32,
    ) -> Self {
        let desc = ViewDesc {
            format,
            layers: (0..layers).into(),
            levels: (0..levels).into(),
            swizzle: Swizzle::IDENTITY,
        };
        Image {
            handle: vk::Image::null(),
            view: vk::ImageView::null(),
            inner: Arc::new(Inner {
                data: Arc::new(ImageData {
                    owner: WeakDevice::null(),
                    format,
                    usage,
                    extent,
                    layers,
                    levels,
                    flavor: Flavor::Swapchain,
                    views: Mutex::new(hashbrown::HashMap::new()),
                }),
                desc,
                extent,
                usage,
                owner: WeakDevice::null(),
            }),
        }
    }

    #[inline]
    pub(super) fn handle(&self) -> vk::Image {
        self.handle
    }

    #[inline]
    pub(super) fn view_handle(&self) -> vk::ImageView {
        self.view
    }

    #[inline]
    pub(super) fn base_layer(&self) -> u32 {
        self.inner.desc.layers.start
    }

    #[inline]
    pub(super) fn layer_count(&self) -> u32 {
        self.inner.desc.layers.end - self.inner.desc.layers.start
    }

    #[inline]
    pub(super) fn base_level(&self) -> u32 {
        self.inner.desc.levels.start
    }

    #[inline]
    pub(super) fn level_count(&self) -> u32 {
        self.inner.desc.levels.end - self.inner.desc.levels.start
    }
}

impl crate::traits::Resource for Image {}

#[hidden_trait::expose]
impl crate::traits::Image for Image {
    #[inline]
    fn format(&self) -> PixelFormat {
        self.inner.desc.format
    }

    #[inline]
    fn extent(&self) -> ImageExtent {
        self.inner.extent
    }

    #[inline]
    fn layers(&self) -> u32 {
        let layers = self.inner.desc.layers;
        layers.end - layers.start
    }

    #[inline]
    fn levels(&self) -> u32 {
        let levels = self.inner.desc.levels;
        levels.end - levels.start
    }

    #[inline]
    fn usage(&self) -> ImageUsage {
        self.inner.usage
    }

    #[inline]
    fn view(&self, device: &Device, desc: ViewDesc) -> Image {
        let levels_start = desc.levels.start + self.inner.desc.levels.start;
        let levels_end = desc.levels.end + self.inner.desc.levels.start;
        let layers_start = desc.layers.start + self.inner.desc.layers.start;
        let layers_end = desc.layers.end + self.inner.desc.layers.start;

        assert!(levels_end <= self.inner.data.levels);
        assert!(layers_end <= self.inner.data.layers);

        let desc = ViewDesc {
            levels: (levels_start..levels_end).into(),
            layers: (layers_start..layers_end).into(),
            ..desc
        };

        if self.handle.is_null() || self.view.is_null() {
            let base_level = desc.levels.start + self.inner.desc.levels.start;
            return Image {
                handle: vk::Image::null(),
                view: vk::ImageView::null(),
                inner: Arc::new(Inner {
                    data: self.inner.data.clone(),
                    desc,
                    extent: self.inner.data.extent.at_level(base_level),
                    usage: self.inner.usage,
                    owner: self.inner.owner.clone(),
                }),
            };
        }

        if self.inner.desc == desc {
            return self.clone();
        }

        let view = match self.inner.data.views.lock().entry(desc) {
            Entry::Occupied(entry) => entry.get().0,
            Entry::Vacant(entry) => {
                match device.new_image_view(self.handle, self.inner.extent.into_ash(), desc) {
                    Ok((view, idx)) => entry.insert((view, idx)).0,
                    Err(OutOfMemory) => {
                        device.set_oom();
                        return Image {
                            handle: vk::Image::null(),
                            view: vk::ImageView::null(),
                            inner: Arc::new(Inner {
                                data: self.inner.data.clone(),
                                desc,
                                extent: self.inner.data.extent.at_level(desc.levels.start),
                                usage: self.inner.usage,
                                owner: self.inner.owner.clone(),
                            }),
                        };
                    }
                }
            }
        };

        Image {
            handle: self.handle,
            view,
            inner: Arc::new(Inner {
                data: self.inner.data.clone(),
                desc,
                extent: self.inner.data.extent.at_level(desc.levels.start),
                usage: self.inner.usage,
                owner: self.inner.owner.clone(),
            }),
        }
    }

    #[inline]
    fn detached(&self) -> bool {
        // If strong is 1, it cannot be changed by another thread if called owns
        // mutable reference to self
        // since there are no weaks.
        debug_assert_eq!(Arc::weak_count(&self.inner), 0, "No weak refs allowed");
        debug_assert_eq!(Arc::weak_count(&self.inner.data), 0, "No weak refs allowed");
        Arc::strong_count(&self.inner) == 1 && Arc::strong_count(&self.inner.data) == 1
    }
}

impl ArgumentsField<Automatic> for Image {
    const KIND: ArgumentKind = ArgumentKind::SampledImage;
    const SIZE: Option<NonZero<usize>> = None;
    const OFFSET: usize = 0;
    const STRIDE: usize = size_of::<vk::DescriptorImageInfo>();

    type Update = <Self as ArgumentsField<Sampled>>::Update;

    #[inline]
    fn update(&self) -> <Self as ArgumentsField<Sampled>>::Update {
        <Self as ArgumentsField<Sampled>>::update(self)
    }

    #[inline]
    fn add_refs(&self, refs: &mut Refs) {
        refs.add_image(self.clone());
    }
}

impl ArgumentsField<Sampled> for Image {
    const KIND: ArgumentKind = ArgumentKind::SampledImage;
    const SIZE: Option<NonZero<usize>> = None;
    const OFFSET: usize = 0;
    const STRIDE: usize = size_of::<vk::DescriptorImageInfo>();

    type Update = vk::DescriptorImageInfo;

    #[inline]
    fn update(&self) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view: self.view,
            image_layout: vk::ImageLayout::GENERAL,
        }
    }

    #[inline]
    fn add_refs(&self, refs: &mut Refs) {
        refs.add_image(self.clone());
    }
}

impl ArgumentsField<Storage> for Image {
    const KIND: ArgumentKind = ArgumentKind::StorageImage;
    const SIZE: Option<NonZero<usize>> = None;
    const OFFSET: usize = 0;
    const STRIDE: usize = size_of::<vk::DescriptorImageInfo>();

    type Update = vk::DescriptorImageInfo;

    #[inline]
    fn update(&self) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view: self.view,
            image_layout: vk::ImageLayout::GENERAL,
        }
    }

    #[inline]
    fn add_refs(&self, refs: &mut Refs) {
        refs.add_image(self.clone());
    }
}

macro_rules! forward_image_arguments_field {
    ($ty:ident, $marker:ident) => {
        impl<const D: usize> ArgumentsField<$marker> for $ty<D> {
            const KIND: ArgumentKind = <Image as ArgumentsField<$marker>>::KIND;
            const SIZE: Option<NonZero<usize>> = <Image as ArgumentsField<$marker>>::SIZE;
            const OFFSET: usize = <Image as ArgumentsField<$marker>>::OFFSET;
            const STRIDE: usize = <Image as ArgumentsField<$marker>>::STRIDE;

            type Update = <Image as ArgumentsField<$marker>>::Update;

            #[inline]
            fn update(&self) -> Self::Update {
                <Image as ArgumentsField<$marker>>::update(self.as_image())
            }

            #[inline]
            fn add_refs(&self, refs: &mut Refs) {
                <Image as ArgumentsField<$marker>>::add_refs(self.as_image(), refs)
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
