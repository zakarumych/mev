use std::{num::NonZero, ptr::NonNull};

use ash::vk::Handle;

use crate::generic::{ArgumentGroupLayout, ArgumentKind, ArgumentsSealed};

use super::{
    command::{ComputeCommandEncoder, RenderCommandEncoder},
    refs::Refs,
};

impl ComputeCommandEncoder<'_> {
    pub(crate) fn with_buffer_arguments(
        &mut self,
        group: u32,
        buffers: &[(ArgumentKind, crate::BufferSlice<'_>)],
    ) {
        if self.handle().is_null() || self.device().get_error().is_err() {
            return;
        }
        let Some(layout) = self.current_layout().cloned() else {
            panic!("Argument binding requires a pipeline to be bound to the encoder");
        };

        let infos: Vec<_> = buffers
            .iter()
            .map(|(_, slice)| {
                ash::vk::DescriptorBufferInfo::default()
                    .buffer(slice.buffer().handle())
                    .offset(slice.offset() as u64)
                    .range(slice.size() as u64)
            })
            .collect();
        let writes: Vec<_> = buffers
            .iter()
            .zip(&infos)
            .enumerate()
            .map(|(binding, ((kind, _), info))| {
                assert!(
                    matches!(
                        kind,
                        ArgumentKind::UniformBuffer | ArgumentKind::StorageBuffer
                    ),
                    "Only buffer arguments can be bound"
                );
                ash::vk::WriteDescriptorSet::default()
                    .dst_binding(binding as u32)
                    .descriptor_type(descriptor_type(*kind))
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();

        unsafe {
            self.device().push_descriptor().cmd_push_descriptor_set(
                self.handle(),
                ash::vk::PipelineBindPoint::COMPUTE,
                layout.handle(),
                group,
                &writes,
            );
        }
        for (_, slice) in buffers {
            self.refs_mut().add_buffer(slice.buffer().clone());
        }
    }
}

#[doc(hidden)]
pub trait Arguments: 'static {
    const LAYOUT: ArgumentGroupLayout<'static>;

    /// Descriptor updates matching template created from this type.
    type Update: Copy;

    /// Create a descriptor update template for this type.
    /// The result must not depend on the value.
    fn template_entries() -> &'static [ash::vk::DescriptorUpdateTemplateEntry];

    /// Fill descriptor update template with the value.
    fn update(&self) -> Self::Update;

    /// Add references to descriptors into the `Refs` object.
    fn add_refs(&self, refs: &mut Refs);
}

impl<T> ArgumentsSealed for T where T: Arguments {}
impl<T> crate::generic::Arguments for T
where
    T: Arguments,
{
    const LAYOUT: ArgumentGroupLayout<'static> = T::LAYOUT;

    #[inline]
    fn bind_render(&self, group: u32, encoder: &mut RenderCommandEncoder) {
        let Some(layout) = encoder.current_layout() else {
            panic!("Argument binding requires a pipeline to be bound to the encoder");
        };

        let device = encoder.device();

        if device.get_error().is_err() {
            return;
        }

        let Ok(template) = device.get_descriptor_update_template::<Self>(
            Self::template_entries(),
            ash::vk::PipelineBindPoint::GRAPHICS,
            layout,
            group,
        ) else {
            device.set_oom();
            return;
        };

        let update = self.update();

        unsafe {
            device
                .push_descriptor()
                .cmd_push_descriptor_set_with_template(
                    encoder.handle(),
                    template,
                    layout.handle(),
                    group,
                    &update as *const _ as *const _,
                )
        }

        self.add_refs(encoder.refs_mut());
    }

    #[inline]
    fn bind_compute(&self, group: u32, encoder: &mut ComputeCommandEncoder) {
        let Some(layout) = encoder.current_layout() else {
            panic!("Argument binding requires a pipeline to be bound to the encoder");
        };

        let device = encoder.device();

        if device.get_error().is_err() {
            return;
        }

        let Ok(template) = device.get_descriptor_update_template::<Self>(
            Self::template_entries(),
            ash::vk::PipelineBindPoint::COMPUTE,
            layout,
            group,
        ) else {
            device.set_oom();
            return;
        };

        let update = self.update();

        unsafe {
            device
                .push_descriptor()
                .cmd_push_descriptor_set_with_template(
                    encoder.handle(),
                    template,
                    layout.handle(),
                    group,
                    &update as *const _ as *const _,
                )
        }

        self.add_refs(encoder.refs_mut());
    }
}

#[doc(hidden)]
pub trait ArgumentsField<T>: 'static {
    const KIND: ArgumentKind;
    const SIZE: Option<NonZero<usize>>;
    const OFFSET: usize;
    const STRIDE: usize;

    type Update: Copy;

    fn update(&self) -> Self::Update;

    /// Add references to descriptors into the `Refs` object.
    fn add_refs(&self, refs: &mut Refs);
}

impl<T, F> crate::generic::ArgumentsField<T> for F
where
    T: ArgumentsSealed,
    F: ArgumentsField<T> + ArgumentsSealed,
{
    const KIND: ArgumentKind = F::KIND;
    const SIZE: Option<NonZero<usize>> = F::SIZE;
}

#[doc(hidden)]
pub const fn descriptor_type(kind: ArgumentKind) -> ash::vk::DescriptorType {
    match kind {
        // ArgumentKind::Constant => ash::vk::DescriptorType::INLINE_UNIFORM_BLOCK,
        ArgumentKind::Sampler => ash::vk::DescriptorType::SAMPLER,
        ArgumentKind::UniformBuffer => ash::vk::DescriptorType::UNIFORM_BUFFER,
        ArgumentKind::StorageBuffer => ash::vk::DescriptorType::STORAGE_BUFFER,
        ArgumentKind::SampledImage => ash::vk::DescriptorType::SAMPLED_IMAGE,
        ArgumentKind::StorageImage => ash::vk::DescriptorType::STORAGE_IMAGE,
    }
}
