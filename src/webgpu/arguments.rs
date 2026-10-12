use std::num::NonZero;

use crate::generic::{ArgumentGroupLayout, ArgumentKind, ArgumentsSealed};

use super::{ComputeCommandEncoder, RenderCommandEncoder};

impl ComputeCommandEncoder<'_> {
    pub(crate) fn with_buffer_arguments(
        &mut self,
        group: u32,
        buffers: &[(ArgumentKind, crate::BufferSlice<'_>)],
    ) {
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, (kind, slice))| {
                assert!(
                    matches!(
                        kind,
                        ArgumentKind::UniformBuffer | ArgumentKind::StorageBuffer
                    ),
                    "Only buffer arguments can be bound"
                );
                wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: slice.buffer().wgpu(),
                        offset: slice.offset() as u64,
                        size: NonZero::new(slice.size() as u64),
                    }),
                }
            })
            .collect();
        let layout = self.bind_group_layout(group).clone();
        let bind_group = self
            .wgpu_device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &entries,
            });
        self.set_bind_group(group, &bind_group);
    }
}

pub trait Arguments: 'static {
    const LAYOUT: ArgumentGroupLayout<'static>;

    fn bind_render(&self, group: u32, encoder: &mut RenderCommandEncoder);
    fn bind_compute(&self, group: u32, encoder: &mut ComputeCommandEncoder);
}

impl<T> ArgumentsSealed for T where T: Arguments {}

impl<T> crate::generic::Arguments for T
where
    T: Arguments,
{
    const LAYOUT: ArgumentGroupLayout<'static> = T::LAYOUT;

    #[inline(always)]
    fn bind_render(&self, group: u32, encoder: &mut RenderCommandEncoder) {
        Arguments::bind_render(self, group, encoder)
    }

    #[inline(always)]
    fn bind_compute(&self, group: u32, encoder: &mut ComputeCommandEncoder) {
        Arguments::bind_compute(self, group, encoder)
    }
}

#[doc(hidden)]
pub trait ArgumentsField<T>: 'static {
    const KIND: ArgumentKind;
    const SIZE: Option<NonZero<usize>>;

    fn as_binding_resource(&self) -> wgpu::BindingResource<'_>;
}

impl<T, F> crate::generic::ArgumentsField<T> for F
where
    T: ArgumentsSealed,
    F: ArgumentsField<T> + ArgumentsSealed,
{
    const KIND: ArgumentKind = F::KIND;
    const SIZE: Option<NonZero<usize>> = F::SIZE;
}
