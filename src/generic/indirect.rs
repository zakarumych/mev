/// Arguments for an indirect (non-indexed) draw call.
/// Layout matches `VkDrawIndirectCommand` / `MTLDrawPrimitivesIndirectArguments` /
/// `wgpu::util::DrawIndirectArgs` byte-for-byte (16 bytes, tightly packed, no padding).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndirectCommand {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

/// Arguments for an indirect indexed draw call.
/// Layout matches `VkDrawIndexedIndirectCommand` / `MTLDrawIndexedPrimitivesIndirectArguments` /
/// `wgpu::util::DrawIndexedIndirectArgs` byte-for-byte (20 bytes, tightly packed, no padding).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndexedIndirectCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
}

/// Arguments for an indirect compute dispatch.
/// Layout matches `VkDispatchIndirectCommand` / `MTLDispatchThreadgroupsIndirectArguments` /
/// `wgpu::util::DispatchIndirectArgs` byte-for-byte (12 bytes, tightly packed, no padding).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DispatchIndirectCommand {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}
