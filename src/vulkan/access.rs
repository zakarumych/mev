use crate::generic::{PipelineStage, PipelineStages};

pub(super) fn access_for_stage(stage: PipelineStage) -> ash::vk::AccessFlags {
    match stage {
        PipelineStage::DrawIndirect => ash::vk::AccessFlags::INDIRECT_COMMAND_READ,
        PipelineStage::VertexInput => {
            ash::vk::AccessFlags::INDEX_READ | ash::vk::AccessFlags::VERTEX_ATTRIBUTE_READ
        }
        PipelineStage::VertexShader => {
            ash::vk::AccessFlags::UNIFORM_READ
                | ash::vk::AccessFlags::SHADER_READ
                | ash::vk::AccessFlags::SHADER_WRITE
        }
        PipelineStage::EarlyFragmentTest => {
            ash::vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | ash::vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
        }
        PipelineStage::FragmentShader => {
            ash::vk::AccessFlags::UNIFORM_READ
                | ash::vk::AccessFlags::INPUT_ATTACHMENT_READ
                | ash::vk::AccessFlags::SHADER_READ
                | ash::vk::AccessFlags::SHADER_WRITE
        }
        PipelineStage::LateFragmentTest => {
            ash::vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | ash::vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
        }
        PipelineStage::ColorOutput => {
            ash::vk::AccessFlags::COLOR_ATTACHMENT_READ
                | ash::vk::AccessFlags::COLOR_ATTACHMENT_WRITE
        }
        PipelineStage::ComputeShader => {
            ash::vk::AccessFlags::UNIFORM_READ
                | ash::vk::AccessFlags::SHADER_READ
                | ash::vk::AccessFlags::SHADER_WRITE
        }
        PipelineStage::Transfer => {
            ash::vk::AccessFlags::TRANSFER_READ | ash::vk::AccessFlags::TRANSFER_WRITE
        }
    }
}

pub(super) fn access_for_stages(stages: PipelineStages) -> ash::vk::AccessFlags {
    let mut access = ash::vk::AccessFlags::empty();

    if stages.contains(PipelineStages::DRAW_INDIRECT) {
        access |= ash::vk::AccessFlags::INDIRECT_COMMAND_READ;
    }
    if stages.contains(PipelineStages::VERTEX_INPUT) {
        access |= ash::vk::AccessFlags::INDEX_READ | ash::vk::AccessFlags::VERTEX_ATTRIBUTE_READ;
    }
    if stages.intersects(
        PipelineStages::VERTEX_SHADER
            | PipelineStages::FRAGMENT_SHADER
            | PipelineStages::COMPUTE_SHADER,
    ) {
        // UNIFORM_READ is a distinct access from SHADER_READ and covers uniform buffer reads.
        //
        // Omitting it leaves the constant cache stale on hardware that keeps uniform reads in a separate cache.
        access |= ash::vk::AccessFlags::UNIFORM_READ
            | ash::vk::AccessFlags::SHADER_READ
            | ash::vk::AccessFlags::SHADER_WRITE;
    }
    if stages.contains(PipelineStages::FRAGMENT_SHADER) {
        access |= ash::vk::AccessFlags::INPUT_ATTACHMENT_READ;
    }
    if stages.intersects(PipelineStages::EARLY_FRAGMENT_TEST | PipelineStages::LATE_FRAGMENT_TEST) {
        access |= ash::vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
            | ash::vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
    }
    if stages.intersects(PipelineStages::COLOR_OUTPUT) {
        access |= ash::vk::AccessFlags::COLOR_ATTACHMENT_READ
            | ash::vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
    }
    if stages.contains(PipelineStages::TRANSFER) {
        access |= ash::vk::AccessFlags::TRANSFER_READ | ash::vk::AccessFlags::TRANSFER_WRITE;
    }
    access
}
