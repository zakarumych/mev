/// Stages in the rendering pipeline.
#[repr(u16)]
pub enum PipelineStage {
    /// Stage at which indirect draw commands are read from buffer.
    DrawIndirect,

    /// Stage at which vertex data is read from buffer.
    VertexInput,

    /// Stage at which vertex shader is executed.
    VertexShader,

    /// Stage at which early fragment tests are performed.
    EarlyFragmentTest,

    /// Stage at which fragment shader is executed.
    FragmentShader,

    /// Stage at which late fragment tests are performed.
    LateFragmentTest,

    /// Stage at which color data is written to buffer.
    ColorOutput,

    /// Stage at which compute shader is executed.
    ComputeShader,

    /// Stage at which transfer operations are performed.
    Transfer,
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct PipelineStages: u32 {
        /// Bit for [`DrawIndirect`](PipelineStage::DrawIndirect) stage.
        const DRAW_INDIRECT = 1 << PipelineStage::DrawIndirect as u32;
        /// Bit for [`VertexInput`](PipelineStage::VertexInput) stage.
        const VERTEX_INPUT = 1 << PipelineStage::VertexInput as u32;
        /// Bit for [`VertexShader`](PipelineStage::VertexShader) stage.
        const VERTEX_SHADER = 1 << PipelineStage::VertexShader as u32;
        /// Bit for [`EarlyFragmentTest`](PipelineStage::EarlyFragmentTest) stage.
        const EARLY_FRAGMENT_TEST = 1 << PipelineStage::EarlyFragmentTest as u32;
        /// Bit for [`FragmentShader`](PipelineStage::FragmentShader) stage.
        const FRAGMENT_SHADER = 1 << PipelineStage::FragmentShader as u32;
        /// Bit for [`LateFragmentTest`](PipelineStage::LateFragmentTest) stage.
        const LATE_FRAGMENT_TEST = 1 << PipelineStage::LateFragmentTest as u32;
        /// Bit for [`ColorOutput`](PipelineStage::ColorOutput) stage.
        const COLOR_OUTPUT = 1 << PipelineStage::ColorOutput as u32;
        /// Bit for [`ComputeShader`](PipelineStage::ComputeShader) stage.
        const COMPUTE_SHADER = 1 << PipelineStage::ComputeShader as u32;
        /// Bit for [`Transfer`](PipelineStage::Transfer) stage.
        const TRANSFER = 1 << PipelineStage::Transfer as u32;
    }
}
