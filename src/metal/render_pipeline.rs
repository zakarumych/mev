use std::{fmt, sync::Arc};

use super::shader::Bindings;

#[derive(Clone)]
pub struct RenderPipeline {
    state: metal::RenderPipelineState,
    primitive: metal::MTLPrimitiveType,
    vertex_bindings: Option<Arc<Bindings>>,
    fragment_bindings: Option<Arc<Bindings>>,
    vertex_buffers_count: u32,
}

unsafe impl Send for RenderPipeline {}
unsafe impl Sync for RenderPipeline {}

impl RenderPipeline {
    pub(super) fn new(
        state: metal::RenderPipelineState,
        primitive: metal::MTLPrimitiveType,
        vertex_bindings: Option<Arc<Bindings>>,
        fragment_bindings: Option<Arc<Bindings>>,
        vertex_buffers_count: u32,
    ) -> Self {
        RenderPipeline {
            state,
            primitive,
            vertex_bindings,
            fragment_bindings,
            vertex_buffers_count,
        }
    }

    pub(super) fn metal(&self) -> &metal::RenderPipelineState {
        &self.state
    }

    pub(super) fn primitive(&self) -> metal::MTLPrimitiveType {
        self.primitive
    }

    pub(super) fn vertex_bindings(&self) -> Option<Arc<Bindings>> {
        self.vertex_bindings.clone()
    }

    pub(super) fn fragment_bindings(&self) -> Option<Arc<Bindings>> {
        self.fragment_bindings.clone()
    }

    pub(super) fn vertex_buffers_count(&self) -> u32 {
        self.vertex_buffers_count
    }
}

#[derive(Debug)]
pub enum CreatePipelineErrorKind {
    InvalidShaderEntry,
    FailedToBuildPipeline(String),
}

impl fmt::Display for CreatePipelineErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CreatePipelineErrorKind::InvalidShaderEntry => {
                write!(f, "Invalid shader entry point")
            }
            CreatePipelineErrorKind::FailedToBuildPipeline(err) => {
                write!(f, "Failed to build pipeline: {}", err)
            }
        }
    }
}
