use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLAccelerationStructure;

/// Bottom-level acceleration structure.
/// Contains ray-tracing acceleration structure for geometry.
/// Created for triangle-meshes or procedural geometry.
#[derive(Clone)]
pub struct Blas {
    accel: Retained<ProtocolObject<dyn MTLAccelerationStructure>>,
}

impl Blas {
    pub(super) fn new(accel: Retained<ProtocolObject<dyn MTLAccelerationStructure>>) -> Self {
        Blas { accel }
    }
}

/// Top-level acceleration structure.
/// Contains ray-tracing acceleration structure for instances.
/// Created for instances of bottom-level acceleration structures.
#[derive(Clone)]
pub struct Tlas {
    accel: Retained<ProtocolObject<dyn MTLAccelerationStructure>>,
}

impl Tlas {
    pub(super) fn new(accel: Retained<ProtocolObject<dyn MTLAccelerationStructure>>) -> Self {
        Tlas { accel }
    }
}
