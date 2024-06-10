/// Bottom-level acceleration structure.
/// Contains ray-tracing acceleration structure for geometry.
/// Created for triangle-meshes or procedural geometry.
#[derive(Clone)]
pub struct Blas {
    accel: metal::AccelerationStructure,
}

impl Blas {
    pub(super) fn new(accel: metal::AccelerationStructure) -> Self {
        Blas { accel }
    }
}

/// Top-level acceleration structure.
/// Contains ray-tracing acceleration structure for instances.
/// Created for instances of bottom-level acceleration structures.
#[derive(Clone)]
pub struct Tlas {
    accel: metal::AccelerationStructure,
}

impl Tlas {
    pub(super) fn new(accel: metal::AccelerationStructure) -> Self {
        Tlas { accel }
    }
}
