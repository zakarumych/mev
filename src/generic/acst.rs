use super::{BufferSlice, VertexFormat};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AccelerationStructureSizes {
    pub size: usize,
    pub scratch_size: usize,
    pub update_scratch_size: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AccelerationStructurePerformance {
    Default,
    FastTrace,
    FastBuild,
}

bitflags::bitflags! {
    /// These flags are used to specify the build properties of an acceleration structure.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct AccelerationStructureBuildFlags: u32 {
        /// Specifies that built acceleration structure could be used as a source
        /// for build with update operation.
        const ALLOW_UPDATE = 0x1;

        /// Specifies that built acceleration structure could be used as a source
        /// for copy operation with `Compact` mode.
        const ALLOW_COMPACTION = 0x2;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlasTriangles<'a> {
    pub opaque: bool,
    pub indices: Option<BufferSlice<'a>>,
    pub vertices: BufferSlice<'a>,
    pub vertex_stride: usize,
    pub vertex_format: VertexFormat,
    pub transform: Option<BufferSlice<'a>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlasAABBs<'a> {
    pub opaque: bool,
    pub boxes: BufferSlice<'a>,
    pub box_stride: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlasGeometryDesc<'a> {
    Triangles(BlasTriangles<'a>),
    AABBs(BlasAABBs<'a>),
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct BlasFlags: u32 {}
}

/// Description of a bottom-level acceleration structure
/// Contains flags and size of the acceleration structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlasDesc {
    pub flags: BlasFlags,
    pub size: usize,
}

/// Description of a bottom-level acceleration structure build.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlasBuildDesc<'a> {
    pub performance: AccelerationStructurePerformance,
    pub flags: AccelerationStructureBuildFlags,
    pub geometry: &'a [BlasGeometryDesc<'a>],
}

pub struct TlasBuildDesc {
    pub flags: AccelerationStructureBuildFlags,
    pub instances: Vec<TlasInstanceDesc>,
}

pub struct TlasInstanceDesc {}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct TlasFlags: u32 {}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TlasDesc {
    pub flags: TlasFlags,
    pub size: usize,
}
