use super::{BufferSlice, VertexFormat};

/// Memory sizes required for acceleration structure operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AccelerationStructureSizes {
    /// Size of the acceleration structure in bytes.
    /// 
    /// Allocate this amount of memory in buffer that will contain the acceleration structure.
    pub size: usize,

    /// Size of the scratch memory in bytes.
    /// 
    /// Allocate this amount of memory in buffer that will be used during acceleration structure build.
    pub scratch_size: usize,

    /// Size of the update scratch memory in bytes.
    /// 
    /// Allocate this amount of memory in buffer that will be used during acceleration structure update.
    pub update_scratch_size: usize,
}

/// Performance hints for acceleration structure operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AccelerationStructurePerformance {
    /// Default preference for acceleration structure operation.
    Default,

    /// Prefer fast trace performance.
    /// 
    /// This may result in slower build times.
    /// Use this when acceleration structure will be used repeatedly.
    FastTrace,

    /// Prefer fast build performance.
    /// 
    /// This may result in slower trace times.
    /// Use this when acceleration structure will be rebuilt frequently.
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

/// Description of a bottom-level acceleration structure triangle-based geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlasTriangles<'a> {
    /// Flag to indicate if geometry is opaque.
    /// If true, tracing will stop on hit to this geometry.
    /// Otherwise opaque test may be invoked.
    /// 
    /// Set this flag is there are not non-opaque surfaces using this geometry.
    pub opaque: bool,

    /// Buffer slice containing triangle indices.
    /// If none, indices are not used and vertices are used directly.
    pub indices: Option<BufferSlice<'a>>,

    /// Buffer slice containing triangle vertices.
    pub vertices: BufferSlice<'a>,

    /// Stride between vertices in bytes.
    pub vertex_stride: usize,

    /// Format of the vertices.
    pub vertex_format: VertexFormat,

    /// Buffer slice containing transform matrix.
    pub transform: Option<BufferSlice<'a>>,
}

/// Description of a bottom-level acceleration structure axis-aligned bounding box-based geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlasAABBs<'a> {
    /// Flag to indicate if geometry is opaque.
    pub opaque: bool,

    /// Buffer slice containing AABBs.
    pub boxes: BufferSlice<'a>,

    /// Stride between AABBs in bytes.
    pub box_stride: usize,
}

/// Description of a bottom-level acceleration structure geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlasGeometryDesc<'a> {
    /// Triangle-based geometry.
    Triangles(BlasTriangles<'a>),

    /// Axis-aligned bounding box-based geometry.
    AABBs(BlasAABBs<'a>),
}

/// Flags for bottom-level acceleration structure.
/// Reserved for future use.
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
