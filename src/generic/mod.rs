// mod _arguments;
mod acst;
mod arguments;
mod buffer;
mod compute_pipeline;
mod data;
mod feature;
mod format;
mod image;
mod instance;
mod queue;
mod render;
mod render_pipeline;
mod sampler;
mod shader;
mod stages;
mod surface;

use std::{error::Error, fmt};

pub use self::{
    acst::{
        AccelerationStructureBuildFlags, AccelerationStructurePerformance, BlasAABBs,
        BlasBuildDesc, BlasDesc, BlasFlags, BlasGeometryDesc, BlasTriangles, TlasBuildDesc,
        TlasDesc, TlasFlags, TlasInstanceDesc,
    },
    arguments::{
        ArgumentGroupLayout, ArgumentKind, ArgumentLayout, Arguments, ArgumentsField, Automatic,
        /*Constant,*/ Sampled, Storage, Uniform,
    },
    buffer::{AsBufferSlice, BufferDesc, BufferInitDesc, BufferSlice, BufferUsage, Memory},
    compute_pipeline::ComputePipelineDesc,
    data::*,
    feature::Features,
    format::{PixelFormat, VertexFormat},
    image::{ComponentSwizzle, ImageDesc, ImageExtent, ImageUsage, Swizzle, ViewDesc},
    instance::{
        Capabilities, CreateError, DeviceCapabilities, DeviceDesc, FamilyCapabilities, LoadError,
    },
    queue::QueueFlags,
    render::{AttachmentDesc, ClearColor, ClearDepthStencil, LoadOp, RenderPassDesc, StoreOp},
    render_pipeline::{
        Blend, BlendDesc, BlendFactor, BlendOp, ColorTargetDesc, CompareFunction,
        CreatePipelineError, Culling, DepthStencilDesc, FrontFace, PrimitiveTopology, RasterDesc,
        RenderPipelineDesc, VertexAttributeDesc, VertexLayoutDesc, VertexStepMode, WriteMask,
    },
    sampler::{AddressMode, Filter, MipMapMode, SamplerDesc},
    shader::{
        CreateLibraryError, LibraryDesc, LibraryInput, Shader, ShaderLanguage, ShaderSource,
        ShaderStage, ShaderStages,
    },
    stages::{PipelineStage, PipelineStages},
    surface::SurfaceError,
};

pub(crate) use self::{
    arguments::ArgumentsSealed,
    shader::{parse_shader, ShaderCompileError},
};

/// Error that can happen when device's memory is exhausted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OutOfMemory;

impl fmt::Display for OutOfMemory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "out of memory")
    }
}

impl Error for OutOfMemory {}

pub enum DeviceError {
    OutOfMemory,
    DeviceLost,
}

impl DeviceError {
    pub fn abort_on_device_lost(self) -> OutOfMemory {
        match self {
            DeviceError::OutOfMemory => OutOfMemory,
            DeviceError::DeviceLost => panic!("device lost"),
        }
    }
}

impl From<OutOfMemory> for DeviceError {
    #[inline(always)]
    fn from(_: OutOfMemory) -> Self {
        DeviceError::OutOfMemory
    }
}

impl fmt::Debug for DeviceError {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceError::OutOfMemory => write!(f, "DeviceError::OutOfMemory"),
            DeviceError::DeviceLost => write!(f, "DeviceError::DeviceLost"),
        }
    }
}

impl fmt::Display for DeviceError {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceError::OutOfMemory => write!(f, "out of memory"),
            DeviceError::DeviceLost => write!(f, "device lost"),
        }
    }
}

impl Error for DeviceError {}

pub trait Zero {
    const ZERO: Self;
}

impl Zero for u32 {
    const ZERO: Self = 0;
}

impl Zero for i32 {
    const ZERO: Self = 0;
}

impl Zero for f32 {
    const ZERO: Self = 0.0;
}

pub trait One {
    const ONE: Self;
}

impl One for u32 {
    const ONE: Self = 1;
}

impl One for i32 {
    const ONE: Self = 1;
}

impl One for f32 {
    const ONE: Self = 1.0;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Offset<T, const D: usize>(pub [T; D]);

impl<T, const D: usize> Offset<T, D>
where
    T: Zero,
{
    pub const ZERO: Self = Self([T::ZERO; D]);
}

pub type Offset1<T = i32> = Offset<T, 1>;
pub type Offset2<T = i32> = Offset<T, 2>;
pub type Offset3<T = i32> = Offset<T, 3>;

impl<T: Copy> Offset1<T> {
    pub const fn new(x: T) -> Self {
        Self([x])
    }

    pub const fn x(&self) -> T {
        self.0[0]
    }

    pub const fn to_2d(&self) -> Offset2<T>
    where
        T: Zero,
    {
        let [x] = self.0;
        Offset2::new(x, T::ZERO)
    }

    pub const fn to_3d(&self) -> Offset3<T>
    where
        T: Zero,
    {
        let [x] = self.0;
        Offset3::new(x, T::ZERO, T::ZERO)
    }
}

impl<T: Copy> Offset2<T> {
    pub const fn new(x: T, y: T) -> Self {
        Self([x, y])
    }

    pub const fn x(&self) -> T {
        self.0[0]
    }

    pub const fn y(&self) -> T {
        self.0[1]
    }

    pub const fn to_1d(&self) -> Offset1<T> {
        let [x, _] = self.0;
        Offset1::new(x)
    }

    pub const fn to_3d(&self) -> Offset3<T>
    where
        T: Zero,
    {
        let [x, y] = self.0;
        Offset3::new(x, y, T::ZERO)
    }
}

impl<T: Copy> Offset3<T> {
    pub const fn new(x: T, y: T, z: T) -> Self {
        Self([x, y, z])
    }

    pub const fn x(&self) -> T {
        self.0[0]
    }

    pub const fn y(&self) -> T {
        self.0[1]
    }

    pub const fn z(&self) -> T {
        self.0[2]
    }

    pub const fn to_1d(&self) -> Offset1<T> {
        let [x, _, _] = self.0;
        Offset1::new(x)
    }

    pub const fn to_2d(&self) -> Offset2<T> {
        let [x, y, _] = self.0;
        Offset2::new(x, y)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Extent<T, const D: usize>(pub [T; D]);

impl<T, const D: usize> Extent<T, D>
where
    T: Zero,
{
    pub const ZERO: Self = Self([T::ZERO; D]);
}

impl<T, const D: usize> Extent<T, D>
where
    T: One,
{
    pub const ONE: Self = Self([T::ONE; D]);
}

pub type Extent1<T = u32> = Extent<T, 1>;
pub type Extent2<T = u32> = Extent<T, 2>;
pub type Extent3<T = u32> = Extent<T, 3>;

impl<T: Copy> Extent1<T> {
    pub const fn new(width: T) -> Self {
        Self([width])
    }

    pub const fn width(&self) -> T {
        self.0[0]
    }

    pub const fn to_2d(&self) -> Extent2<T>
    where
        T: One,
    {
        let [width] = self.0;
        Extent2::new(width, T::ONE)
    }

    pub const fn to_3d(&self) -> Extent3<T>
    where
        T: One,
    {
        let [width] = self.0;
        Extent3::new(width, T::ONE, T::ONE)
    }
}

impl<T: Copy> Extent2<T> {
    pub const fn new(width: T, height: T) -> Self {
        Self([width, height])
    }

    pub const fn width(&self) -> T {
        self.0[0]
    }

    pub const fn height(&self) -> T {
        self.0[1]
    }

    pub const fn to_1d(&self) -> Extent1<T> {
        let [width, _] = self.0;
        Extent1::new(width)
    }

    pub const fn to_3d(&self) -> Extent3<T>
    where
        T: One,
    {
        let [width, height] = self.0;
        Extent3::new(width, height, T::ONE)
    }
}

impl<T: Copy> Extent3<T> {
    pub const fn new(width: T, height: T, depth: T) -> Self {
        Self([width, height, depth])
    }

    pub const fn width(&self) -> T {
        self.0[0]
    }

    pub const fn height(&self) -> T {
        self.0[1]
    }

    pub const fn depth(&self) -> T {
        self.0[2]
    }

    pub const fn to_1d(&self) -> Extent1<T> {
        let [width, _, _] = self.0;
        Extent1::new(width)
    }

    pub const fn to_2d(&self) -> Extent2<T> {
        let [width, height, _] = self.0;
        Extent2::new(width, height)
    }
}
