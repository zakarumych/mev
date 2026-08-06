use hashbrown::HashMap;
use parking_lot::Mutex;

use crate::{
    AddressMode, Arguments, AttachmentDesc, BlendDesc, Buffer, BufferInitDesc, BufferUsage,
    ColorTargetDesc, CommandEncoder, ComputePipeline, ComputePipelineDesc, Culling, Device,
    Extent2, Extent3, Filter, FrontFace, Image, LibraryDesc, LibraryInput, MipMapMode, Offset2,
    Offset3, PixelFormat, PrimitiveTopology, RasterDesc, RenderPassDesc, RenderPipeline,
    RenderPipelineDesc, Sampler, SamplerDesc,
};

/// Kernel that blits (copies with resampling) a rectangular region
/// of a 2D source image into a rectangular region of a 2D destination image.
///
/// Depending on the classification of the destination pixel format
/// (see [`BlitKind`]) this uses one of four pipeline shapes:
///
/// - `Float`: a compute shader that hardware-samples `src` through a
///   `sampler` and `textureStore`s into a `texture_storage_2d<FORMAT, write>`.
/// - `Srgb`: a render pipeline (fullscreen triangle + fragment shader that
///   hardware-samples `src`) drawing into the destination image as a color
///   attachment, so the sRGB gamma encoding on write is handled by fixed
///   function hardware.
/// - `Uint` / `Sint`: a compute shader that uses raw `textureLoad` (no
///   sampler) with manual, `Nearest`-only address-mode math, since WGSL
///   samplers cannot be used with integer texture formats.
pub struct Blit {
    device: Device,
    float_pipelines: Mutex<HashMap<PixelFormat, ComputePipeline>>,
    uint_pipelines: Mutex<HashMap<PixelFormat, ComputePipeline>>,
    sint_pipelines: Mutex<HashMap<PixelFormat, ComputePipeline>>,
    srgb_pipelines: Mutex<HashMap<PixelFormat, RenderPipeline>>,
    samplers: Mutex<HashMap<SamplerDesc, Sampler>>,
}

/// Classification of a `PixelFormat` for the purposes of `Blit`.
///
/// This is what makes "minimal number of pipeline permutations, maximum
/// number of supported src/dst pairs" possible: instead of one pipeline per
/// src×dst pair, there is exactly one pipeline shape per `BlitKind`, and any
/// src/dst pair that shares the same `BlitKind` (modulo the `Float`/`Srgb`
/// pairing, see [`classify`]) is supported by it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum BlitKind {
    /// Float-representable (unorm/snorm/float) formats.
    /// Handled by a compute shader that samples `src` through a `Sampler`.
    Float,

    /// sRGB formats.
    /// Handled by a render pipeline; gamma encoding on write is automatic
    /// hardware behavior of the color attachment.
    Srgb,

    /// Unsigned integer formats.
    /// Handled by a compute shader using raw `textureLoad`.
    Uint,

    /// Signed integer formats.
    /// Handled by a compute shader using raw `textureLoad`.
    Sint,
}

/// Classifies a `PixelFormat` into a [`BlitKind`], or returns `None` if the
/// format is not supported by `Blit` at all (3-channel Rgb/Bgr formats,
/// depth/stencil formats and BC-compressed formats).
fn classify(format: PixelFormat) -> Option<BlitKind> {
    match format {
        PixelFormat::R8Unorm
        | PixelFormat::R8Snorm
        | PixelFormat::R16Unorm
        | PixelFormat::R16Snorm
        | PixelFormat::R16Float
        | PixelFormat::R32Float
        | PixelFormat::Rg8Unorm
        | PixelFormat::Rg8Snorm
        | PixelFormat::Rg16Unorm
        | PixelFormat::Rg16Snorm
        | PixelFormat::Rg16Float
        | PixelFormat::Rg32Float
        | PixelFormat::Rgba8Unorm
        | PixelFormat::Rgba8Snorm
        | PixelFormat::Rgba16Unorm
        | PixelFormat::Rgba16Snorm
        | PixelFormat::Rgba16Float
        | PixelFormat::Rgba32Float
        | PixelFormat::Bgra8Unorm => Some(BlitKind::Float),

        PixelFormat::R8Srgb
        | PixelFormat::Rg8Srgb
        | PixelFormat::Rgba8Srgb
        | PixelFormat::Bgra8Srgb => Some(BlitKind::Srgb),

        PixelFormat::R8Uint
        | PixelFormat::R16Uint
        | PixelFormat::R32Uint
        | PixelFormat::Rg8Uint
        | PixelFormat::Rg16Uint
        | PixelFormat::Rg32Uint
        | PixelFormat::Rgba8Uint
        | PixelFormat::Rgba16Uint
        | PixelFormat::Rgba32Uint => Some(BlitKind::Uint),

        PixelFormat::R8Sint
        | PixelFormat::R16Sint
        | PixelFormat::R32Sint
        | PixelFormat::Rg8Sint
        | PixelFormat::Rg16Sint
        | PixelFormat::Rg32Sint
        | PixelFormat::Rgba8Sint
        | PixelFormat::Rgba16Sint
        | PixelFormat::Rgba32Sint => Some(BlitKind::Sint),

        _ => None,
    }
}

#[derive(Arguments)]
struct BlitFloatArguments {
    #[mev(uniform)]
    #[mev(compute)]
    constants: Buffer,

    #[mev(sampled)]
    #[mev(compute)]
    src: Image,

    #[mev(compute)]
    sampler: Sampler,

    #[mev(storage)]
    #[mev(compute)]
    dst: Image,
}

/// Shared argument layout for the `Uint` and `Sint` compute pipelines.
///
/// `ArgumentKind` only encodes resource *kind* (sampled image, storage
/// image, ...), not scalar type, so one Rust struct serves both pipelines -
/// the `u32` vs `i32` distinction lives entirely in the compiled WGSL text.
#[derive(Arguments)]
struct BlitIntArguments {
    #[mev(uniform)]
    #[mev(compute)]
    constants: Buffer,

    #[mev(sampled)]
    #[mev(compute)]
    src: Image,

    #[mev(storage)]
    #[mev(compute)]
    dst: Image,
}

/// Argument layout for the `Srgb` render pipeline.
/// There is no `dst` field: the destination is the render pass's color
/// attachment, not a shader-bound resource.
#[derive(Arguments)]
struct BlitRenderArguments {
    #[mev(uniform)]
    #[mev(fragment)]
    constants: Buffer,

    #[mev(sampled)]
    #[mev(fragment)]
    src: Image,

    #[mev(fragment)]
    sampler: Sampler,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod, mev::AutoDeviceRepr)]
struct BlitConstants {
    dst_offset: [i32; 2],
    dst_extent: [u32; 2],
    dst_image_size: [u32; 2],
    src_offset: [f32; 2],
    src_extent: [f32; 2],
    src_image_size: [f32; 2],
    // Per-axis `AddressMode` as `u32` (declaration-order discriminant:
    // Repeat=0, MirrorRepeat=1, ClampToEdge=2). Only read by the Uint/Sint
    // shaders - unused (but harmless) elsewhere.
    address_mode: [u32; 2],
}

/// Maps a `PixelFormat` to the WGSL storage-format keyword to use
/// for the `texture_storage_2d<FORMAT, write>` binding in the `Float`
/// compute pipeline.
///
/// Only float-representable formats are supported since the shader
/// samples `src` as `vec4<f32>` and stores it directly into `dst`.
fn wgsl_storage_format(format: PixelFormat) -> &'static str {
    match format {
        PixelFormat::R8Unorm => "r8unorm",
        PixelFormat::R8Snorm => "r8snorm",
        PixelFormat::R16Unorm => "r16unorm",
        PixelFormat::R16Snorm => "r16snorm",
        PixelFormat::R16Float => "r16float",
        PixelFormat::R32Float => "r32float",
        PixelFormat::Rg8Unorm => "rg8unorm",
        PixelFormat::Rg8Snorm => "rg8snorm",
        PixelFormat::Rg16Unorm => "rg16unorm",
        PixelFormat::Rg16Snorm => "rg16snorm",
        PixelFormat::Rg16Float => "rg16float",
        PixelFormat::Rg32Float => "rg32float",
        PixelFormat::Rgba8Unorm => "rgba8unorm",
        PixelFormat::Rgba8Snorm => "rgba8snorm",
        PixelFormat::Rgba16Unorm => "rgba16unorm",
        PixelFormat::Rgba16Snorm => "rgba16snorm",
        PixelFormat::Rgba16Float => "rgba16float",
        PixelFormat::Rgba32Float => "rgba32float",
        PixelFormat::Bgra8Unorm => "bgra8unorm",
        _ => panic!("Blit: unsupported dst pixel format {:?}", format),
    }
}

/// Maps a `PixelFormat` to the WGSL storage-format keyword to use for the
/// `texture_storage_2d<FORMAT, write>` binding in the `Uint`/`Sint` compute
/// pipelines. There is no `Bgra8Uint`/`Bgra8Sint` entry: WGSL has no such
/// storage-format keyword (naga's WGSL frontend `map_storage_format` table
/// only has `bgra8unorm`, no integer Bgra variant).
fn wgsl_int_storage_format(format: PixelFormat) -> &'static str {
    match format {
        PixelFormat::R8Uint => "r8uint",
        PixelFormat::R8Sint => "r8sint",
        PixelFormat::R16Uint => "r16uint",
        PixelFormat::R16Sint => "r16sint",
        PixelFormat::R32Uint => "r32uint",
        PixelFormat::R32Sint => "r32sint",
        PixelFormat::Rg8Uint => "rg8uint",
        PixelFormat::Rg8Sint => "rg8sint",
        PixelFormat::Rg16Uint => "rg16uint",
        PixelFormat::Rg16Sint => "rg16sint",
        PixelFormat::Rg32Uint => "rg32uint",
        PixelFormat::Rg32Sint => "rg32sint",
        PixelFormat::Rgba8Uint => "rgba8uint",
        PixelFormat::Rgba8Sint => "rgba8sint",
        PixelFormat::Rgba16Uint => "rgba16uint",
        PixelFormat::Rgba16Sint => "rgba16sint",
        PixelFormat::Rgba32Uint => "rgba32uint",
        PixelFormat::Rgba32Sint => "rgba32sint",
        _ => panic!("Blit: unsupported integer dst pixel format {:?}", format),
    }
}

fn make_float_pipeline(device: &Device, format: PixelFormat) -> ComputePipeline {
    let storage_format = wgsl_storage_format(format);

    let code = format!(
        r#"
struct Constants {{
    dst_offset: vec2<i32>,
    dst_extent: vec2<u32>,
    dst_image_size: vec2<u32>,
    src_offset: vec2<f32>,
    src_extent: vec2<f32>,
    src_image_size: vec2<f32>,
    address_mode: vec2<u32>,
}}

@group(0) @binding(0) var<uniform> constants: Constants;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var src_sampler: sampler;
@group(0) @binding(3) var dst: texture_storage_2d<{storage_format}, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let p = gid.xy;
    if (p.x >= constants.dst_extent.x || p.y >= constants.dst_extent.y) {{
        return;
    }}
    let dst_pixel = constants.dst_offset + vec2<i32>(p);
    if (dst_pixel.x < 0 || dst_pixel.y < 0 ||
        u32(dst_pixel.x) >= constants.dst_image_size.x || u32(dst_pixel.y) >= constants.dst_image_size.y) {{
        return;
    }}
    let t = (vec2<f32>(p) + vec2<f32>(0.5, 0.5)) / vec2<f32>(constants.dst_extent);
    let src_coord = constants.src_offset + t * constants.src_extent;
    let uv = src_coord / constants.src_image_size;
    let color = textureSampleLevel(src, src_sampler, uv, 0.0);
    textureStore(dst, dst_pixel, color);
}}
"#,
        storage_format = storage_format,
    );

    let library = device
        .new_shader_library(LibraryDesc {
            name: "blit_float",
            input: LibraryInput::wgsl(code),
        })
        .expect("Blit: failed to compile shader library");

    device
        .new_compute_pipeline(ComputePipelineDesc {
            name: "blit_float",
            shader: library.entry("main"),
            work_group_size: [8, 8, 1],
            constants: 0,
            arguments: &[BlitFloatArguments::LAYOUT],
        })
        .expect("Blit: failed to create compute pipeline")
}

/// Builds the `Uint`/`Sint` compute pipeline for `format`, parameterized by
/// the WGSL scalar type name (`"u32"` or `"i32"`) to use for the source
/// texture and loaded texel.
fn make_int_pipeline(device: &Device, format: PixelFormat, scalar: &str) -> ComputePipeline {
    let storage_format = wgsl_int_storage_format(format);

    let code = format!(
        r#"
struct Constants {{
    dst_offset: vec2<i32>,
    dst_extent: vec2<u32>,
    dst_image_size: vec2<u32>,
    src_offset: vec2<f32>,
    src_extent: vec2<f32>,
    src_image_size: vec2<f32>,
    address_mode: vec2<u32>,
}}

@group(0) @binding(0) var<uniform> constants: Constants;
@group(0) @binding(1) var src: texture_2d<{scalar}>;
@group(0) @binding(2) var dst: texture_storage_2d<{storage_format}, write>;

fn apply_address_mode(x: i32, size: i32, mode: u32) -> i32 {{
    if (mode == 2u) {{
        return clamp(x, 0, size - 1);
    }} else if (mode == 1u) {{
        let period = 2 * size;
        var m = x % period;
        if (m < 0) {{ m = m + period; }}
        if (m >= size) {{ m = period - 1 - m; }}
        return m;
    }} else {{
        var m = x % size;
        if (m < 0) {{ m = m + size; }}
        return m;
    }}
}}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let p = gid.xy;
    if (p.x >= constants.dst_extent.x || p.y >= constants.dst_extent.y) {{
        return;
    }}
    let dst_pixel = constants.dst_offset + vec2<i32>(p);
    if (dst_pixel.x < 0 || dst_pixel.y < 0 ||
        u32(dst_pixel.x) >= constants.dst_image_size.x || u32(dst_pixel.y) >= constants.dst_image_size.y) {{
        return;
    }}
    let t = (vec2<f32>(p) + vec2<f32>(0.5, 0.5)) / vec2<f32>(constants.dst_extent);
    let src_coord_f = constants.src_offset + t * constants.src_extent;
    let src_texel = vec2<i32>(floor(src_coord_f));
    let sx = apply_address_mode(src_texel.x, i32(constants.src_image_size.x), constants.address_mode.x);
    let sy = apply_address_mode(src_texel.y, i32(constants.src_image_size.y), constants.address_mode.y);
    let color = textureLoad(src, vec2<i32>(sx, sy), 0);
    textureStore(dst, dst_pixel, color);
}}
"#,
        scalar = scalar,
        storage_format = storage_format,
    );

    let library = device
        .new_shader_library(LibraryDesc {
            name: "blit_int",
            input: LibraryInput::wgsl(code),
        })
        .expect("Blit: failed to compile shader library");

    device
        .new_compute_pipeline(ComputePipelineDesc {
            name: "blit_int",
            shader: library.entry("main"),
            work_group_size: [8, 8, 1],
            constants: 0,
            arguments: &[BlitIntArguments::LAYOUT],
        })
        .expect("Blit: failed to create compute pipeline")
}

fn make_srgb_pipeline(device: &Device, format: PixelFormat) -> RenderPipeline {
    let code = r#"
struct Constants {
    dst_offset: vec2<i32>,
    dst_extent: vec2<u32>,
    dst_image_size: vec2<u32>,
    src_offset: vec2<f32>,
    src_extent: vec2<f32>,
    src_image_size: vec2<f32>,
    address_mode: vec2<u32>,
}

@group(0) @binding(0) var<uniform> constants: Constants;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var src_sampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let t = (frag_pos.xy - vec2<f32>(constants.dst_offset)) / vec2<f32>(constants.dst_extent);
    let src_coord = constants.src_offset + t * constants.src_extent;
    let uv = src_coord / constants.src_image_size;
    return textureSampleLevel(src, src_sampler, uv, 0.0);
}
"#;

    let library = device
        .new_shader_library(LibraryDesc {
            name: "blit_srgb",
            input: LibraryInput::wgsl(code.to_string()),
        })
        .expect("Blit: failed to compile shader library");

    device
        .new_render_pipeline(RenderPipelineDesc {
            name: "blit_srgb",
            vertex_shader: library.entry("vs_main"),
            vertex_attributes: vec![],
            vertex_layouts: vec![],
            primitive_topology: PrimitiveTopology::Triangle,
            raster: Some(RasterDesc {
                fragment_shader: Some(library.entry("fs_main")),
                color_targets: vec![ColorTargetDesc::new(format)],
                depth_stencil: None,
                front_face: FrontFace::default(),
                culling: Culling::None,
            }),
            arguments: &[BlitRenderArguments::LAYOUT],
            constants: 0,
        })
        .expect("Blit: failed to create render pipeline")
}

impl Blit {
    pub fn new(device: &Device) -> Self {
        Blit {
            device: device.clone(),
            float_pipelines: Mutex::new(HashMap::new()),
            uint_pipelines: Mutex::new(HashMap::new()),
            sint_pipelines: Mutex::new(HashMap::new()),
            srgb_pipelines: Mutex::new(HashMap::new()),
            samplers: Mutex::new(HashMap::new()),
        }
    }

    pub fn blit(
        &self,
        encoder: &mut CommandEncoder,
        src: &Image,
        src_offset: Offset3,
        src_extent: Extent3,
        dst: &Image,
        dst_offset: Offset3,
        dst_extent: Extent3,
        address_mode: [AddressMode; 3],
        filter: Filter,
    ) {
        let dst_format = dst.format();
        let dst_kind = classify(dst_format)
            .unwrap_or_else(|| panic!("Blit: unsupported dst pixel format {:?}", dst_format));

        let src_format = src.format();
        let src_kind = classify(src_format)
            .unwrap_or_else(|| panic!("Blit: unsupported src pixel format {:?}", src_format));

        let same_kind = match (dst_kind, src_kind) {
            (BlitKind::Float | BlitKind::Srgb, BlitKind::Float | BlitKind::Srgb) => true,
            (BlitKind::Uint, BlitKind::Uint) => true,
            (BlitKind::Sint, BlitKind::Sint) => true,
            _ => false,
        };

        if !same_kind {
            panic!(
                "Blit: src format {:?} is not compatible with dst format {:?} (integer/float mismatch)",
                src_format, dst_format
            );
        }

        if matches!(dst_kind, BlitKind::Uint | BlitKind::Sint) && filter == Filter::Linear {
            panic!("Blit: only Filter::Nearest is supported for integer pixel formats");
        }

        let src_image_size = src.extent().into_2d();
        let dst_image_size = dst.extent().into_2d();

        let src_offset = src_offset.to_2d();
        let dst_offset = dst_offset.to_2d();
        let src_extent = src_extent.to_2d();
        let dst_extent = dst_extent.to_2d();

        let constants = BlitConstants {
            dst_offset: [dst_offset.x(), dst_offset.y()],
            dst_extent: [dst_extent.width(), dst_extent.height()],
            dst_image_size: [dst_image_size.width(), dst_image_size.height()],
            src_offset: [src_offset.x() as f32, src_offset.y() as f32],
            src_extent: [src_extent.width() as f32, src_extent.height() as f32],
            src_image_size: [
                src_image_size.width() as f32,
                src_image_size.height() as f32,
            ],
            address_mode: [address_mode[0] as u32, address_mode[1] as u32],
        };

        // Sampler is only needed for the hardware-sampled Float/Srgb paths.
        let sampler = match dst_kind {
            BlitKind::Float | BlitKind::Srgb => {
                let sampler_desc = SamplerDesc {
                    min_filter: filter,
                    mag_filter: filter,
                    mip_map_mode: MipMapMode::Nearest,
                    address_mode,
                    anisotropy: None,
                    min_lod: 0.0,
                    max_lod: 0.0,
                    normalized: true,
                };

                let mut samplers = self.samplers.lock();
                Some(
                    samplers
                        .entry(sampler_desc)
                        .or_insert_with(|| self.device.new_sampler(sampler_desc))
                        .clone(),
                )
            }
            BlitKind::Uint | BlitKind::Sint => None,
        };

        let groups = Extent3::new(
            (dst_extent.width() + 7) / 8,
            (dst_extent.height() + 7) / 8,
            1,
        );

        match dst_kind {
            BlitKind::Float => {
                let buffer = self.device.new_buffer_init(BufferInitDesc {
                    data: bytemuck::bytes_of(&constants),
                    usage: BufferUsage::UNIFORM,
                    name: "blit_constants",
                });

                let pipeline = {
                    let mut pipelines = self.float_pipelines.lock();
                    pipelines
                        .entry(dst_format)
                        .or_insert_with(|| make_float_pipeline(&self.device, dst_format))
                        .clone()
                };

                let mut compute = encoder.compute();
                compute.with_pipeline(&pipeline);
                compute.with_arguments(
                    0,
                    &BlitFloatArguments {
                        constants: buffer,
                        src: src.clone(),
                        sampler: sampler.expect("Float blit always has a sampler"),
                        dst: dst.clone(),
                    },
                );
                compute.dispatch(groups);
            }
            BlitKind::Uint | BlitKind::Sint => {
                let buffer = self.device.new_buffer_init(BufferInitDesc {
                    data: bytemuck::bytes_of(&constants),
                    usage: BufferUsage::UNIFORM,
                    name: "blit_constants",
                });

                let pipelines_cache = match dst_kind {
                    BlitKind::Uint => &self.uint_pipelines,
                    BlitKind::Sint => &self.sint_pipelines,
                    _ => unreachable!(),
                };
                let scalar = match dst_kind {
                    BlitKind::Uint => "u32",
                    BlitKind::Sint => "i32",
                    _ => unreachable!(),
                };

                let pipeline = {
                    let mut pipelines = pipelines_cache.lock();
                    pipelines
                        .entry(dst_format)
                        .or_insert_with(|| make_int_pipeline(&self.device, dst_format, scalar))
                        .clone()
                };

                let mut compute = encoder.compute();
                compute.with_pipeline(&pipeline);
                compute.with_arguments(
                    0,
                    &BlitIntArguments {
                        constants: buffer,
                        src: src.clone(),
                        dst: dst.clone(),
                    },
                );
                compute.dispatch(groups);
            }
            BlitKind::Srgb => {
                // Viewport uses the requested (unclamped) dst rect.
                let viewport_offset = Offset3::new(
                    dst_offset.x().min(dst_image_size.width() as i32) as f32,
                    dst_offset.y().min(dst_image_size.height() as i32) as f32,
                    0.0,
                );

                let viewport_extent = Extent3::new(
                    dst_extent.width().min(dst_image_size.width()) as f32,
                    dst_extent.height().min(dst_image_size.height()) as f32,
                    1.0,
                );

                // Scissor is the intersection of the requested rect with the
                // destination image bounds; if empty, there is nothing to
                // draw, so skip opening the render pass entirely.
                let vx0 = dst_offset.x().max(0);
                let vy0 = dst_offset.y().max(0);
                let vx1 =
                    (dst_offset.x() + dst_extent.width() as i32).min(dst_image_size.width() as i32);
                let vy1 = (dst_offset.y() + dst_extent.height() as i32)
                    .min(dst_image_size.height() as i32);

                if vx1 <= vx0 || vy1 <= vy0 {
                    return;
                }

                let buffer = self.device.new_buffer_init(BufferInitDesc {
                    data: bytemuck::bytes_of(&constants),
                    usage: BufferUsage::UNIFORM,
                    name: "blit_constants",
                });

                let pipeline = {
                    let mut pipelines = self.srgb_pipelines.lock();
                    pipelines
                        .entry(dst_format)
                        .or_insert_with(|| make_srgb_pipeline(&self.device, dst_format))
                        .clone()
                };

                let mut render = encoder.render(RenderPassDesc {
                    name: "blit",
                    color_attachments: &[AttachmentDesc::new(dst)],
                    depth_stencil_attachment: None,
                });

                render.with_pipeline(&pipeline);
                render.with_viewport(viewport_offset, viewport_extent);
                render.with_scissor(
                    Offset2::new(vx0, vy0),
                    Extent2::new((vx1 - vx0) as u32, (vy1 - vy0) as u32),
                );
                render.with_arguments(
                    0,
                    &BlitRenderArguments {
                        constants: buffer,
                        src: src.clone(),
                        sampler: sampler.expect("Srgb blit always has a sampler"),
                    },
                );
                render.draw(0..3, 0..1);
            }
        }
    }
}
