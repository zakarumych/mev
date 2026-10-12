//! GPU-side format-converting copy kernel.
//!
//! `Cvt` converts between `Image`/`Buffer` endpoints described by
//! `PixelFormat`/`VertexFormat`, decoding source components to a canonical
//! value and re-encoding them into the destination's own type (not a raw
//! bit copy), with component swizzle and axis remap
//! (rotation/mirror/splat/diagonal-read) along the way.

use std::fmt;

use hashbrown::HashMap;
use parking_lot::Mutex;

use crate::{
    AddressMode, Arguments, AsBufferSlice, AsImageSlice, AttachmentDesc, Buffer, BufferSlice,
    ColorTargetDesc, CommandEncoder, ComputePipeline, ComputePipelineDesc, Culling, Device,
    DeviceRepr, Extent, Extent1, Extent2, Extent3, Filter, FrontFace, Image, ImageSlice,
    ImageUsage, LibraryDesc, LibraryInput, MipMapMode, Offset, Offset2, Offset3, PixelFormat,
    PrimitiveTopology, RasterDesc, RenderPassDesc, RenderPipeline, RenderPipelineDesc, Sampler,
    SamplerDesc, Swizzle, TexelShape, VertexFormat, ViewDesc,
};

// ============================================================================
// Format description
// ============================================================================

/// Shader-level storage category of a format's components.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum CanonicalKind {
    Float,
    Uint,
    Sint,
}

/// Canonical channel identity, independent of physical memory order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Channel {
    R,
    G,
    B,
    A,
}

const CHANNELS: [Channel; 4] = [Channel::R, Channel::G, Channel::B, Channel::A];

impl Channel {
    fn lower(self) -> &'static str {
        match self {
            Channel::R => "r",
            Channel::G => "g",
            Channel::B => "b",
            Channel::A => "a",
        }
    }

    /// Default value used when a swizzle references a channel the format
    /// does not physically carry (mirrors typical hardware texture
    /// component defaults: RGB missing -> 0, alpha missing -> 1).
    fn default_is_one(self) -> bool {
        matches!(self, Channel::A)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ComponentDesc {
    /// Canonical channel this physical/memory-order component represents.
    channel: Channel,
    /// Bit width: 8, 16 or 32.
    bits: u32,
    /// Whether this is a Unorm/Snorm component (only meaningful within
    /// `CanonicalKind::Float`).
    normalized: bool,
    /// Whether this component is signed.
    signed: bool,
}

struct FormatLayout {
    kind: CanonicalKind,
    srgb: bool,
    byte_size: usize,
    /// Components in physical/memory order. Length 1..=4.
    components: Vec<ComponentDesc>,
}

impl FormatLayout {
    fn channel_position(&self, channel: Channel) -> Option<usize> {
        self.components.iter().position(|c| c.channel == channel)
    }
}

fn comps(bits: u32, normalized: bool, signed: bool, channels: &[Channel]) -> Vec<ComponentDesc> {
    channels
        .iter()
        .map(|&channel| ComponentDesc {
            channel,
            bits,
            normalized,
            signed,
        })
        .collect()
}

/// Describes a `PixelFormat` for `Cvt`'s purposes, or `None` if the format
/// has no per-component decode model (depth/stencil or BC-compressed).
fn describe(format: PixelFormat) -> Option<FormatLayout> {
    use Channel::*;
    use PixelFormat::*;

    let byte_size = format.block_size();
    let srgb = format.is_srgb();

    let (kind, normalized, signed, bits, channels): (CanonicalKind, bool, bool, u32, &[Channel]) =
        match format {
            R8Unorm => (CanonicalKind::Float, true, false, 8, &[R]),
            R8Snorm => (CanonicalKind::Float, true, true, 8, &[R]),
            R8Uint => (CanonicalKind::Uint, false, false, 8, &[R]),
            R8Sint => (CanonicalKind::Sint, false, true, 8, &[R]),
            R8Srgb => (CanonicalKind::Float, true, false, 8, &[R]),
            R16Unorm => (CanonicalKind::Float, true, false, 16, &[R]),
            R16Snorm => (CanonicalKind::Float, true, true, 16, &[R]),
            R16Uint => (CanonicalKind::Uint, false, false, 16, &[R]),
            R16Sint => (CanonicalKind::Sint, false, true, 16, &[R]),
            R16Float => (CanonicalKind::Float, false, false, 16, &[R]),
            R32Unorm => (CanonicalKind::Float, true, false, 32, &[R]),
            R32Snorm => (CanonicalKind::Float, true, true, 32, &[R]),
            R32Uint => (CanonicalKind::Uint, false, false, 32, &[R]),
            R32Sint => (CanonicalKind::Sint, false, true, 32, &[R]),
            R32Float => (CanonicalKind::Float, false, false, 32, &[R]),

            Rg8Unorm => (CanonicalKind::Float, true, false, 8, &[R, G]),
            Rg8Snorm => (CanonicalKind::Float, true, true, 8, &[R, G]),
            Rg8Uint => (CanonicalKind::Uint, false, false, 8, &[R, G]),
            Rg8Sint => (CanonicalKind::Sint, false, true, 8, &[R, G]),
            Rg8Srgb => (CanonicalKind::Float, true, false, 8, &[R, G]),
            Rg16Unorm => (CanonicalKind::Float, true, false, 16, &[R, G]),
            Rg16Snorm => (CanonicalKind::Float, true, true, 16, &[R, G]),
            Rg16Uint => (CanonicalKind::Uint, false, false, 16, &[R, G]),
            Rg16Sint => (CanonicalKind::Sint, false, true, 16, &[R, G]),
            Rg16Float => (CanonicalKind::Float, false, false, 16, &[R, G]),
            Rg32Unorm => (CanonicalKind::Float, true, false, 32, &[R, G]),
            Rg32Snorm => (CanonicalKind::Float, true, true, 32, &[R, G]),
            Rg32Uint => (CanonicalKind::Uint, false, false, 32, &[R, G]),
            Rg32Sint => (CanonicalKind::Sint, false, true, 32, &[R, G]),
            Rg32Float => (CanonicalKind::Float, false, false, 32, &[R, G]),

            Rgb8Unorm => (CanonicalKind::Float, true, false, 8, &[R, G, B]),
            Rgb8Snorm => (CanonicalKind::Float, true, true, 8, &[R, G, B]),
            Rgb8Uint => (CanonicalKind::Uint, false, false, 8, &[R, G, B]),
            Rgb8Sint => (CanonicalKind::Sint, false, true, 8, &[R, G, B]),
            Rgb8Srgb => (CanonicalKind::Float, true, false, 8, &[R, G, B]),
            Rgb16Unorm => (CanonicalKind::Float, true, false, 16, &[R, G, B]),
            Rgb16Snorm => (CanonicalKind::Float, true, true, 16, &[R, G, B]),
            Rgb16Uint => (CanonicalKind::Uint, false, false, 16, &[R, G, B]),
            Rgb16Sint => (CanonicalKind::Sint, false, true, 16, &[R, G, B]),
            Rgb16Float => (CanonicalKind::Float, false, false, 16, &[R, G, B]),
            Rgb32Unorm => (CanonicalKind::Float, true, false, 32, &[R, G, B]),
            Rgb32Snorm => (CanonicalKind::Float, true, true, 32, &[R, G, B]),
            Rgb32Uint => (CanonicalKind::Uint, false, false, 32, &[R, G, B]),
            Rgb32Sint => (CanonicalKind::Sint, false, true, 32, &[R, G, B]),
            Rgb32Float => (CanonicalKind::Float, false, false, 32, &[R, G, B]),

            Rgba8Unorm => (CanonicalKind::Float, true, false, 8, &[R, G, B, A]),
            Rgba8Snorm => (CanonicalKind::Float, true, true, 8, &[R, G, B, A]),
            Rgba8Uint => (CanonicalKind::Uint, false, false, 8, &[R, G, B, A]),
            Rgba8Sint => (CanonicalKind::Sint, false, true, 8, &[R, G, B, A]),
            Rgba8Srgb => (CanonicalKind::Float, true, false, 8, &[R, G, B, A]),
            Rgba16Unorm => (CanonicalKind::Float, true, false, 16, &[R, G, B, A]),
            Rgba16Snorm => (CanonicalKind::Float, true, true, 16, &[R, G, B, A]),
            Rgba16Uint => (CanonicalKind::Uint, false, false, 16, &[R, G, B, A]),
            Rgba16Sint => (CanonicalKind::Sint, false, true, 16, &[R, G, B, A]),
            Rgba16Float => (CanonicalKind::Float, false, false, 16, &[R, G, B, A]),
            Rgba32Unorm => (CanonicalKind::Float, true, false, 32, &[R, G, B, A]),
            Rgba32Snorm => (CanonicalKind::Float, true, true, 32, &[R, G, B, A]),
            Rgba32Uint => (CanonicalKind::Uint, false, false, 32, &[R, G, B, A]),
            Rgba32Sint => (CanonicalKind::Sint, false, true, 32, &[R, G, B, A]),
            Rgba32Float => (CanonicalKind::Float, false, false, 32, &[R, G, B, A]),

            Bgr8Unorm => (CanonicalKind::Float, true, false, 8, &[B, G, R]),
            Bgr8Snorm => (CanonicalKind::Float, true, true, 8, &[B, G, R]),
            Bgr8Uint => (CanonicalKind::Uint, false, false, 8, &[B, G, R]),
            Bgr8Sint => (CanonicalKind::Sint, false, true, 8, &[B, G, R]),
            Bgr8Srgb => (CanonicalKind::Float, true, false, 8, &[B, G, R]),

            Bgra8Unorm => (CanonicalKind::Float, true, false, 8, &[B, G, R, A]),
            Bgra8Snorm => (CanonicalKind::Float, true, true, 8, &[B, G, R, A]),
            Bgra8Uint => (CanonicalKind::Uint, false, false, 8, &[B, G, R, A]),
            Bgra8Sint => (CanonicalKind::Sint, false, true, 8, &[B, G, R, A]),
            Bgra8Srgb => (CanonicalKind::Float, true, false, 8, &[B, G, R, A]),

            D16Unorm | D32Float | S8Uint | D16UnormS8Uint | D24UnormS8Uint | D32FloatS8Uint => {
                return None;
            }

            Bc1RgbUnorm | Bc1RgbSrgb | Bc1RgbaUnorm | Bc1RgbaSrgb | Bc2Unorm | Bc2Srgb
            | Bc3Unorm | Bc3Srgb | Bc4Unorm | Bc4Snorm | Bc5Unorm | Bc5Snorm => return None,
        };

    Some(FormatLayout {
        kind,
        srgb,
        byte_size,
        components: comps(bits, normalized, signed, channels),
    })
}

/// `true` iff both formats describe (i.e. neither is depth/stencil/BC).
/// Cross-`CanonicalKind` pairs ARE allowed - this is deliberately
/// permissive beyond `Blit`'s own same-kind-only `classify`.
pub fn cvt_compatible(src: PixelFormat, dst: PixelFormat) -> bool {
    describe(src).is_some() && describe(dst).is_some()
}

// ============================================================================
// Axis remapping
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Axis {
    X,
    Y,
    Z,
    W,
}

impl fmt::Display for Axis {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Axis::X => f.write_str("X"),
            Axis::Y => f.write_str("Y"),
            Axis::Z => f.write_str("Z"),
            Axis::W => f.write_str("W"),
        }
    }
}

impl fmt::Debug for Axis {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AxisMap {
    Zero,
    Direct(Axis),
    Mirror(Axis),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AxisSwizzle {
    pub x: AxisMap,
    pub y: AxisMap,
    pub z: AxisMap,
    pub w: AxisMap,
}

impl AxisSwizzle {
    pub const IDENTITY: Self = AxisSwizzle {
        x: AxisMap::Direct(Axis::X),
        y: AxisMap::Direct(Axis::Y),
        z: AxisMap::Direct(Axis::Z),
        w: AxisMap::Direct(Axis::W),
    };

    pub fn index(&self, src: usize) -> AxisMap {
        match src {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            3 => self.w,
            _ => panic!("AxisSwizzle::index out of range: {src}"),
        }
    }
}

impl Default for AxisSwizzle {
    fn default() -> Self {
        Self::IDENTITY
    }
}

// ============================================================================
// `Cvt` kernel state and pipeline cache keys
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum EndpointKind {
    Image,
    Buffer,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct EndpointDesc {
    kind: EndpointKind,
    format: PixelFormat,
}

/// Cache key for the render pipelines built by `make_render_pipeline`.
///
/// One pipeline shape per distinct (endpoint kinds/formats, swizzle,
/// axis map, ranks, resample, wrap) combination actually used;
/// `axes`'s slots beyond `src_rank` are ignored.
///
/// `wrap` records whether the generated WGSL wraps exact-index source
/// coordinates through `apply_address_mode` (a text difference, hence
/// a distinct pipeline; the `AddressMode` *values* flow as runtime
/// constants and are not part of the key).
///
/// The render pipeline's destination is always `Image` (never `Buffer` -
/// sRGB `Image` destinations route here specifically
/// because a WGSL storage texture has no sRGB write format),
/// so unlike [`CvtComputeKey`] it never needs a `dst_tight_len` shape discriminant.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct CvtKey {
    src: EndpointDesc,
    dst: EndpointDesc,
    swizzle: Swizzle,
    src_rank: u8,
    dst_rank: u8,
    axes: AxisSwizzle,
    resample: bool,
    wrap: bool,
}

/// Cache key for the compute pipelines built by `make_compute_pipeline_nd`.
///
/// Same fields as [`CvtKey`], plus `dst_tight_len` (`T`, see
/// module docs' tiled-dispatch algorithm): when `dst.kind == Buffer`,
/// the generated WGSL's *shape* differs qualitatively per `T` (which axes
/// fold into the flat tight-chain wrap vs. get their own
/// independently-grouped dispatch dimension), so it must be part of what selects/builds
/// a distinct pipeline.
///
/// Meaningless (always `0`) when `dst.kind != Buffer`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct CvtComputeKey {
    src: EndpointDesc,
    dst: EndpointDesc,
    swizzle: Swizzle,
    src_rank: u8,
    dst_rank: u8,
    axes: AxisSwizzle,
    resample: bool,
    wrap: bool,
    dst_tight_len: u8,
}

/// Cache key for the dedicated single-resource in-place buffer pipeline.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct CvtInPlaceKey {
    src_format: PixelFormat,
    dst_format: PixelFormat,
    swizzle: Swizzle,
}

/// Kernel that performs GPU-side format-converting copies between
/// `Image`/`Buffer` endpoints (value-converting, not a raw bit copy), with
/// component swizzle and axis remap (rotation/mirror/splat/diagonal-read).
///
/// Mirrors `Blit`'s architecture: pipelines are lazily built from
/// `format!`-generated WGSL and cached by a small hashable key, one
/// pipeline shape per distinct (formats, swizzle, axis map, resample)
/// combination actually used.
pub struct Cvt {
    device: Device,
    render_pipelines: Mutex<HashMap<CvtKey, RenderPipeline>>,
    compute_pipelines: Mutex<HashMap<CvtComputeKey, ComputePipeline>>,
    in_place_pipelines: Mutex<HashMap<CvtInPlaceKey, ComputePipeline>>,
    samplers: Mutex<HashMap<SamplerDesc, Sampler>>,
}

// ============================================================================
// Small numeric helpers shared by codegen
// ============================================================================

/// Divisor used to decode/encode a normalized (Unorm/Snorm) component of
/// `bits` width: `2^bits - 1` unsigned, `2^(bits-1) - 1` signed.
fn normalized_max(bits: u32, signed: bool) -> f64 {
    if signed {
        ((1u64 << (bits - 1)) - 1) as f64
    } else {
        ((1u64 << bits) - 1) as f64
    }
}

/// Representable integer range of a raw `Uint`/`Sint` component of `bits`
/// width.
fn int_range(bits: u32, signed: bool) -> (i64, i64) {
    if signed {
        let max = (1i64 << (bits - 1)) - 1;
        let min = -(1i64 << (bits - 1));
        (min, max)
    } else {
        (0, (1i64 << bits) - 1)
    }
}

fn wgsl_scalar(kind: CanonicalKind) -> &'static str {
    match kind {
        CanonicalKind::Float => "f32",
        CanonicalKind::Uint => "u32",
        CanonicalKind::Sint => "i32",
    }
}

/// Formats an `f64` as a WGSL `f32` literal (always with a decimal point,
/// so it can't be mistaken for an abstract-int literal).
fn fmt_f32(v: f64) -> String {
    let s = format!("{v}");
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{s}.0")
    }
}

/// Formats an `i64` as a WGSL integer literal of the given signedness.
fn int_lit(v: i64, unsigned: bool) -> String {
    if unsigned {
        format!("{v}u")
    } else {
        format!("{v}")
    }
}

impl FormatLayout {
    /// Bit width shared by all components of this format.
    fn bits(&self) -> u32 {
        self.components[0].bits
    }

    /// Normalized flag shared by all components of this format.
    fn normalized(&self) -> bool {
        self.components[0].normalized
    }

    /// Signedness shared by all components of this format.
    fn signed(&self) -> bool {
        self.components[0].signed
    }
}

// ============================================================================
// WGSL codegen: shared helper function bodies
// ============================================================================

/// `apply_address_mode`, reused verbatim from `Blit`'s int pipeline: wraps
/// or clamps an integer coordinate per `AddressMode` (0 = Repeat, 1 =
/// MirrorRepeat, 2 = ClampToEdge - declaration-order discriminant).
const ADDRESS_MODE_HELPER: &str = r#"
fn apply_address_mode(x: i32, size: i32, mode: u32) -> i32 {
    if (mode == 2u) {
        return clamp(x, 0, size - 1);
    } else if (mode == 1u) {
        let period = 2 * size;
        var m = x % period;
        if (m < 0) { m = m + period; }
        if (m >= size) { m = period - 1 - m; }
        return m;
    } else {
        var m = x % size;
        if (m < 0) { m = m + size; }
        return m;
    }
}
"#;

/// Half<->float conversion, needed whenever a `Buffer` endpoint has a true
/// (non-normalized) 16-bit float component (`VertexFormat::Float16` /
/// `PixelFormat::R16Float`-family used as a buffer format).
///
/// Image endpoints never need this - hardware handles f16 texture formats natively.
const HALF_FLOAT_HELPERS: &str = r#"
fn mev_half_to_float(h: u32) -> f32 {
    return unpack2x16float(h).x;
}

fn mev_float_to_half(value: f32) -> u32 {
    return pack2x16float(vec2<f32>(value, 0.0)) & 0xFFFFu;
}
"#;

// ============================================================================
// WGSL codegen: decode / swizzle / convert / encode
// ============================================================================

/// Generates `let {var}: {scalar} = ...;` decoding canonical channel
/// `channel` of `layout` from a buffer element whose byte-offset expression
/// is `elem_byte`.
///
/// The word index and in-word bit offset for this component are computed
/// at *runtime* from `elem_byte` (not assumed to be a compile-time
/// constant) - correct for any `elem_byte` value, as long as the caller
/// keeps every component's absolute byte offset a multiple of that
/// component's own byte width (1/2/4), which every entry point asserts up
/// front (a component can never straddle a 4-byte storage word otherwise).
///
/// `buf` is the storage array's identifier (`array<u32>`; every read in this
/// kernel is a plain, non-atomic load - no binding ever needs `atomic<u32>`
/// since destination writes are structurally disjoint across threads, see
/// module docs).
fn gen_buffer_decode_channel(
    var: &str,
    buf: &str,
    elem_byte: &str,
    layout: &FormatLayout,
    channel: Channel,
) -> String {
    let scalar = wgsl_scalar(layout.kind);

    let Some(i) = layout.channel_position(channel) else {
        let lit = match layout.kind {
            CanonicalKind::Float => {
                if channel.default_is_one() {
                    "1.0".to_string()
                } else {
                    "0.0".to_string()
                }
            }
            CanonicalKind::Uint => {
                if channel.default_is_one() {
                    "1u".to_string()
                } else {
                    "0u".to_string()
                }
            }
            CanonicalKind::Sint => {
                if channel.default_is_one() {
                    "1".to_string()
                } else {
                    "0".to_string()
                }
            }
        };
        return format!("    let {var}: {scalar} = {lit};\n");
    };

    let comp = &layout.components[i];
    let local_off = i * (comp.bits as usize / 8);
    let comp_byte_expr = if local_off == 0 {
        elem_byte.to_string()
    } else {
        format!("{elem_byte} + {local_off}u")
    };

    let mut prelude = format!("    let {var}_byte: u32 = {comp_byte_expr};\n");
    prelude += &format!("    let {var}_word: u32 = {var}_byte / 4u;\n");
    let bit_off = format!("{var}_bit");
    if comp.bits != 32 {
        prelude += &format!("    let {bit_off}: u32 = ({var}_byte % 4u) * 8u;\n");
    }

    let word_expr = format!("{buf}[{var}_word]");

    let body = if layout.kind == CanonicalKind::Float && !comp.normalized && comp.bits == 32 {
        format!("bitcast<f32>({word_expr})")
    } else if layout.kind == CanonicalKind::Float && !comp.normalized && comp.bits == 16 {
        format!("mev_half_to_float(extractBits({word_expr}, {bit_off}, 16u))")
    } else if layout.kind == CanonicalKind::Float {
        // Normalized (Unorm/Snorm), any width.
        let max = fmt_f32(normalized_max(comp.bits, comp.signed));
        let raw = if comp.signed {
            if comp.bits == 32 {
                format!("bitcast<i32>({word_expr})")
            } else {
                format!(
                    "extractBits(bitcast<i32>({word_expr}), {bit_off}, {}u)",
                    comp.bits
                )
            }
        } else if comp.bits == 32 {
            word_expr
        } else {
            format!("extractBits({word_expr}, {bit_off}, {}u)", comp.bits)
        };
        if comp.signed {
            format!("max(f32({raw}) / {max}, -1.0)")
        } else {
            format!("(f32({raw}) / {max})")
        }
    } else if layout.kind == CanonicalKind::Uint {
        if comp.bits == 32 {
            word_expr
        } else {
            format!("extractBits({word_expr}, {bit_off}, {}u)", comp.bits)
        }
    } else {
        // Sint
        if comp.bits == 32 {
            format!("bitcast<i32>({word_expr})")
        } else {
            format!(
                "extractBits(bitcast<i32>({word_expr}), {bit_off}, {}u)",
                comp.bits
            )
        }
    };

    format!("{prelude}    let {var}: {scalar} = {body};\n")
}

/// Generates the four `let sw_{r,g,b,a}: {scalar} = ...;` statements
/// applying `swizzle` to the decoded `src_{r,g,b,a}` canonical values.
fn gen_swizzle(swizzle: Swizzle, src_kind: CanonicalKind) -> String {
    use crate::ComponentMap;

    let scalar = wgsl_scalar(src_kind);
    let (zero, one) = match src_kind {
        CanonicalKind::Float => ("0.0", "1.0"),
        CanonicalKind::Uint => ("0u", "1u"),
        CanonicalKind::Sint => ("0", "1"),
    };

    let pick = |cs: ComponentMap, identity: &'static str| -> &'static str {
        match cs {
            ComponentMap::Zero => zero,
            ComponentMap::One => one,
            ComponentMap::R => "src_r",
            ComponentMap::G => "src_g",
            ComponentMap::B => "src_b",
            ComponentMap::A => "src_a",
        }
    };

    let r = pick(swizzle.r, "src_r");
    let g = pick(swizzle.g, "src_g");
    let b = pick(swizzle.b, "src_b");
    let a = pick(swizzle.a, "src_a");

    format!(
        "    let sw_r: {scalar} = {r};\n    let sw_g: {scalar} = {g};\n    let sw_b: {scalar} = {b};\n    let sw_a: {scalar} = {a};\n"
    )
}

/// Generates `let {var}: {ty} = ...;` converting swizzled canonical value
/// `sw_var` (of `src`'s canonical type) into `dst`'s target representation,
/// per the cross-kind conversion table (module docs / plan §3).
///
/// Returns the WGSL scalar type name of the produced value.
///
/// When `dst_is_image` is set and `dst` is `Float`-kind, the final
/// `* dst_max` requantization step is skipped:
/// `textureStore` into a Unorm/Snorm/Float storage format
/// auto-quantizes from the canonical `[0,1]`/`[-1,1]`/true-float value,
/// so doing it manually here would double-scale.
fn gen_convert_channel(
    var: &str,
    sw_var: &str,
    src: &FormatLayout,
    dst: &FormatLayout,
    dst_is_image: bool,
) -> (String, &'static str) {
    let dst_bits = dst.bits();
    let dst_signed = dst.signed();
    let dst_normalized = dst.normalized();

    let (expr, ty): (String, &'static str) = match (src.kind, dst.kind) {
        (CanonicalKind::Float, CanonicalKind::Float) => {
            if dst_normalized {
                let (lo, hi) = if dst_signed {
                    ("-1.0", "1.0")
                } else {
                    ("0.0", "1.0")
                };
                if dst_is_image {
                    (format!("clamp({sw_var}, {lo}, {hi})"), "f32")
                } else {
                    let max = fmt_f32(normalized_max(dst_bits, dst_signed));
                    (format!("round(clamp({sw_var}, {lo}, {hi}) * {max})"), "f32")
                }
            } else {
                (sw_var.to_string(), "f32")
            }
        }

        (CanonicalKind::Float, CanonicalKind::Uint)
        | (CanonicalKind::Float, CanonicalKind::Sint) => {
            let unsigned = dst.kind == CanonicalKind::Uint;
            let cast_ty = if unsigned { "u32" } else { "i32" };
            let (dmin, dmax) = int_range(dst_bits, dst_signed);
            let dmin_lit = int_lit(dmin, false);
            let dmax_lit = int_lit(dmax, false);

            let magnitude = if src.normalized() {
                let (lo, hi) = if src.signed() {
                    ("-1.0", "1.0")
                } else {
                    ("0.0", "1.0")
                };
                let scale = fmt_f32(dmax as f64);
                format!("round(clamp({sw_var}, {lo}, {hi}) * {scale})")
            } else {
                format!("round({sw_var})")
            };

            let clamped_i32 = format!("clamp(i32({magnitude}), {dmin_lit}, {dmax_lit})");
            if unsigned {
                (format!("u32({clamped_i32})"), cast_ty)
            } else {
                (clamped_i32, cast_ty)
            }
        }

        (CanonicalKind::Uint, CanonicalKind::Float)
        | (CanonicalKind::Sint, CanonicalKind::Float) => {
            let src_max = fmt_f32(int_range(src.bits(), src.signed()).1 as f64);
            let canonical = format!("(f32({sw_var}) / {src_max})");
            if dst_normalized {
                let (lo, hi) = if dst_signed {
                    ("-1.0", "1.0")
                } else {
                    ("0.0", "1.0")
                };
                let clamped = format!("clamp({canonical}, {lo}, {hi})");
                if dst_is_image {
                    (clamped, "f32")
                } else {
                    let dmax = fmt_f32(normalized_max(dst_bits, dst_signed));
                    (format!("round(({clamped}) * {dmax})"), "f32")
                }
            } else {
                (format!("f32({sw_var})"), "f32")
            }
        }

        (CanonicalKind::Uint, CanonicalKind::Uint) => {
            let dmax = int_range(dst_bits, false).1;
            (format!("min({sw_var}, {})", int_lit(dmax, true)), "u32")
        }
        (CanonicalKind::Sint, CanonicalKind::Sint) => {
            let (dmin, dmax) = int_range(dst_bits, true);
            (
                format!(
                    "clamp({sw_var}, {}, {})",
                    int_lit(dmin, false),
                    int_lit(dmax, false)
                ),
                "i32",
            )
        }
        (CanonicalKind::Uint, CanonicalKind::Sint) => {
            let dmax = int_range(dst_bits, true).1;
            (
                format!("i32(min({sw_var}, {}))", int_lit(dmax, true)),
                "i32",
            )
        }
        (CanonicalKind::Sint, CanonicalKind::Uint) => {
            let dmax = int_range(dst_bits, false).1;
            (
                format!("u32(clamp({sw_var}, 0, {}))", int_lit(dmax, false)),
                "u32",
            )
        }
    };

    (format!("    let {var}: {ty} = {expr};\n"), ty)
}

/// Generates the store statements writing `out_{r,g,b,a}`
/// (already in `dst`'s target representation, per [`gen_convert_channel`])
/// into a buffer named `target` (always declared plain `array<u32>`),
/// at the element whose byte-offset expression is `elem_byte`.
///
/// `bits == 32` components are a single plain store
/// (a 32-bit component already fully owns its word).
///
/// `bits < 32` components go through a plain (non-atomic)
/// masked read-modify-write - safe because the caller's dispatch structure
/// guarantees no *other* thread ever touches this word
/// (see module docs' tiled-dispatch algorithm) and, within one thread,
/// statements execute sequentially, so a load-modify-store
/// from the same thread cannot race with itself.
///
/// Per-component locals are suffixed by the component's index `i`
/// so multiple sub-word components in the same call (e.g. `Rg8Uint`)
/// don't collide in the same WGSL scope.
fn gen_buffer_encode(target: &str, elem_byte: &str, dst: &FormatLayout) -> String {
    let mut s = String::new();

    for (i, comp) in dst.components.iter().enumerate() {
        let out_var = format!("out_{}", comp.channel.lower());
        let byte_off = i * (comp.bits as usize / 8);
        let byte_expr = if byte_off == 0 {
            elem_byte.to_string()
        } else {
            format!("{elem_byte} + {byte_off}u")
        };

        let bits_expr = if dst.kind == CanonicalKind::Float && !comp.normalized && comp.bits == 32 {
            format!("bitcast<u32>({out_var})")
        } else if dst.kind == CanonicalKind::Float && !comp.normalized && comp.bits == 16 {
            format!("mev_float_to_half({out_var})")
        } else if dst.kind == CanonicalKind::Float {
            if comp.signed {
                format!("bitcast<u32>(i32({out_var}))")
            } else {
                format!("u32({out_var})")
            }
        } else if dst.kind == CanonicalKind::Sint {
            format!("bitcast<u32>({out_var})")
        } else {
            out_var.clone()
        };

        if comp.bits == 32 {
            s += &format!("    {target}[{byte_expr} / 4u] = {bits_expr};\n");
        } else {
            let mask_hex = if comp.bits == 16 { "0xFFFFu" } else { "0xFFu" };
            s += &format!("    let word_index_{i}: u32 = ({byte_expr}) / 4u;\n");
            s += &format!("    let bit_offset_{i}: u32 = (({byte_expr}) % 4u) * 8u;\n");
            s += &format!("    let mask_{i}: u32 = {mask_hex} << bit_offset_{i};\n");
            s += &format!(
                "    {target}[word_index_{i}] = ({target}[word_index_{i}] & ~mask_{i}) | (({bits_expr} << bit_offset_{i}) & mask_{i});\n"
            );
        }
    }

    s
}

fn needs_half_float(layout: &FormatLayout) -> bool {
    layout.kind == CanonicalKind::Float && !layout.normalized() && layout.bits() == 16
}

/// Byte size of one of `layout`'s (uniformly-sized) components: 1, 2 or 4.
///
/// A component can only ever straddle a 4-byte storage-buffer word
/// if its absolute byte offset isn't a multiple of its own width
/// - a 1-byte component never straddles regardless of offset;
/// - 2/4-byte components need their offset 2-/4-byte aligned respectively.
///
/// Requiring every byte-offset/stride to be a multiple of `component_byte_size`
/// (rather than a flat 4 bytes) is therefore the minimal correct precondition,
/// and keeps common tightly-packed narrow-format buffers
/// (e.g. a plain `R8` row) usable without artificial padding.
fn component_byte_size(layout: &FormatLayout) -> usize {
    layout.bits() as usize / 8
}

/// Resolves an `ImageSlice<N>` into a single-layer, single-level `Image` view
/// (bound at the slice's `level`/`layers`) plus the slice's offset/extent,
/// for binding into a Cvt pipeline.
///
/// Cvt's shader bindings are always plain (non-array) 2D textures
/// reading level 0 of whatever's bound,
/// so the slice must describe exactly one layer - asserted here.
fn resolve_image_slice<const N: usize>(
    device: &Device,
    slice: &ImageSlice<'_, N>,
) -> (Image, Offset<N, i32>, Extent<N, u32>) {
    assert_eq!(
        slice.layers().end - slice.layers().start,
        1,
        "Cvt: image slice must describe exactly one layer (no array-texture support)"
    );

    let view = slice.image().view(
        device,
        ViewDesc::new(slice.image().format())
            .layers(slice.layers())
            .levels((slice.level()..slice.level() + 1).into()),
    );

    (view, slice.offset(), slice.extent())
}

// ============================================================================
// WGSL codegen: shared per-axis source addressing + decode/swizzle/convert
// ============================================================================

/// Appends, for each axis index `s` in `axis_indices` whose `axes.index(s)`
/// is a resampled `MappedAxis::Dst` (i.e. has a `src_coord_f{s}` already
/// generated), the `floor` + `apply_address_mode` lines that convert it to
/// a final integer `src_coord{s}`, wrapped against `size_expr(s)`.
/// `MappedAxis::Fixed` axes already have their final `src_coord{s}` from the
/// caller's initial per-axis loop and are left untouched (no scaling
/// participates).
fn gen_resample_int_wrap(
    body: &mut String,
    axes: AxisSwizzle,
    axis_indices: &[usize],
    size_expr: impl Fn(usize) -> String,
) {
    for &s in axis_indices {
        if let AxisMap::Direct(_) | AxisMap::Mirror(_) = axes.index(s) {
            *body += &format!("    let src_texel{s}: i32 = i32(floor(src_coord_f{s}));\n");
            *body += &format!(
                "    let src_coord{s}: i32 = apply_address_mode(src_texel{s}, {}, constants.address_mode[{s}]);\n",
                size_expr(s)
            );
        }
    }
}

/// Generates the shared body (per-source-axis coordinate, decode, swizzle,
/// convert) used by every compute/render pipeline variant, for a source
/// endpoint of rank `src_rank` (<= 4) described by `axes`
/// (source-axis indexed; a `MappedAxis::Dst` axis references a destination
/// axis whose unraveled coordinate `dst_c{axis}` the caller has already
/// computed before calling this).
///
/// `dst_is_image` controls whether `gen_convert_channel` skips
/// the final normalized-dst requantization
/// (deferred to hardware on `textureStore`).
///
/// When `resample` is `false`, every source axis is exact-index:
/// a plain integer `src_coord{s}`, either `constants.src_offset[s]`
/// (`MappedAxis::Fixed`) or `constants.src_offset[s] + dst_c{axis}`
/// (mirrored as `... + (dst_extent[axis] - 1 - dst_c{axis})`)
/// for `MappedAxis::Dst` - except when `src.kind == EndpointKind::Buffer`
/// (no offset concept; fold any desired starting coordinate
/// into `src_byte_offset` instead), where the `constants.src_offset[s]` term
/// is elided entirely (literal `0` / `i32(dst_c{axis})` / the mirrored
/// equivalent), since the `Buffer`-src constants structs carry
/// no `src_offset` field at all.
///
/// When `wrap` is additionally set, each exact-index `MappedAxis::Dst`
/// coordinate is passed through `apply_address_mode` before use
/// against the image's real physical extent for an `Image` src,
/// or the logical `constants.src_extent` region for a `Buffer` src.
///
/// `wrap` has no effect on the resample path, which always wraps (see below).
///
/// When `resample` is `true`, a `MappedAxis::Dst` source axis instead
/// computes a ratio `t = (dst_c{axis} + 0.5) / dst_extent[axis]`
/// (mirrored as `t = 1.0 - t`), then a continuous source coordinate
/// `src_coord_f{s} = src_offset[s] + t * src_extent[s]`
/// (for a `Buffer` src, elided to just `t * src_extent[s]`,
/// same reasoning as the non-resample path above).
///
/// What happens to that continuous coordinate then depends on the source endpoint:
///
/// - `Image`, `CanonicalKind::Float`: normalized into a `uv`
///   and hardware-sampled via `textureSampleLevel` through a real `Sampler`
///   (built by the caller from the requested filter/address mode),
///   with address-mode wrapping handled by the sampler itself
///   against the image's real physical extent.
/// - `Image`, `CanonicalKind::Uint`/`Sint`: no hardware sampler exists
///   for integer texture formats, so it's `floor`ed to an integer texel
///   and wrapped manually via `apply_address_mode`
///   against the image's real physical extent
///   (`constants.src_image_size`), then read with `textureLoad`
///   - always nearest, never filtered.
/// - `Buffer` (regardless of `CanonicalKind`): buffers have
///   no hardware sampler and no meaningful notion of bilinear interpolation
///   between struct elements, so it's always `floor`ed and wrapped
///   via `apply_address_mode` exactly like the integer-image case,
///   but against the logical `constants.src_extent` region itself
///   (there is no larger "physical buffer extent" concept in this API,
///   unlike the image case).
///
/// `MappedAxis::Fixed` axes are unaffected by resampling (no scaling
/// participates, same exact-index formula as above).
#[allow(clippy::too_many_arguments)]
fn gen_endpoint_decode_convert(
    src: EndpointDesc,
    dst_layout: &FormatLayout,
    src_layout: &FormatLayout,
    swizzle: Swizzle,
    axes: AxisSwizzle,
    src_rank: usize,
    dst_is_image: bool,
    resample: bool,
    wrap: bool,
) -> String {
    let mut body = String::new();

    let samples_via_sampler =
        resample && src.kind == EndpointKind::Image && src_layout.kind == CanonicalKind::Float;

    let src_is_buffer = src.kind == EndpointKind::Buffer;

    // Wrap boundary for exact-index axis `s`, or `None` to use the plain
    // coordinate: `Image` sources wrap only axes 0/1 (the only axes
    // `textureLoad` ever addresses; `src_image_size` has no further
    // entries), `Buffer` sources wrap every axis against `src_extent`.
    let exact_wrap_size = |s: usize| -> Option<String> {
        if !wrap || resample {
            return None;
        }
        match src.kind {
            EndpointKind::Image if s < 2 => Some(format!("i32(constants.src_image_size[{s}])")),
            EndpointKind::Image => None,
            EndpointKind::Buffer => Some(format!("i32(constants.src_extent[{s}])")),
        }
    };

    for s in 0..src_rank {
        let line = match axes.index(s) {
            AxisMap::Zero => {
                if src_is_buffer {
                    format!("    let src_coord{s}: i32 = 0;\n")
                } else {
                    format!("    let src_coord{s}: i32 = constants.src_offset[{s}];\n")
                }
            }
            AxisMap::Direct(axis) if resample => {
                let axis = axis as usize;

                let mut line = format!(
                    "    var t{s}: f32 = (f32(dst_c{axis}) + 0.5) / f32(constants.dst_extent[{axis}]);\n"
                );
                line += &if src_is_buffer {
                    format!("    let src_coord_f{s}: f32 = t{s} * constants.src_extent[{s}];\n")
                } else {
                    format!(
                        "    let src_coord_f{s}: f32 = f32(constants.src_offset[{s}]) + t{s} * constants.src_extent[{s}];\n"
                    )
                };
                line
            }
            AxisMap::Mirror(axis) if resample => {
                let axis = axis as usize;

                let mut line = format!(
                    "    var t{s}: f32 = (f32(dst_c{axis}) + 0.5) / f32(constants.dst_extent[{axis}]);\n"
                );

                line += &format!("    t{s} = 1.0 - t{s};\n");

                line += &if src_is_buffer {
                    format!("    let src_coord_f{s}: f32 = t{s} * constants.src_extent[{s}];\n")
                } else {
                    format!(
                        "    let src_coord_f{s}: f32 = f32(constants.src_offset[{s}]) + t{s} * constants.src_extent[{s}];\n"
                    )
                };
                line
            }
            AxisMap::Direct(axis) => {
                let axis = axis as usize;
                let expr = if src_is_buffer {
                    format!("i32(dst_c{axis})")
                } else {
                    format!("constants.src_offset[{s}] + i32(dst_c{axis})")
                };
                match exact_wrap_size(s) {
                    Some(size) => format!(
                        "    let src_coord{s}: i32 = apply_address_mode({expr}, {size}, constants.address_mode[{s}]);\n"
                    ),
                    None => format!("    let src_coord{s}: i32 = {expr};\n"),
                }
            }
            AxisMap::Mirror(axis) => {
                let axis = axis as usize;
                let expr = if src_is_buffer {
                    format!("i32(constants.dst_extent[{axis}]) - 1 - i32(dst_c{axis})")
                } else {
                    format!(
                        "constants.src_offset[{s}] + i32(constants.dst_extent[{axis}]) - 1 - i32(dst_c{axis})"
                    )
                };
                match exact_wrap_size(s) {
                    Some(size) => format!(
                        "    let src_coord{s}: i32 = apply_address_mode({expr}, {size}, constants.address_mode[{s}]);\n"
                    ),
                    None => format!("    let src_coord{s}: i32 = {expr};\n"),
                }
            }
        };
        body += &line;
    }

    match src.kind {
        EndpointKind::Image => {
            if samples_via_sampler {
                let uv_expr = |s: usize| -> String {
                    match axes.index(s) {
                        AxisMap::Zero => format!(
                            "(f32(src_coord{s}) + 0.5) / f32(constants.src_image_size[{s}])"
                        ),
                        AxisMap::Direct(_) | AxisMap::Mirror(_) => {
                            format!("src_coord_f{s} / f32(constants.src_image_size[{s}])")
                        }
                    }
                };
                body += &format!("    let uv = vec2<f32>({}, {});\n", uv_expr(0), uv_expr(1));
                body += "    let loaded = textureSampleLevel(src, src_sampler, uv, 0.0);\n";
            } else {
                if resample {
                    gen_resample_int_wrap(&mut body, axes, &[0, 1], |s| {
                        format!("i32(constants.src_image_size[{s}])")
                    });
                }
                body +=
                    "    let loaded = textureLoad(src, vec2<i32>(src_coord0, src_coord1), 0);\n";
            }
            body += "    let src_r = loaded.r;\n    let src_g = loaded.g;\n    let src_b = loaded.b;\n    let src_a = loaded.a;\n";
        }
        EndpointKind::Buffer => {
            if resample {
                let axis_indices: Vec<usize> = (0..src_rank).collect();
                gen_resample_int_wrap(&mut body, axes, &axis_indices, |s| {
                    format!("i32(constants.src_extent[{s}])")
                });
            }
            let mut src_byte_expr = "constants.src_byte_offset".to_string();
            for s in 0..src_rank {
                src_byte_expr += &format!(" + u32(src_coord{s}) * constants.src_stride[{s}]");
            }
            body += &format!("    let src_elem_byte: u32 = {src_byte_expr};\n");
            for &ch in &CHANNELS {
                body += &gen_buffer_decode_channel(
                    &format!("src_{}", ch.lower()),
                    "src",
                    "src_elem_byte",
                    src_layout,
                    ch,
                );
            }
        }
    }

    body += &gen_swizzle(swizzle, src_layout.kind);

    for &ch in &CHANNELS {
        let var = format!("out_{}", ch.lower());
        let sw_var = format!("sw_{}", ch.lower());
        let (line, _ty) = gen_convert_channel(&var, &sw_var, src_layout, dst_layout, dst_is_image);
        body += &line;
    }

    body
}

// ============================================================================
// WGSL codegen: storage-format keywords
// ============================================================================

/// Maps a describable, non-sRGB `PixelFormat` to the WGSL storage-format
/// keyword for a `texture_storage_2d<FORMAT, write>` binding,
/// for `Float`-kind (Unorm/Snorm/Float) destinations.
///
/// Mirrors `Blit`'s own table exactly: WGSL's storage-format keyword set
/// has no entry for 3-channel (`Rgb*`) or 32-bit-normalized
/// (`R32Unorm`/`R32Snorm`/...) formats, so `Image` destinations
/// of those formats are not supported
/// by the compute path (panics)
/// - the same inherent limitation `Blit` has.
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
        _ => panic!(
            "Cvt: pixel format {:?} has no WGSL storage-format keyword and cannot be used as an Image destination",
            format
        ),
    }
}

/// Same as [`wgsl_storage_format`], for `Uint`/`Sint`-kind destinations.
/// No `Bgra8Uint`/`Bgra8Sint` entry: WGSL has no such storage-format
/// keyword.
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
        _ => panic!(
            "Cvt: pixel format {:?} has no WGSL storage-format keyword and cannot be used as an Image destination",
            format
        ),
    }
}

fn dst_image_storage_format(dst_layout: &FormatLayout, format: PixelFormat) -> &'static str {
    match dst_layout.kind {
        CanonicalKind::Float => wgsl_storage_format(format),
        CanonicalKind::Uint | CanonicalKind::Sint => wgsl_int_storage_format(format),
    }
}

/// Whether `Cvt` must route the write to `dst` through the render pipeline
/// instead of the compute pipeline, given the destination Image's actual
/// allowed usage - not just its format.
///
/// sRGB destinations always require the render path: WGSL's storage-texture
/// format keyword set has no sRGB entries, so a compute pipeline cannot
/// write to it regardless of `usage`.
///
/// Otherwise the choice follows `dst`'s `ImageUsage`: `STORAGE` (even
/// alongside `TARGET`) prefers the compute/dispatch path; lacking
/// `STORAGE`, `TARGET` falls back to the render path; lacking both, `dst`
/// cannot be written by `Cvt` at all.
fn cvt_use_render_pipeline(
    dst_layout: &FormatLayout,
    dst_format: PixelFormat,
    dst_usage: ImageUsage,
) -> bool {
    if dst_layout.srgb {
        assert!(
            dst_usage.contains(ImageUsage::TARGET),
            "Cvt: dst format {:?} is sRGB, which WGSL storage-texture formats cannot express; dst Image needs ImageUsage::TARGET (usage: {:?})",
            dst_format,
            dst_usage
        );
        return true;
    }

    if dst_usage.contains(ImageUsage::STORAGE) {
        false
    } else if dst_usage.contains(ImageUsage::TARGET) {
        true
    } else {
        panic!(
            "Cvt: dst Image usage {:?} allows neither STORAGE (compute dispatch) nor TARGET (render); dst format {:?} cannot be written by Cvt",
            dst_usage,
            dst_format
        );
    }
}

// ============================================================================
// ND compute pipeline (`Buffer<->Buffer`, `Image<->Image`, `Image<->Buffer`,
// `Buffer<->Image`; any rank <= 4)
// ============================================================================

// The constants are passed as push constants (`with_constants`),
// one specialized struct per (src.kind, dst.kind) endpoint combination,
// each carrying only the fields that combination actually needs
// (an `Image` endpoint contributes an `_offset`/`_image_size` pair,
// a `Buffer` endpoint contributes a `_byte_offset`/`_stride` pair
// - `dst_extent`, `total_dst_elements`, `src_extent`, `address_mode`
// are common to all four).
//
// WGSL's `push_constant` (`immediate`) address space
// has no 16-byte-array-stride rule the way `uniform` does,
// so every field here packs tightly,
// matching `AutoDeviceRepr`'s no-implicit-padding requirement.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod, mev::AutoDeviceRepr)]
struct CvtConstantsBufferToBuffer {
    dst_extent: [u32; 4],
    dst_byte_offset: u32,
    dst_stride: [u32; 4],
    /// No offset concept for a `Buffer` src - fold any desired starting
    /// coordinate into `src_byte_offset` instead.
    src_byte_offset: u32,
    src_stride: [u32; 4],
    total_dst_elements: u32,
    /// Resample-only: the logical source region extent, used both to scale
    /// the destination-driven ratio `t` into a source coordinate, and as
    /// the wrap boundary for `apply_address_mode`.
    src_extent: [f32; 4],
    /// Resample-only: per-axis `AddressMode` (declaration-order
    /// discriminant: Repeat=0, MirrorRepeat=1, ClampToEdge=2).
    address_mode: [u32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod, mev::AutoDeviceRepr)]
struct CvtConstantsBufferToImage {
    dst_extent: [u32; 4],
    /// Pixel offset of the destination region within the physical image.
    dst_offset: [i32; 4],
    /// The destination image's physical extent, used for the write-bounds
    /// check.
    dst_image_size: [u32; 2],
    /// No offset concept for a `Buffer` src - fold any desired starting
    /// coordinate into `src_byte_offset` instead.
    src_byte_offset: u32,
    src_stride: [u32; 4],
    total_dst_elements: u32,
    /// Resample-only: the logical source region extent, used both to scale
    /// the destination-driven ratio `t` into a source coordinate, and as
    /// the wrap boundary for `apply_address_mode`.
    src_extent: [f32; 4],
    /// Resample-only: per-axis `AddressMode` (declaration-order
    /// discriminant: Repeat=0, MirrorRepeat=1, ClampToEdge=2).
    address_mode: [u32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod, mev::AutoDeviceRepr)]
struct CvtConstantsImageToBuffer {
    dst_extent: [u32; 4],
    dst_byte_offset: u32,
    dst_stride: [u32; 4],
    /// The source `Image` region's pixel offset.
    src_offset: [i32; 4],
    /// The source `Image`'s physical extent, used by the resample
    /// path's `apply_address_mode` wrapping.
    ///
    /// Zero/unused on the exact-index path.
    src_image_size: [u32; 2],
    total_dst_elements: u32,
    /// Resample-only: the logical source region extent, used both to scale
    /// the destination-driven ratio `t` into a source coordinate, and (for
    /// `Uint`/`Sint` src) as the wrap boundary for `apply_address_mode`.
    src_extent: [f32; 4],
    /// Resample-only: per-axis `AddressMode` (declaration-order
    /// discriminant: Repeat=0, MirrorRepeat=1, ClampToEdge=2).
    address_mode: [u32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod, mev::AutoDeviceRepr)]
struct CvtConstantsImageToImage {
    dst_extent: [u32; 4],
    /// Pixel offset of the destination region within the physical image.
    dst_offset: [i32; 4],
    /// The destination image's physical extent, used for the write-bounds
    /// check.
    dst_image_size: [u32; 2],
    /// The source `Image` region's pixel offset.
    src_offset: [i32; 4],
    /// The source `Image`'s physical extent, used by the resample
    /// path's `apply_address_mode` wrapping.
    ///
    /// Zero/unused on the exact-index path.
    src_image_size: [u32; 2],
    total_dst_elements: u32,
    /// Resample-only: the logical source region extent, used both to scale
    /// the destination-driven ratio `t` into a source coordinate, and (for
    /// `Uint`/`Sint` src) as the wrap boundary for `apply_address_mode`.
    src_extent: [f32; 4],
    /// Resample-only: per-axis `AddressMode` (declaration-order
    /// discriminant: Repeat=0, MirrorRepeat=1, ClampToEdge=2).
    address_mode: [u32; 4],
}

#[derive(Arguments)]
struct CvtNdArguments {
    #[mev(storage)]
    #[mev(compute)]
    src: Buffer,

    #[mev(storage)]
    #[mev(compute)]
    dst: Buffer,
}

#[derive(Arguments)]
struct CvtImageToImageArguments {
    #[mev(sampled)]
    #[mev(compute)]
    src: Image,

    #[mev(storage)]
    #[mev(compute)]
    dst: Image,
}

#[derive(Arguments)]
struct CvtImageToBufferArguments {
    #[mev(sampled)]
    #[mev(compute)]
    src: Image,

    #[mev(storage)]
    #[mev(compute)]
    dst: Buffer,
}

#[derive(Arguments)]
struct CvtBufferToImageArguments {
    #[mev(storage)]
    #[mev(compute)]
    src: Buffer,

    #[mev(storage)]
    #[mev(compute)]
    dst: Image,
}

/// Argument layout for the `_resample` `Image->Image` pipeline when the
/// source is `CanonicalKind::Float` (hardware-sampled through `sampler`).
/// Not used for `Uint`/`Sint` source kinds - those have no hardware sampler
/// and reuse the plain [`CvtImageToImageArguments`] layout, doing manual
/// `textureLoad` + `apply_address_mode` instead.
#[derive(Arguments)]
struct CvtImageToImageResampleArguments {
    #[mev(sampled)]
    #[mev(compute)]
    src: Image,

    #[mev(compute)]
    sampler: Sampler,

    #[mev(storage)]
    #[mev(compute)]
    dst: Image,
}

/// `_resample` counterpart of [`CvtImageToImageResampleArguments`] for an
/// `Image->Buffer` pipeline with a `CanonicalKind::Float` source.
#[derive(Arguments)]
struct CvtImageToBufferResampleArguments {
    #[mev(sampled)]
    #[mev(compute)]
    src: Image,

    #[mev(compute)]
    sampler: Sampler,

    #[mev(storage)]
    #[mev(compute)]
    dst: Buffer,
}

#[derive(Arguments)]
struct CvtImageSrcRenderArguments {
    #[mev(sampled)]
    #[mev(fragment)]
    src: Image,
}

/// `_resample` counterpart of [`CvtImageSrcRenderArguments`]
/// for a `CanonicalKind::Float` `Image` source
/// (hardware-sampled through `sampler`).
///
/// Not used for `Uint`/`Sint` source kinds - those have no hardware sampler
/// and reuse the plain [`CvtImageSrcRenderArguments`] layout,
/// doing manual `textureLoad` + `apply_address_mode` instead
/// (mirrors [`CvtImageToImageResampleArguments`] on the compute side).
#[derive(Arguments)]
struct CvtImageSrcRenderResampleArguments {
    #[mev(sampled)]
    #[mev(fragment)]
    src: Image,

    #[mev(fragment)]
    sampler: Sampler,
}

#[derive(Arguments)]
struct CvtBufferSrcRenderArguments {
    #[mev(storage)]
    #[mev(fragment)]
    src: Buffer,
}

/// Generates the `struct Constants { ... }` WGSL text matching whichever of
/// [`CvtConstantsBufferToBuffer`]/[`CvtConstantsBufferToImage`]/
/// [`CvtConstantsImageToBuffer`]/[`CvtConstantsImageToImage`] the
/// `(src.kind, dst.kind)` combination uses, field-for-field in the same
/// declared order (see those structs' definitions for why that order is
/// byte-layout-safe under the `immediate` address space with no explicit
/// `@size`/`@align` needed).
fn cvt_nd_constants_wgsl(src_kind: EndpointKind, dst_kind: EndpointKind) -> &'static str {
    match (src_kind, dst_kind) {
        (EndpointKind::Buffer, EndpointKind::Buffer) => {
            r#"struct Constants {
    dst_extent: array<u32, 4>,
    dst_byte_offset: u32,
    dst_stride: array<u32, 4>,
    src_byte_offset: u32,
    src_stride: array<u32, 4>,
    total_dst_elements: u32,
    src_extent: array<f32, 4>,
    address_mode: array<u32, 4>,
}
"#
        }
        (EndpointKind::Buffer, EndpointKind::Image) => {
            r#"struct Constants {
    dst_extent: array<u32, 4>,
    dst_offset: array<i32, 4>,
    dst_image_size: array<u32, 2>,
    src_byte_offset: u32,
    src_stride: array<u32, 4>,
    total_dst_elements: u32,
    src_extent: array<f32, 4>,
    address_mode: array<u32, 4>,
}
"#
        }
        (EndpointKind::Image, EndpointKind::Buffer) => {
            r#"struct Constants {
    dst_extent: array<u32, 4>,
    dst_byte_offset: u32,
    dst_stride: array<u32, 4>,
    src_offset: array<i32, 4>,
    src_image_size: array<u32, 2>,
    total_dst_elements: u32,
    src_extent: array<f32, 4>,
    address_mode: array<u32, 4>,
}
"#
        }
        (EndpointKind::Image, EndpointKind::Image) => {
            r#"struct Constants {
    dst_extent: array<u32, 4>,
    dst_offset: array<i32, 4>,
    dst_image_size: array<u32, 2>,
    src_offset: array<i32, 4>,
    src_image_size: array<u32, 2>,
    total_dst_elements: u32,
    src_extent: array<f32, 4>,
    address_mode: array<u32, 4>,
}
"#
        }
    }
}

// ============================================================================
// Buffer destination: tiled, non-atomic write dispatch
//
// Every destination `Buffer` write is safe without atomics
// because the dispatch structure below guarantees, purely geometrically,
// that no two GPU threads ever touch the same 4-byte storage word.
//
// Destination axes `0..dst_rank` (X, Y, Z, W - fastest to slowest)
// are classified `tight` (exactly contiguous with the predecessor)
// or `aligned` (word-aligned, possibly with a gap) or `bad`
// (unsupported, panics) - `aligned` is checked *before* `tight`,
// so an axis satisfying both is `aligned`.
//
// `T` (the length of the unbroken tight prefix from axis 0)
// selects one of a small number of generated-WGSL dispatch shapes:
// axes `0..T` collapse into one flat "tight chain" thread index
// (`gid.x`, 4 elements/thread, wrapping across axis boundaries);
// axes `T..dst_rank` (or `0..dst_rank` if `T == 0`)
// are each independently grouped by 4 with no wrapping,
// one per remaining dispatch dimension (`gid.y`/`gid.z`,
// folding two axes into `gid.z` via a secondary flat unravel
// if 3 remain - only possible at `dst_rank == 4`).
//
// See the design notes accompanying this kernel's implementation
// for the full algorithm.
// ============================================================================

/// Computes `T` - the length of the unbroken tight-chain prefix
/// starting at destination axis 0 - for a `Buffer` destination
/// described by `dst_extent`/`dst_stride` (both length `dst_rank`,
/// `dst_rank <= 4`) and `byte_size` (the destination format's element byte size).
///
/// Panics (before any GPU work) if:
/// - the general no-aliasing precondition is violated at any axis
///   (`dst_stride[0] >= byte_size`; `dst_stride[d] >= dst_extent[d-1] *
///   dst_stride[d-1]` for `d >= 1`), or
/// - any axis is classified neither `tight` nor `aligned` ("bad" -
///   unsupported dst layout).
///
/// Classification checks `aligned` (`dst_stride[d] % 4 == 0`) before `tight`
/// (`dst_stride[d] == dst_extent[d-1] * dst_stride[d-1]`,
/// or `dst_stride[0] == byte_size` for axis 0):
/// an axis satisfying both is `aligned`, not `tight`.
///
/// Once the chain breaks (an axis is `aligned`),
/// every subsequent axis is independently `aligned`-or-`bad`
/// the chain never re-extends even if a later axis is numerically tight
/// relative to its immediate predecessor.
fn classify_dst_tight_chain(dst_extent: &[u32], dst_stride: &[usize], byte_size: usize) -> usize {
    let dst_rank = dst_extent.len();
    debug_assert_eq!(dst_stride.len(), dst_rank);

    for d in 0..dst_rank {
        if d == 0 {
            assert!(
                dst_stride[0] >= byte_size,
                "Cvt: dst axis 0 stride ({}) must be >= the destination format's element byte size ({byte_size}) - would alias",
                dst_stride[0]
            );
        } else {
            let prev_span = dst_extent[d - 1] as usize * dst_stride[d - 1];
            assert!(
                dst_stride[d] >= prev_span,
                "Cvt: dst axis {d} stride ({}) must be >= axis {}'s full extent*stride ({prev_span}) - would alias",
                dst_stride[d],
                d - 1
            );
        }
    }

    let mut tight_len = 0;
    let mut chain_open = true;
    for d in 0..dst_rank {
        let aligned = dst_stride[d].is_multiple_of(4);
        if chain_open && !aligned {
            let tight = if d == 0 {
                dst_stride[0] == byte_size
            } else {
                dst_stride[d] == dst_extent[d - 1] as usize * dst_stride[d - 1]
            };
            assert!(
                tight,
                "Cvt: dst axis {d} (stride {}) is neither word-aligned (a multiple of 4) nor tightly packed against its predecessor - unsupported dst layout",
                dst_stride[d]
            );
            tight_len += 1;
        } else {
            chain_open = false;
            assert!(
                aligned,
                "Cvt: dst axis {d} (stride {}) is neither word-aligned (a multiple of 4) nor tightly packed against its predecessor - unsupported dst layout",
                dst_stride[d]
            );
        }
    }

    tight_len
}

/// Derived dispatch-shape facts for a `Buffer` destination's tiled write
/// scheme, purely a function of `(rank, tight_len)` (`tight_len` = `T`,
/// from [`classify_dst_tight_chain`]) - deterministic, so `(rank,
/// tight_len)` alone fully determines the generated WGSL shape and doubles
/// as the compute-pipeline cache-key discriminant (see [`CvtComputeKey`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DstTileUnits {
    tight_len: usize,
    rank: usize,
}

/// How the (up to 2) axes beyond the `gid.x`/`gid.y` units map onto
/// `gid.z`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ZUnit {
    None,
    Single(usize),
    Folded(usize, usize),
}

impl DstTileUnits {
    fn new(rank: usize, tight_len: usize) -> Self {
        debug_assert!(tight_len <= rank);
        DstTileUnits { tight_len, rank }
    }

    /// `true` if `gid.x` carries the tight-chain's flat wrap group index
    /// (`T >= 1`); `false` if `T == 0`, in which case axis 0 itself rides
    /// `gid.x` as just another independently-grouped aligned axis.
    fn x_is_chain(&self) -> bool {
        self.tight_len >= 1
    }

    /// Axis indices handled independently on `gid.y`/`gid.z`
    /// (or folded together into `gid.z`), in increasing order.
    ///
    /// Does not include the `gid.x` unit
    /// (the tight chain, or axis 0 alone if `T == 0`).
    fn remaining(&self) -> Vec<usize> {
        if self.x_is_chain() {
            (self.tight_len..self.rank).collect()
        } else {
            (1..self.rank).collect()
        }
    }

    fn z_unit(&self) -> ZUnit {
        let r = self.remaining();
        match r.len() {
            0 | 1 => ZUnit::None,
            2 => ZUnit::Single(r[1]),
            3 => ZUnit::Folded(r[1], r[2]),
            n => unreachable!(
                "Cvt: dst rank <= 4 implies at most 3 remaining axes beyond the x/y units, got {n}"
            ),
        }
    }
}

/// Number of dispatch groups along each of `gid.x`/`gid.y`/`gid.z`,
/// given the actual runtime `dst_extent` (each group covers up to 4 elements
/// along its unit's axis/axes - see module docs).
///
/// A dimension with no assigned unit is `1` (not `0`):
/// it must stay a no-op single group so the *other* dimensions' real work
/// still executes, unlike a real axis whose own extent is legitimately `0`
/// (correctly yielding an empty dispatch along that axis).
fn dst_tile_groups(units: DstTileUnits, dst_extent: &[u32]) -> (u32, u32, u32) {
    let groups_x = if units.x_is_chain() {
        let total: u64 = dst_extent[..units.tight_len]
            .iter()
            .map(|&e| e as u64)
            .product();
        total.div_ceil(4) as u32
    } else {
        dst_extent[0].div_ceil(4)
    };

    let remaining = units.remaining();
    let groups_for = |axis: usize| dst_extent[axis].div_ceil(4);
    let (groups_y, groups_z) = match remaining.len() {
        0 => (1u32, 1u32),
        1 => (groups_for(remaining[0]), 1u32),
        2 => (groups_for(remaining[0]), groups_for(remaining[1])),
        3 => (
            groups_for(remaining[0]),
            groups_for(remaining[1]) * groups_for(remaining[2]),
        ),
        n => unreachable!("Cvt: dst rank <= 4 implies at most 3 remaining axes, got {n}"),
    };

    (groups_x, groups_y, groups_z)
}

/// The actual dispatch `Extent3` (in workgroups of `[64, 1, 1]`) for a
/// `Buffer` destination's tiled write, given the real runtime `dst_extent`.
fn dst_tile_dispatch_extent(units: DstTileUnits, dst_extent: &[u32]) -> Extent3 {
    let (groups_x, groups_y, groups_z) = dst_tile_groups(units, dst_extent);
    Extent3::new(groups_x.div_ceil(64), groups_y, groups_z)
}

/// One independently-grouped-by-4 "unit" of the tiled dispatch: either the
/// tight chain (`chain: true`, covering `axes` `0..T` via one flat wrapped
/// sub-index) or a single aligned axis (`chain: false`, `axes` holding
/// exactly one index).
struct DstTileUnit {
    axes: Vec<usize>,
    base_expr: String,
    chain: bool,
}

/// Builds the ordered list of dispatch units for `units`: the `gid.x` unit
/// first (tight chain or axis-0-alone), then the remaining aligned axes
/// riding `gid.y`/`gid.z` (or `group_zb`/`group_zc`, the two locals
/// [`gen_dst_buffer_tiled_body`] emits when 3 axes remain and 2 must fold
/// into `gid.z`).
fn dst_tile_unit_list(units: DstTileUnits) -> Vec<DstTileUnit> {
    let mut list = Vec::new();
    if units.x_is_chain() {
        list.push(DstTileUnit {
            axes: (0..units.tight_len).collect(),
            base_expr: "gid.x".to_string(),
            chain: true,
        });
    } else {
        list.push(DstTileUnit {
            axes: vec![0],
            base_expr: "gid.x".to_string(),
            chain: false,
        });
    }

    let remaining = units.remaining();
    match remaining.len() {
        0 => {}
        1 => list.push(DstTileUnit {
            axes: vec![remaining[0]],
            base_expr: "gid.y".to_string(),
            chain: false,
        }),
        2 => {
            list.push(DstTileUnit {
                axes: vec![remaining[0]],
                base_expr: "gid.y".to_string(),
                chain: false,
            });
            list.push(DstTileUnit {
                axes: vec![remaining[1]],
                base_expr: "gid.z".to_string(),
                chain: false,
            });
        }
        3 => {
            list.push(DstTileUnit {
                axes: vec![remaining[0]],
                base_expr: "gid.y".to_string(),
                chain: false,
            });
            list.push(DstTileUnit {
                axes: vec![remaining[1]],
                base_expr: "group_zb".to_string(),
                chain: false,
            });
            list.push(DstTileUnit {
                axes: vec![remaining[2]],
                base_expr: "group_zc".to_string(),
                chain: false,
            });
        }
        n => unreachable!("Cvt: dst rank <= 4 implies at most 3 remaining axes, got {n}"),
    }
    list
}

/// Generates the full tiled, non-atomic dispatch body for a `Buffer`
/// destination (see module docs above).
///
/// Emits its own `gid.x` bounds check
/// (the workgroup-of-64 rounding tail along the `gid.x` unit)
/// and, for every combination of up to 4 sub-indices per unit,
/// a scoped block computing that sub-element's `dst_c{d}` coordinates,
/// decoding/converting via [`gen_endpoint_decode_convert`]
/// (unchanged - called once per in-bounds sub-element instead of once per thread),
/// then encoding via [`gen_buffer_encode`]'s plain (non-atomic) masked read-modify-write.
///
/// Each combination's block is its own WGSL scope (`{ ... }`),
/// so the repeated `dst_c{d}`/`out_r`/etc. declarations across combinations never collide.
#[allow(clippy::too_many_arguments)]
fn gen_dst_buffer_tiled_body(
    src: EndpointDesc,
    dst_layout: &FormatLayout,
    src_layout: &FormatLayout,
    swizzle: Swizzle,
    axes: AxisSwizzle,
    src_rank: usize,
    dst_rank: usize,
    resample: bool,
    wrap: bool,
    units: DstTileUnits,
) -> String {
    let mut body = String::new();
    let t = units.tight_len;
    let x_is_chain = units.x_is_chain();

    // `gid.x`'s own bound: the workgroup-of-64 rounding tail along the
    // tight-chain-or-axis-0 unit (every other dispatch dimension is
    // launched with an *exact* group count, so no bound check is needed
    // there).
    if x_is_chain {
        let mut total_expr = String::new();
        for d in 0..t {
            if d > 0 {
                total_expr += " * ";
            }
            total_expr += &format!("constants.dst_extent[{d}]");
        }
        body += &format!("    let tight_total: u32 = {total_expr};\n");
    } else {
        body += "    let tight_total: u32 = constants.dst_extent[0];\n";
    }
    body += "    let groups_x: u32 = (tight_total + 3u) / 4u;\n";
    body += "    if (gid.x >= groups_x) {\n        return;\n    }\n";

    // Unravel `gid.z` into its two folded group indices, if 3 axes remain.
    if let ZUnit::Folded(b_axis, _c_axis) = units.z_unit() {
        body += &format!("    let groups_zb: u32 = (constants.dst_extent[{b_axis}] + 3u) / 4u;\n");
        body += "    let group_zb: u32 = gid.z % groups_zb;\n";
        body += "    let group_zc: u32 = gid.z / groups_zb;\n";
    }

    let unit_list = dst_tile_unit_list(units);

    // Precompute each unit's 4 sub-id variants once (its coordinate(s) and
    // an in-bounds flag), avoiding redundant axis arithmetic across the
    // combinations generated below.
    for (u, unit) in unit_list.iter().enumerate() {
        for s in 0..4u32 {
            if unit.chain {
                body += &format!("    let eid_v{u}_{s}: u32 = gid.x * 4u + {s}u;\n");
                body += &format!("    let ok_v{u}_{s}: bool = eid_v{u}_{s} < tight_total;\n");
                body += &format!("    var rem_v{u}_{s}: u32 = eid_v{u}_{s};\n");
                for &axis in &unit.axes {
                    body += &format!(
                        "    let dst_c{axis}_v{u}_{s}: u32 = rem_v{u}_{s} % constants.dst_extent[{axis}];\n"
                    );
                    body += &format!(
                        "    rem_v{u}_{s} = rem_v{u}_{s} / constants.dst_extent[{axis}];\n"
                    );
                }
            } else {
                let axis = unit.axes[0];
                body += &format!(
                    "    let dst_c{axis}_v{u}_{s}: u32 = {} * 4u + {s}u;\n",
                    unit.base_expr
                );
                body += &format!(
                    "    let ok_v{u}_{s}: bool = dst_c{axis}_v{u}_{s} < constants.dst_extent[{axis}];\n"
                );
            }
        }
    }

    // Cartesian product over every unit's sub-id (0..4 each): one scoped
    // block per combination.
    let n_units = unit_list.len();
    let mut subids = vec![0u32; n_units];
    loop {
        let mut cond = String::new();
        let mut coord_lets = String::new();
        for (u, unit) in unit_list.iter().enumerate() {
            let s = subids[u];
            if u > 0 {
                cond += " && ";
            }
            cond += &format!("ok_v{u}_{s}");
            for &axis in &unit.axes {
                coord_lets += &format!("        let dst_c{axis}: u32 = dst_c{axis}_v{u}_{s};\n");
            }
        }

        body += &format!("    if ({cond}) {{\n");
        body += &coord_lets;
        body += &gen_endpoint_decode_convert(
            src, dst_layout, src_layout, swizzle, axes, src_rank, false, resample, wrap,
        );

        let mut dst_byte_expr = "constants.dst_byte_offset".to_string();
        for d in 0..dst_rank {
            dst_byte_expr += &format!(" + dst_c{d} * constants.dst_stride[{d}]");
        }
        body += &format!("        let dst_elem_byte: u32 = {dst_byte_expr};\n");
        body += &gen_buffer_encode("dst", "dst_elem_byte", dst_layout);
        body += "    }\n";

        // Odometer-increment `subids` (last unit fastest-varying); stop
        // once incrementing carries out past unit 0 (all combos done).
        let mut i = n_units;
        let mut carry = true;
        while carry && i > 0 {
            i -= 1;
            subids[i] += 1;
            if subids[i] == 4 {
                subids[i] = 0;
            } else {
                carry = false;
            }
        }
        if carry {
            break;
        }
    }

    body
}

/// Builds the compute pipeline for one specific `(src, dst, swizzle, ranks,
/// axes, resample)` combination, for every endpoint-kind pairing except
/// sRGB `Image` destinations (WGSL storage textures have no sRGB write
/// format - those route through [`make_render_pipeline`] instead; callers
/// must reject sRGB `Image` destinations for `_resample` entry points
/// before reaching here, since the render pipeline does not support
/// `resample`).
///
/// `src_rank`/`dst_rank` (each <= 4) and `axes` are baked into the
/// generated WGSL as compile-time-unrolled code (no runtime rank
/// branching), matching the `CvtComputeKey` cache key that selects this
/// pipeline. `Image` endpoints are always addressed at rank 3 by every
/// current call site (2 spatial axes + 1 always-degenerate iteration
/// axis), but this function does not assume that - a `Buffer` endpoint
/// paired with an `Image` endpoint may be described at any rank up to 4.
///
/// When `dst.kind == Buffer`, `dst_tight_len` (`T`, from
/// [`classify_dst_tight_chain`]) selects the generated tiled non-atomic
/// dispatch shape (see [`gen_dst_buffer_tiled_body`]); ignored otherwise
/// (`Image` destinations keep the original one-thread-per-element scheme -
/// `textureStore` never aliases across threads, so it never needed this).
///
/// When `resample` is `true` and the source is an `Image`
/// of `CanonicalKind::Float`, an extra `sampler` binding is declared
/// and the argument layout switches to the matching `*ResampleArguments` struct (see [`gen_endpoint_decode_convert`]).
///
/// Every other `(src, dst)` combination keeps using the plain argument struct
/// regardless of `resample`, since only the hardware-sampled path needs
/// a bound `Sampler` resource (integer-image and buffer resample paths
/// do their own address-mode wrapping from a plain `u32` uniform,
/// no extra binding required).
#[allow(clippy::too_many_arguments)]
fn make_compute_pipeline_nd(
    device: &Device,
    src: EndpointDesc,
    dst: EndpointDesc,
    swizzle: Swizzle,
    src_rank: usize,
    dst_rank: usize,
    axes: AxisSwizzle,
    resample: bool,
    wrap: bool,
    dst_tight_len: usize,
) -> ComputePipeline {
    let src_layout = describe(src.format).expect("Cvt: src format is not describable");
    let dst_layout = describe(dst.format).expect("Cvt: dst format is not describable");
    assert!(
        !(dst.kind == EndpointKind::Image && dst_layout.srgb),
        "Cvt: sRGB Image destinations must use the render pipeline"
    );

    let dst_is_image = dst.kind == EndpointKind::Image;
    let needs_sampler =
        resample && src.kind == EndpointKind::Image && src_layout.kind == CanonicalKind::Float;

    let mut body = String::new();

    if dst_is_image {
        // Unravel the linear destination index into per-axis `dst_c{d}`
        // coordinates (row-major, fastest-varying last) - one thread per
        // destination element; safe unconditionally since `textureStore`
        // never aliases across threads (never needed the tiled scheme
        // below, which is `Buffer`-destination-only).
        body += "    var rem = linear;\n";
        for d in 0..dst_rank {
            body += &format!(
                "    let dst_c{d} = rem % constants.dst_extent[{d}];\n    rem = rem / constants.dst_extent[{d}];\n"
            );
        }

        body += &gen_endpoint_decode_convert(
            src,
            &dst_layout,
            &src_layout,
            swizzle,
            axes,
            src_rank,
            true,
            resample,
            wrap,
        );
    }

    let (bindings, arguments_layout): (String, &'static str) = match (src.kind, dst.kind) {
        (EndpointKind::Buffer, EndpointKind::Buffer) => (
            "@group(0) @binding(0) var<storage, read> src: array<u32>;\n@group(0) @binding(1) var<storage, read_write> dst: array<u32>;\n".to_string(),
            "CvtNdArguments",
        ),
        (EndpointKind::Image, EndpointKind::Image) if needs_sampler => (
            format!(
                "@group(0) @binding(0) var src: texture_2d<{}>;\n@group(0) @binding(1) var src_sampler: sampler;\n@group(0) @binding(2) var dst: texture_storage_2d<{}, write>;\n",
                wgsl_scalar(src_layout.kind),
                dst_image_storage_format(&dst_layout, dst.format),
            ),
            "CvtImageToImageResampleArguments",
        ),
        (EndpointKind::Image, EndpointKind::Image) => (
            format!(
                "@group(0) @binding(0) var src: texture_2d<{}>;\n@group(0) @binding(1) var dst: texture_storage_2d<{}, write>;\n",
                wgsl_scalar(src_layout.kind),
                dst_image_storage_format(&dst_layout, dst.format),
            ),
            "CvtImageToImageArguments",
        ),
        (EndpointKind::Image, EndpointKind::Buffer) if needs_sampler => (
            format!(
                "@group(0) @binding(0) var src: texture_2d<{}>;\n@group(0) @binding(1) var src_sampler: sampler;\n@group(0) @binding(2) var<storage, read_write> dst: array<u32>;\n",
                wgsl_scalar(src_layout.kind),
            ),
            "CvtImageToBufferResampleArguments",
        ),
        (EndpointKind::Image, EndpointKind::Buffer) => (
            format!(
                "@group(0) @binding(0) var src: texture_2d<{}>;\n@group(0) @binding(1) var<storage, read_write> dst: array<u32>;\n",
                wgsl_scalar(src_layout.kind),
            ),
            "CvtImageToBufferArguments",
        ),
        (EndpointKind::Buffer, EndpointKind::Image) => (
            format!(
                "@group(0) @binding(0) var<storage, read> src: array<u32>;\n@group(0) @binding(1) var dst: texture_storage_2d<{}, write>;\n",
                dst_image_storage_format(&dst_layout, dst.format),
            ),
            "CvtBufferToImageArguments",
        ),
    };

    if dst_is_image {
        body += "    let dst_px0 = constants.dst_offset[0] + i32(dst_c0);\n    let dst_px1 = constants.dst_offset[1] + i32(dst_c1);\n";
        body += "    if (dst_px0 < 0 || dst_px1 < 0 || u32(dst_px0) >= constants.dst_image_size[0] || u32(dst_px1) >= constants.dst_image_size[1]) {\n        return;\n    }\n";
        let scalar = wgsl_scalar(dst_layout.kind);
        body += &format!(
            "    textureStore(dst, vec2<i32>(dst_px0, dst_px1), vec4<{scalar}>(out_r, out_g, out_b, out_a));\n"
        );
    } else {
        let units = DstTileUnits::new(dst_rank, dst_tight_len);
        body += &gen_dst_buffer_tiled_body(
            src,
            &dst_layout,
            &src_layout,
            swizzle,
            axes,
            src_rank,
            dst_rank,
            resample,
            wrap,
            units,
        );
    }

    let mut helpers = String::new();
    if needs_half_float(&src_layout) || needs_half_float(&dst_layout) {
        helpers += HALF_FLOAT_HELPERS;
    }
    if (resample || wrap) && !needs_sampler {
        helpers += ADDRESS_MODE_HELPER;
    }

    // `Image` destinations still dispatch one thread per destination
    // element off a flat `gid.x` (`linear`), bounded by
    // `total_dst_elements`; `Buffer` destinations generate their own
    // `gid.x`/`gid.y`/`gid.z` bounds checks inside `body` (see
    // `gen_dst_buffer_tiled_body`) and need no preamble here.
    let entry_preamble = if dst_is_image {
        "    let linear = gid.x;\n    if (linear >= constants.total_dst_elements) {\n        return;\n    }\n"
    } else {
        ""
    };

    let constants_wgsl = cvt_nd_constants_wgsl(src.kind, dst.kind);

    let code = format!(
        r#"
{constants_wgsl}
var<immediate> constants: Constants;
{bindings}{helpers}
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
{entry_preamble}{body}
}}
"#,
    );

    let library = device
        .new_shader_library(LibraryDesc {
            name: "cvt_nd",
            input: LibraryInput::wgsl(code),
        })
        .expect("Cvt: failed to compile nd shader library");

    let arguments = match arguments_layout {
        "CvtNdArguments" => CvtNdArguments::LAYOUT,
        "CvtImageToImageArguments" => CvtImageToImageArguments::LAYOUT,
        "CvtImageToImageResampleArguments" => CvtImageToImageResampleArguments::LAYOUT,
        "CvtImageToBufferArguments" => CvtImageToBufferArguments::LAYOUT,
        "CvtImageToBufferResampleArguments" => CvtImageToBufferResampleArguments::LAYOUT,
        "CvtBufferToImageArguments" => CvtBufferToImageArguments::LAYOUT,
        _ => unreachable!(),
    };

    let constants_size = match (src.kind, dst.kind) {
        (EndpointKind::Buffer, EndpointKind::Buffer) => CvtConstantsBufferToBuffer::SIZE,
        (EndpointKind::Buffer, EndpointKind::Image) => CvtConstantsBufferToImage::SIZE,
        (EndpointKind::Image, EndpointKind::Buffer) => CvtConstantsImageToBuffer::SIZE,
        (EndpointKind::Image, EndpointKind::Image) => CvtConstantsImageToImage::SIZE,
    };

    device
        .new_compute_pipeline(ComputePipelineDesc {
            name: "cvt_nd",
            shader: library.entry("main"),
            work_group_size: [64, 1, 1],
            constants: constants_size,
            arguments: &[arguments],
        })
        .expect("Cvt: failed to create nd compute pipeline")
}

/// Builds the render pipeline for an sRGB `Image` destination
/// (fullscreen triangle + fragment shader; hardware gamma-encodes on write
/// to the sRGB color attachment, mirroring `Blit`'s `BlitKind::Srgb` path exactly).
///
/// The destination is always an `Image` addressed at rank 3
/// (2 spatial axes driven by `frag_pos`, one always-degenerate iteration axis)
/// - only `Image` destinations reach this pipeline in the first place.
///
/// `src_rank` (<= 4) generalizes the source endpoint's addressed rank: for
/// an `Image` source this is always 3 (only 2 axes are ever addressable -
/// see [`gen_endpoint_decode_convert`]), but a `Buffer` source may be
/// described at any rank up to 4, exactly like the compute path's `Buffer`
/// endpoints.
///
/// When `resample` is `true` and the source is an `Image`
/// of `CanonicalKind::Float`, an extra `sampler` binding is declared
/// and the argument layout switches to [`CvtImageSrcRenderResampleArguments`],
/// mirroring how `make_compute_pipeline_nd` picks
/// its own `*ResampleArguments` struct for the same condition.
///
/// Every other `(src, dst)` combination keeps using the plain argument struct
/// regardless of `resample` (integer-image and buffer resample paths
/// do their own address-mode wrapping from a plain `u32` uniform,
/// no extra binding required).
#[allow(clippy::too_many_arguments)]
fn make_render_pipeline(
    device: &Device,
    src: EndpointDesc,
    dst_format: PixelFormat,
    swizzle: Swizzle,
    axes: AxisSwizzle,
    src_rank: usize,
    resample: bool,
    wrap: bool,
) -> RenderPipeline {
    let src_layout = describe(src.format).expect("Cvt: src format is not describable");
    let dst_layout = describe(dst_format).expect("Cvt: dst format is not describable");

    let mut body = String::new();
    body += "    let dst_c0 = u32(clamp(i32(frag_pos.x) - constants.dst_offset[0], 0, i32(constants.dst_extent[0]) - 1));\n";
    body += "    let dst_c1 = u32(clamp(i32(frag_pos.y) - constants.dst_offset[1], 0, i32(constants.dst_extent[1]) - 1));\n";
    body += "    let dst_c2 = 0u;\n";

    body += &gen_endpoint_decode_convert(
        src,
        &dst_layout,
        &src_layout,
        swizzle,
        axes,
        src_rank,
        true,
        resample,
        wrap,
    );
    body += "    return vec4<f32>(out_r, out_g, out_b, out_a);\n";

    let needs_sampler =
        resample && src.kind == EndpointKind::Image && src_layout.kind == CanonicalKind::Float;

    let (binding, arguments_layout): (&str, &'static str) = match src.kind {
        EndpointKind::Image if needs_sampler => (
            "@group(0) @binding(0) var src: texture_2d<f32>;\n@group(0) @binding(1) var src_sampler: sampler;\n",
            "CvtImageSrcRenderResampleArguments",
        ),
        EndpointKind::Image => (
            "@group(0) @binding(0) var src: texture_2d<f32>;\n",
            "CvtImageSrcRenderArguments",
        ),
        EndpointKind::Buffer => (
            "@group(0) @binding(0) var<storage, read> src: array<u32>;\n",
            "CvtBufferSrcRenderArguments",
        ),
    };

    let mut helpers = String::new();
    if needs_half_float(&src_layout) {
        helpers += HALF_FLOAT_HELPERS;
    }
    if (resample || wrap) && !needs_sampler {
        helpers += ADDRESS_MODE_HELPER;
    }

    // The render pipeline's destination is always `Image` (see module
    // docs), so only the `BufferToImage`/`ImageToImage` constants shapes
    // are ever reachable here.
    let constants_wgsl = cvt_nd_constants_wgsl(src.kind, EndpointKind::Image);

    let code = format!(
        r#"
{constants_wgsl}
var<immediate> constants: Constants;
{binding}{helpers}
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {{
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {{
{body}
}}
"#,
    );

    let library = device
        .new_shader_library(LibraryDesc {
            name: "cvt_srgb",
            input: LibraryInput::wgsl(code),
        })
        .expect("Cvt: failed to compile srgb shader library");

    let arguments = match arguments_layout {
        "CvtImageSrcRenderArguments" => CvtImageSrcRenderArguments::LAYOUT,
        "CvtImageSrcRenderResampleArguments" => CvtImageSrcRenderResampleArguments::LAYOUT,
        "CvtBufferSrcRenderArguments" => CvtBufferSrcRenderArguments::LAYOUT,
        _ => unreachable!(),
    };

    let constants_size = match src.kind {
        EndpointKind::Buffer => CvtConstantsBufferToImage::SIZE,
        EndpointKind::Image => CvtConstantsImageToImage::SIZE,
    };

    device
        .new_render_pipeline(RenderPipelineDesc {
            name: "cvt_srgb",
            vertex_shader: library.entry("vs_main"),
            vertex_attributes: vec![],
            vertex_layouts: vec![],
            primitive_topology: PrimitiveTopology::Triangle,
            raster: Some(RasterDesc {
                fragment_shader: Some(library.entry("fs_main")),
                color_targets: vec![ColorTargetDesc::new(dst_format)],
                depth_stencil: None,
                front_face: FrontFace::default(),
                culling: Culling::None,
            }),
            arguments: &[arguments],
            constants: constants_size,
        })
        .expect("Cvt: failed to create srgb render pipeline")
}

// ============================================================================
// Buffer in-place compute pipeline (single-resource)
// ============================================================================

/// Argument layout for the dedicated in-place pipeline: a single
/// `storage, read_write` buffer binding, not two `Buffer` handles aliasing
/// the same allocation through two `Arguments` fields.
#[derive(Arguments)]
struct CvtInPlaceArguments {
    #[mev(storage)]
    #[mev(compute)]
    buf: Buffer,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod, mev::AutoDeviceRepr)]
struct CvtInPlaceConstants {
    byte_offset: u32,
    element_count: u32,
}

fn make_in_place_pipeline(
    device: &Device,
    src_format: PixelFormat,
    dst_format: PixelFormat,
    swizzle: Swizzle,
) -> ComputePipeline {
    let src_layout = describe(src_format).expect("Cvt: src format is not describable");
    let dst_layout = describe(dst_format).expect("Cvt: dst format is not describable");

    assert_eq!(
        src_layout.byte_size, dst_layout.byte_size,
        "Cvt: in-place conversion requires src/dst element byte size to match"
    );

    let byte_size = dst_layout.byte_size;

    // Always rank-1 and inherently tight by contract
    // (flat `byte_offset + linear*byte_size` addressing, no stride concept
    // at all - `T = D = 1`, no outer aligned axes ever needed):
    // the simplest instance of the tiled non-atomic scheme
    // (see module docs above `classify_dst_tight_chain`).
    //
    // Each thread owns 4 consecutive elements,
    // wrapping is always safe since there's nothing
    // to wrap into but more of the same tightly-packed run.
    let mut body = String::new();
    body += "    let groups: u32 = (constants.element_count + 3u) / 4u;\n";
    body += "    if (gid.x >= groups) {\n        return;\n    }\n";

    for s in 0..4u32 {
        body += &format!("    let eid_{s}: u32 = gid.x * 4u + {s}u;\n");
        body += &format!("    if (eid_{s} < constants.element_count) {{\n");
        body += &format!(
            "        let elem_byte_{s}: u32 = constants.byte_offset + eid_{s} * {byte_size}u;\n"
        );
        for &ch in &CHANNELS {
            body += &gen_buffer_decode_channel(
                &format!("src_{}", ch.lower()),
                "buf",
                &format!("elem_byte_{s}"),
                &src_layout,
                ch,
            );
        }
        body += &gen_swizzle(swizzle, src_layout.kind);
        for &ch in &CHANNELS {
            let var = format!("out_{}", ch.lower());
            let sw_var = format!("sw_{}", ch.lower());
            let (line, _ty) = gen_convert_channel(&var, &sw_var, &src_layout, &dst_layout, false);
            body += &line;
        }
        body += &gen_buffer_encode("buf", &format!("elem_byte_{s}"), &dst_layout);
        body += "    }\n";
    }

    let mut helpers = String::new();
    if needs_half_float(&src_layout) || needs_half_float(&dst_layout) {
        helpers += HALF_FLOAT_HELPERS;
    }

    let code = format!(
        r#"
struct Constants {{
    byte_offset: u32,
    element_count: u32,
}}

var<immediate> constants: Constants;
@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
{helpers}
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
{body}
}}
"#,
    );

    let library = device
        .new_shader_library(LibraryDesc {
            name: "cvt_in_place",
            input: LibraryInput::wgsl(code),
        })
        .expect("Cvt: failed to compile in-place shader library");

    device
        .new_compute_pipeline(ComputePipelineDesc {
            name: "cvt_in_place",
            shader: library.entry("main"),
            work_group_size: [64, 1, 1],
            constants: CvtInPlaceConstants::SIZE,
            arguments: &[CvtInPlaceArguments::LAYOUT],
        })
        .expect("Cvt: failed to create in-place compute pipeline")
}

// ============================================================================
// Small helpers shared by the public API
// ============================================================================

fn pad4_u32<const N: usize>(a: [u32; N]) -> [u32; 4] {
    let mut out = [0u32; 4];
    out[..N].copy_from_slice(&a);
    out
}

fn pad4_i32<const N: usize>(a: [i32; N]) -> [i32; 4] {
    let mut out = [0i32; 4];
    out[..N].copy_from_slice(&a);
    out
}

fn pad4_f32<const N: usize>(a: [f32; N]) -> [f32; 4] {
    let mut out = [0f32; 4];
    out[..N].copy_from_slice(&a);
    out
}

fn pad4_addr_mode<const N: usize>(a: [AddressMode; N]) -> [u32; 4] {
    let mut out = [0u32; 4];
    for i in 0..N {
        out[i] = a[i] as u32;
    }
    out
}

/// Converts an arbitrary-rank `address_mode` array into the fixed rank-3
/// array required by `SamplerDesc` (hardware samplers are always built
/// against a 2D `Image`, so only indices 0/1 are ever meaningfully read
/// - see the module's Image-endpoint decode codegen).
///
/// Extra entries beyond index 2 are dropped;
/// missing entries are filled with `AddressMode`'s default.
fn sampler_addr_mode3<const N: usize>(a: [AddressMode; N]) -> [AddressMode; 3] {
    let mut out = [AddressMode::default(); 3];
    let n = N.min(3);
    out[..n].copy_from_slice(&a[..n]);
    out
}

/// Panics unless the strided region (`extent` elements per axis, laid out
/// by `strides`, each element `elem_byte_size` bytes) fits within `slice`,
/// starting at the slice's own byte offset
/// - the buffer-endpoint counterpart of [`ImageSlice::assert_fits_image`].
///
/// An empty region (any zero extent) touches nothing and always fits.
fn assert_fits_buffer_slice(
    role: &str,
    slice: &BufferSlice<'_>,
    strides: &[usize],
    extent: &[u32],
    elem_byte_size: usize,
) {
    if extent.contains(&0) {
        return;
    }
    let mut span = elem_byte_size;
    for (i, &stride) in strides.iter().enumerate() {
        span += (extent[i] as usize - 1) * stride;
    }
    assert!(
        span <= slice.size(),
        "Cvt: {role} region spans {span} bytes but the buffer slice holds only {} bytes past its offset",
        slice.size()
    );
}

// ============================================================================
// Per-pair options
// ============================================================================

/// Options shared by all four `Cvt` pair `_with` methods
/// (`buffer_to_buffer_with`, `image_to_image_with`, `image_to_buffer_with`, `buffer_to_image_with`).
///
/// Defaults to a plain exact copy: identity axis map, identity swizzle,
/// no resize, out-of-bounds source reads denied.
///
/// `resize`'s two independent filters (`min_filter`, `mag_filter`, mirroring
/// [`SamplerDesc`]'s fields of the same name) are only meaningful
/// for an `Image` src with a Float-kind pixel format,
/// where they drive a real hardware `Sampler`.
///
/// In every other case - a `Buffer` src (no hardware sampler at all)
/// or an `Image` src with an integer-kind (Uint/Sint) pixel format
/// (no hardware sampler for integer texture formats) - both filters
/// must be `Filter::Nearest`; passing `Filter::Linear` in either slot panics
/// (before recording any GPU work) at the call site that can't honor it.
///
/// Accepted directly at the call site (via `Into`): an [`AxisSwizzle`],
/// a [`Swizzle`], a [`Filter`] (enables `resize`, setting both min and mag
/// to the same value), an `[AddressMode; SRC]` (enables out-of-bounds reads),
/// `(AxisSwizzle, Swizzle)` or `(Filter, [AddressMode; SRC])`.
///
/// Any other combination goes through the builder methods or a struct literal.
#[derive(Clone, Copy, Debug)]
pub struct CvtOptions<const SRC: usize> {
    /// Per-source-axis map from destination axes
    /// (rotation/mirror/splat/diagonal-read).
    pub axes: AxisSwizzle,
    /// Component swizzle applied between decode and encode.
    pub swizzle: Swizzle,
    /// `None`: exact copy - extents must match along every mapped axis.
    /// `Some((min_filter, mag_filter))`: ratio-based resample using these
    /// filters - see the struct docs for when non-`Nearest` values are
    /// actually honored versus panicking.
    pub resize: Option<(Filter, Filter)>,
    /// `None`: the source region must fit its resource (panics otherwise).
    ///
    /// `Some(modes)`: out-of-range source reads wrap/clamp/mirror per axis instead.
    ///
    /// Never affects destination writes.
    pub address_mode: Option<[AddressMode; SRC]>,
}

impl<const SRC: usize> Default for CvtOptions<SRC> {
    fn default() -> Self {
        CvtOptions {
            axes: AxisSwizzle::IDENTITY,
            swizzle: Swizzle::IDENTITY,
            resize: None,
            address_mode: None,
        }
    }
}

impl<const SRC: usize> CvtOptions<SRC> {
    pub fn axes(mut self, axes: AxisSwizzle) -> Self {
        self.axes = axes;
        self
    }

    pub fn swizzle(mut self, swizzle: Swizzle) -> Self {
        self.swizzle = swizzle;
        self
    }

    /// Enables `resize`, setting both `min_filter` and `mag_filter` to
    /// `filter`.
    pub fn filter(mut self, filter: Filter) -> Self {
        self.resize = Some((filter, filter));
        self
    }

    /// Enables `resize`, setting `min_filter`/`mag_filter` independently
    /// from `(min, mag)`.
    pub fn resize(mut self, (min, mag): (Filter, Filter)) -> Self {
        self.resize = Some((min, mag));
        self
    }

    pub fn address_mode(mut self, modes: [AddressMode; SRC]) -> Self {
        self.address_mode = Some(modes);
        self
    }
}

impl<const SRC: usize> From<AxisSwizzle> for CvtOptions<SRC> {
    fn from(axes: AxisSwizzle) -> Self {
        CvtOptions::default().axes(axes)
    }
}

impl<const SRC: usize> From<Swizzle> for CvtOptions<SRC> {
    fn from(swizzle: Swizzle) -> Self {
        CvtOptions::default().swizzle(swizzle)
    }
}

impl<const SRC: usize> From<Filter> for CvtOptions<SRC> {
    fn from(filter: Filter) -> Self {
        CvtOptions::default().filter(filter)
    }
}

impl<const SRC: usize> From<[AddressMode; SRC]> for CvtOptions<SRC> {
    fn from(modes: [AddressMode; SRC]) -> Self {
        CvtOptions::default().address_mode(modes)
    }
}

impl<const SRC: usize> From<(AxisSwizzle, Swizzle)> for CvtOptions<SRC> {
    fn from((axes, swizzle): (AxisSwizzle, Swizzle)) -> Self {
        CvtOptions::default().axes(axes).swizzle(swizzle)
    }
}

impl<const SRC: usize> From<(Filter, [AddressMode; SRC])> for CvtOptions<SRC> {
    fn from((filter, modes): (Filter, [AddressMode; SRC])) -> Self {
        CvtOptions::default().filter(filter).address_mode(modes)
    }
}

// ============================================================================
// Public API
// ============================================================================

impl Cvt {
    pub fn new(device: &Device) -> Self {
        Cvt {
            device: device.clone(),
            compute_pipelines: Mutex::new(HashMap::new()),
            render_pipelines: Mutex::new(HashMap::new()),
            in_place_pipelines: Mutex::new(HashMap::new()),
            samplers: Mutex::new(HashMap::new()),
        }
    }

    /// Panics if any `MappedAxis::Dst` axis's source/destination extents
    /// differ (exact-index entry points do not resample).
    fn assert_exact_extents<const SRC: usize, const DST: usize>(
        axis_map: AxisSwizzle,
        src_extent: [u32; SRC],
        dst_extent: [u32; DST],
    ) {
        for s in 0..SRC {
            if let AxisMap::Direct(axis) | AxisMap::Mirror(axis) = axis_map.index(s) {
                assert_eq!(
                    src_extent[s], dst_extent[axis as usize],
                    "Cvt: requires matching extents along mapped axes (no resampling); source axis {s} -> destination axis {axis}"
                );
            }
        }
    }

    fn sampler_for(&self, sampler_desc: SamplerDesc) -> Sampler {
        let mut samplers = self.samplers.lock();
        samplers
            .entry(sampler_desc)
            .or_insert_with(|| self.device.new_sampler(sampler_desc))
            .clone()
    }

    /// Converts `src` into `dst`, both `Buffer`s of the same rank `N`, with
    /// default options: exact copy, identity axis map and swizzle,
    /// out-of-bounds reads denied (src/dst element formats may still differ).
    /// Use [`Cvt::buffer_to_buffer_with`] for anything else.
    pub fn buffer_to_buffer<const N: usize>(
        &self,
        encoder: &mut CommandEncoder,
        src: impl AsBufferSlice,
        src_shape: TexelShape<N>,
        dst: impl AsBufferSlice,
        dst_shape: TexelShape<N>,
    ) {
        self.buffer_to_buffer_with::<N, N>(
            encoder,
            src,
            src_shape,
            dst,
            dst_shape,
            CvtOptions::default(),
        );
    }

    /// Converts `src`'s contents into `dst`, both `Buffer` endpoints,
    /// up to 4 dimensions, per `opts`
    /// (see [`CvtOptions`], including the values accepted directly via `Into`).
    ///
    /// With `opts.resize` unset this is an exact-index copy:
    /// extents must match along every mapped axis.
    ///
    /// With it set, each destination element reads the source coordinate
    /// computed by scaling the destination's position ratio across `src_extent` - always nearest-index,
    /// wrapped by the address mode against the `src_extent` region itself
    /// (there is no larger "physical buffer extent" concept in this API,
    /// unlike `Image` endpoints which wrap against their real physical extent).
    ///
    /// With `opts.address_mode` unset, the source region must fit
    /// within `src`'s slice (asserted; where the resample path needs
    /// a mode value anyway, `ClampToEdge` is used).
    ///
    /// The destination region must fit within `dst`'s slice unconditionally
    /// - address mode never redirects writes.
    ///
    /// Panics (before recording any GPU work) if the formats are not `compatible`,
    /// or `SRC`/`DST` exceed 4,
    /// or a region fails its fit check as above,
    /// or any byte offset/stride isn't a multiple of its side's component byte size
    /// (1/2/4 - required so that no component ever straddles
    /// a 4-byte storage-buffer word),
    /// or the destination stride/offset pattern fails the tiled
    /// non-atomic write dispatch's own preconditions (no-aliasing stride requirement,
    /// 4-byte-aligned `dst_byte_offset`, every axis classified `tight` or `aligned`
    /// - see `classify_dst_tight_chain`).
    pub fn buffer_to_buffer_with<const SRC: usize, const DST: usize>(
        &self,
        encoder: &mut CommandEncoder,
        src: impl AsBufferSlice,
        src_shape: TexelShape<SRC>,
        dst: impl AsBufferSlice,
        dst_shape: TexelShape<DST>,
        opts: impl Into<CvtOptions<SRC>>,
    ) {
        #![allow(clippy::needless_range_loop)]

        let CvtOptions {
            axes,
            swizzle,
            resize,
            address_mode,
        } = opts.into();

        if let Some((min, mag)) = resize {
            assert!(
                min == Filter::Nearest && mag == Filter::Nearest,
                "Cvt: buffer source has no sampler, filter must be Filter::Nearest"
            );
        }
        let resample = resize.is_some();

        assert!(SRC <= 4, "Cvt: buffer<->buffer source rank must be <= 4");
        assert!(
            DST <= 4,
            "Cvt: buffer<->buffer destination rank must be <= 4"
        );

        let src_slice = src.as_buffer_slice();
        let dst_slice = dst.as_buffer_slice();

        let src_byte_offset = src_slice.offset();
        let dst_byte_offset = dst_slice.offset();
        let src_format = src_shape.format;
        let dst_format = dst_shape.format;
        let src_strides = src_shape.byte_stride();
        let dst_strides = dst_shape.byte_stride();
        let src_extent = src_shape.extent;
        let dst_extent = dst_shape.extent;

        assert!(
            cvt_compatible(src_format, dst_format),
            "Cvt: src format {:?} is not compatible with dst format {:?}",
            src_format,
            dst_format
        );

        let src_layout = describe(src_format).expect("Cvt: src format is not describable");
        let dst_layout = describe(dst_format).expect("Cvt: dst format is not describable");
        let src_align = component_byte_size(&src_layout);
        let dst_align = component_byte_size(&dst_layout);

        assert_eq!(
            src_byte_offset % src_align,
            0,
            "Cvt: src_byte_offset must be a multiple of {src_align} (src's component byte size)"
        );
        assert_eq!(
            dst_byte_offset % dst_align,
            0,
            "Cvt: dst_byte_offset must be a multiple of {dst_align} (dst's component byte size)"
        );
        for &s in &src_strides {
            assert_eq!(
                s % src_align,
                0,
                "Cvt: src strides must be a multiple of {src_align} (src's component byte size)"
            );
        }
        for &s in &dst_strides {
            assert_eq!(
                s % dst_align,
                0,
                "Cvt: dst strides must be a multiple of {dst_align} (dst's component byte size)"
            );
        }
        assert_eq!(
            dst_byte_offset % 4,
            0,
            "Cvt: dst_byte_offset must be a multiple of 4 (word-aligned) - required by the tiled non-atomic destination-write dispatch"
        );

        if !resample {
            Self::assert_exact_extents(axes, src_extent.0, dst_extent.0);
        }

        if address_mode.is_none() {
            assert_fits_buffer_slice(
                "src",
                &src_slice,
                &src_strides,
                &src_extent.0,
                src_layout.byte_size,
            );
        }
        assert_fits_buffer_slice(
            "dst",
            &dst_slice,
            &dst_strides,
            &dst_extent.0,
            dst_layout.byte_size,
        );

        let dst_tight_len =
            classify_dst_tight_chain(&dst_extent.0, &dst_strides, dst_layout.byte_size);

        let wrap = !resample && address_mode.is_some();

        let total_dst_elements: u32 = dst_extent.0.iter().product();

        let src_endpoint = EndpointDesc {
            kind: EndpointKind::Buffer,
            format: src_format,
        };
        let dst_endpoint = EndpointDesc {
            kind: EndpointKind::Buffer,
            format: dst_format,
        };

        let key = CvtComputeKey {
            src: src_endpoint,
            dst: dst_endpoint,
            swizzle,
            src_rank: SRC as u8,
            dst_rank: DST as u8,
            axes,
            resample,
            wrap,
            dst_tight_len: dst_tight_len as u8,
        };

        let pipeline = {
            let mut pipelines = self.compute_pipelines.lock();
            pipelines
                .entry(key)
                .or_insert_with(|| {
                    make_compute_pipeline_nd(
                        &self.device,
                        src_endpoint,
                        dst_endpoint,
                        swizzle,
                        SRC,
                        DST,
                        axes,
                        resample,
                        wrap,
                        dst_tight_len,
                    )
                })
                .clone()
        };

        let mut dst_stride4 = [0u32; 4];
        for d in 0..DST {
            dst_stride4[d] = dst_strides[d] as u32;
        }
        let mut src_stride4 = [0u32; 4];
        for s in 0..SRC {
            src_stride4[s] = src_strides[s] as u32;
        }
        let mut src_extent4 = [0.0f32; 4];
        for s in 0..SRC {
            src_extent4[s] = src_extent.0[s] as f32;
        }

        let constants = CvtConstantsBufferToBuffer {
            dst_extent: pad4_u32(dst_extent.0),
            dst_byte_offset: dst_byte_offset as u32,
            dst_stride: dst_stride4,
            src_byte_offset: src_byte_offset as u32,
            src_stride: src_stride4,
            total_dst_elements,
            src_extent: src_extent4,
            address_mode: pad4_addr_mode(address_mode.unwrap_or([AddressMode::ClampToEdge; SRC])),
        };

        let groups = dst_tile_dispatch_extent(DstTileUnits::new(DST, dst_tight_len), &dst_extent.0);

        let mut compute = encoder.compute();
        compute.with_pipeline(&pipeline);
        compute.with_constants(&constants);
        compute.with_arguments(
            0,
            &CvtNdArguments {
                src: src_slice.buffer().clone(),
                dst: dst_slice.buffer().clone(),
            },
        );
        compute.dispatch(groups);
    }

    /// Converts a buffer's contents in place: reads each element through
    /// `src_format`, converts, and writes it back through `dst_format` at
    /// the same element position, starting at byte offset `0`.
    ///
    /// Buffer-only (there is no `cvt_image_in_place`); requires
    /// `describe(src_format).byte_size == describe(dst_format).byte_size` (asserted).
    ///
    /// Uses a dedicated pipeline binding exactly one storage resource
    /// - not a two-`Buffer` alias of the two-resource copy path.
    ///
    /// Takes `src_format`/`dst_format`/`swizzle` directly rather than a `CvtDesc`:
    /// there is no axis map here (a single buffer, same element position
    /// in and out).
    pub fn buffer_in_place(
        &self,
        encoder: &mut CommandEncoder,
        buffer: &Buffer,
        src_format: PixelFormat,
        dst_format: PixelFormat,
        element_count: u32,
        swizzle: Swizzle,
    ) {
        assert!(
            cvt_compatible(src_format, dst_format),
            "Cvt: src format {:?} is not compatible with dst format {:?}",
            src_format,
            dst_format
        );

        let src_layout = describe(src_format).expect("Cvt: src format is not describable");
        let dst_layout = describe(dst_format).expect("Cvt: dst format is not describable");
        assert_eq!(
            src_layout.byte_size, dst_layout.byte_size,
            "Cvt: in-place conversion requires src/dst element byte size to match"
        );

        let key = CvtInPlaceKey {
            src_format,
            dst_format,
            swizzle,
        };

        let pipeline = {
            let mut pipelines = self.in_place_pipelines.lock();
            pipelines
                .entry(key)
                .or_insert_with(|| {
                    make_in_place_pipeline(&self.device, src_format, dst_format, swizzle)
                })
                .clone()
        };

        let constants = CvtInPlaceConstants {
            byte_offset: 0,
            element_count,
        };

        // 4 elements/thread (see `make_in_place_pipeline`'s tiled scheme),
        // grouped further into workgroups of 64.
        let groups = Extent3::new(element_count.div_ceil(4).div_ceil(64), 1, 1);

        let mut compute = encoder.compute();
        compute.with_pipeline(&pipeline);
        compute.with_constants(&constants);
        compute.with_arguments(
            0,
            &CvtInPlaceArguments {
                buf: buffer.clone(),
            },
        );
        compute.dispatch(groups);
    }

    /// Converts `src` into `dst`, both `Image`s of the same rank `N`,
    /// with default options: exact copy, identity axis map and swizzle,
    /// out-of-bounds reads denied (src/dst pixel formats may still differ
    /// - this is a format conversion, not a byte copy).
    ///
    /// Use [`Cvt::image_to_image_with`] for anything else.
    pub fn image_to_image<const N: usize>(
        &self,
        encoder: &mut CommandEncoder,
        src: impl AsImageSlice<N>,
        dst: impl AsImageSlice<N>,
    ) {
        self.image_to_image_with::<N, N>(encoder, src, dst, CvtOptions::default());
    }

    /// Converts a rectangular region of `src` into a rectangular region of `dst`,
    /// both `Image`s, up to 4 dimensions
    /// (`2 <= SRC <= 4` and `2 <= DST <= 4` - `Image` endpoints
    /// are always sampled/stored in at least X/Y,
    /// so unlike `buffer_to_buffer_with` there is a rank floor of 2),
    /// per `opts` (see [`CvtOptions`], including the values accepted directly via `Into`).
    ///
    /// With `opts.resize` unset this is an exact-index copy:
    /// extents must match along every mapped axis.
    ///
    /// With `opts.resize` set, the src/dst extents may differ along mapped axes:
    /// `dst`'s content is read from `src` by scaling each destination
    /// axis's position ratio across `src_extent`
    /// (see [`gen_endpoint_decode_convert`] for the exact per-axis formula).
    ///
    /// When `src`'s pixel format is float-representable (Unorm/Snorm/Float),
    /// the resampled source is hardware sampled through a real `Sampler`
    /// built from the resize filter and address mode;
    /// when it is `Uint`/`Sint`, there is no hardware sampler
    /// for integer texture formats, so it is always nearest-index
    /// with manual address-mode wrapping - requesting `Filter::Linear`
    /// for an integer-kind `src` panics rather than silently falling back to nearest.
    ///
    /// With `opts.address_mode` unset, the source region must fit its image
    /// (asserted; where the resample path needs a mode value anyway,
    /// `ClampToEdge` is used).
    ///
    /// When set, out-of-range source reads wrap/clamp/mirror per axis
    /// against the image's real physical extent instead.
    ///
    /// The destination region must fit its image unconditionally
    /// - address mode never redirects writes.
    ///
    /// Panics (before recording any GPU work) if `SRC`/`DST` are outside
    /// `2..=4`, the formats are not `compatible`, a region fails its fit
    /// check as above, extents mismatch on a mapped axis without `resize`,
    /// or `Filter::Linear` is requested for an integer-kind `src`.
    ///
    /// Routes through a render pipeline (fullscreen triangle) when `dst`'s
    /// format is sRGB (`textureStore` cannot target an sRGB WGSL storage
    /// format; hardware gamma-encodes on write to the sRGB color
    /// attachment), a compute pipeline otherwise - mirroring `Blit`.
    pub fn image_to_image_with<const SRC: usize, const DST: usize>(
        &self,
        encoder: &mut CommandEncoder,
        src: impl AsImageSlice<SRC>,
        dst: impl AsImageSlice<DST>,
        opts: impl Into<CvtOptions<SRC>>,
    ) {
        let CvtOptions {
            axes,
            swizzle,
            resize,
            address_mode,
        } = opts.into();
        let resample = resize.is_some();

        assert!(
            (2..=4).contains(&SRC),
            "Cvt: image_to_image requires 2 <= SRC <= 4 (Image endpoints need at least X/Y)"
        );
        assert!(
            (2..=4).contains(&DST),
            "Cvt: image_to_image requires 2 <= DST <= 4 (Image endpoints need at least X/Y)"
        );

        let src = src.as_image_slice();
        if address_mode.is_none() {
            src.assert_fits_image();
        }
        let (src_view, src_offset, src_extent) = resolve_image_slice(&self.device, &src);

        let dst = dst.as_image_slice();
        let (dst_view, dst_offset, dst_extent) = resolve_image_slice(&self.device, &dst);

        let src_format = src_view.format();
        let dst_format = dst_view.format();
        assert!(
            cvt_compatible(src_format, dst_format),
            "Cvt: src format {:?} is not compatible with dst format {:?}",
            src_format,
            dst_format
        );

        let src_layout = describe(src_format).expect("Cvt: src format is not describable");
        let dst_layout = describe(dst_format).expect("Cvt: dst format is not describable");

        match resize {
            None => Self::assert_exact_extents(axes, src_extent.0, dst_extent.0),
            Some((min, mag)) => assert!(
                src_layout.kind == CanonicalKind::Float
                    || (min == Filter::Nearest && mag == Filter::Nearest),
                "Cvt: only Filter::Nearest is supported for integer-kind (Uint/Sint) image src pixel formats"
            ),
        }

        let wrap = !resample && address_mode.is_some();
        let address_mode = address_mode.unwrap_or([AddressMode::ClampToEdge; SRC]);

        let src_endpoint = EndpointDesc {
            kind: EndpointKind::Image,
            format: src_format,
        };
        let dst_endpoint = EndpointDesc {
            kind: EndpointKind::Image,
            format: dst_format,
        };

        let key = CvtKey {
            src: src_endpoint,
            dst: dst_endpoint,
            swizzle,
            src_rank: SRC as u8,
            dst_rank: DST as u8,
            axes,
            resample,
            wrap,
        };

        let src_image_size = src_view.extent().into_2d();
        let dst_image_size = dst_view.extent().into_2d();
        let total_dst_elements: u32 = dst_extent.0.iter().product();

        let constants = CvtConstantsImageToImage {
            dst_extent: pad4_u32(dst_extent.0),
            dst_offset: pad4_i32(dst_offset.0),
            dst_image_size: [dst_image_size.width(), dst_image_size.height()],
            src_offset: pad4_i32(src_offset.0),
            src_image_size: [src_image_size.width(), src_image_size.height()],
            total_dst_elements,
            src_extent: pad4_f32(src_extent.0.map(|v| v as f32)),
            address_mode: pad4_addr_mode(address_mode),
        };

        if cvt_use_render_pipeline(&dst_layout, dst_format, dst_view.usage()) {
            let pipeline = {
                let mut pipelines = self.render_pipelines.lock();
                pipelines
                    .entry(key)
                    .or_insert_with(|| {
                        make_render_pipeline(
                            &self.device,
                            src_endpoint,
                            dst_format,
                            swizzle,
                            axes,
                            SRC,
                            resample,
                            wrap,
                        )
                    })
                    .clone()
            };

            let viewport_offset = Offset3::new(
                dst_offset.0[0].min(dst_image_size.width() as i32) as f32,
                dst_offset.0[1].min(dst_image_size.height() as i32) as f32,
                0.0,
            );
            let viewport_extent = Extent3::new(
                dst_extent.0[0].min(dst_image_size.width()) as f32,
                dst_extent.0[1].min(dst_image_size.height()) as f32,
                1.0,
            );

            let vx0 = dst_offset.0[0].max(0);
            let vy0 = dst_offset.0[1].max(0);
            let vx1 = (dst_offset.0[0] + dst_extent.0[0] as i32).min(dst_image_size.width() as i32);
            let vy1 =
                (dst_offset.0[1] + dst_extent.0[1] as i32).min(dst_image_size.height() as i32);

            if vx1 <= vx0 || vy1 <= vy0 {
                return;
            }

            let mut render = encoder.render(RenderPassDesc {
                name: "cvt_srgb",
                color_attachments: &[AttachmentDesc::new(&dst_view)],
                depth_stencil_attachment: None,
            });

            render.with_pipeline(&pipeline);
            render.with_constants(&constants);
            render.with_viewport(viewport_offset, viewport_extent);
            render.with_scissor(
                Offset2::new(vx0, vy0),
                Extent2::new((vx1 - vx0) as u32, (vy1 - vy0) as u32),
            );
            if resample && src_layout.kind == CanonicalKind::Float {
                let (min_filter, mag_filter) = resize.expect("resample implies resize.is_some()");
                let sampler_desc = SamplerDesc {
                    min_filter,
                    mag_filter,
                    mip_map_mode: MipMapMode::Nearest,
                    address_mode: sampler_addr_mode3(address_mode),
                    anisotropy: None,
                    min_lod: 0.0,
                    max_lod: 0.0,
                    normalized: true,
                };
                let sampler = self.sampler_for(sampler_desc);
                render.with_arguments(
                    0,
                    &CvtImageSrcRenderResampleArguments {
                        src: src_view.clone(),
                        sampler,
                    },
                );
            } else {
                render.with_arguments(
                    0,
                    &CvtImageSrcRenderArguments {
                        src: src_view.clone(),
                    },
                );
            }
            render.draw(0..3, 0..1);
        } else {
            // `dst` is `Image` here, so the tiled non-atomic scheme doesn't
            // apply (`textureStore` never aliases across threads) -
            // `dst_tight_len` is meaningless, always `0`.
            let compute_key = CvtComputeKey {
                src: src_endpoint,
                dst: dst_endpoint,
                swizzle,
                src_rank: SRC as u8,
                dst_rank: DST as u8,
                axes,
                resample,
                wrap,
                dst_tight_len: 0,
            };
            let pipeline = {
                let mut pipelines = self.compute_pipelines.lock();
                pipelines
                    .entry(compute_key)
                    .or_insert_with(|| {
                        make_compute_pipeline_nd(
                            &self.device,
                            src_endpoint,
                            dst_endpoint,
                            swizzle,
                            SRC,
                            DST,
                            axes,
                            resample,
                            wrap,
                            0,
                        )
                    })
                    .clone()
            };

            let groups = Extent3::new(total_dst_elements.div_ceil(64), 1, 1);

            let mut compute = encoder.compute();
            compute.with_pipeline(&pipeline);
            compute.with_constants(&constants);

            if resample && src_layout.kind == CanonicalKind::Float {
                let (min_filter, mag_filter) = resize.expect("resample implies resize.is_some()");
                let sampler_desc = SamplerDesc {
                    min_filter,
                    mag_filter,
                    mip_map_mode: MipMapMode::Nearest,
                    address_mode: sampler_addr_mode3(address_mode),
                    anisotropy: None,
                    min_lod: 0.0,
                    max_lod: 0.0,
                    normalized: true,
                };
                let sampler = self.sampler_for(sampler_desc);
                compute.with_arguments(
                    0,
                    &CvtImageToImageResampleArguments {
                        src: src_view.clone(),
                        sampler,
                        dst: dst_view.clone(),
                    },
                );
            } else {
                compute.with_arguments(
                    0,
                    &CvtImageToImageArguments {
                        src: src_view.clone(),
                        dst: dst_view.clone(),
                    },
                );
            }
            compute.dispatch(groups);
        }
    }

    /// Converts `src` into `dst`, `src` an `Image` and `dst` a `Buffer`
    /// of matching rank `N`, with default options: exact copy,
    /// identity axis map and swizzle, out-of-bounds source reads denied.
    ///
    /// Use [`Cvt::image_to_buffer_with`] for anything else.
    pub fn image_to_buffer<const N: usize>(
        &self,
        encoder: &mut CommandEncoder,
        src: impl AsImageSlice<N>,
        dst: impl AsBufferSlice,
        dst_shape: TexelShape<N>,
    ) {
        self.image_to_buffer_with::<N, N>(encoder, src, dst, dst_shape, CvtOptions::default());
    }

    /// Converts a rectangular region of `src` (an `Image`) into `dst` (a `Buffer`),
    /// described by `dst_format` with per-axis byte strides `dst_strides`
    /// (`dst_strides[0]` is the X-axis stride - the byte offset
    /// from one element to the next; it need not equal
    /// the format's tightly-packed element size), per `opts`
    /// (see [`CvtOptions`], including the values accepted directly via `Into`).
    ///
    /// See [`Cvt::image_to_image_with`]'s docs for the `resize`/`address_mode`
    /// semantics (hardware-sampled resample for a float-representable `src`,
    /// manual nearest-index address-mode wrapping for `Uint`/`Sint`,
    /// `dst` fit always asserted, `src` fit asserted unless `address_mode` is set)
    /// - they apply identically here, keyed off `src`'s (the `Image`) pixel format.
    ///
    /// There is no sRGB-destination restriction here: `dst` is a `Buffer`,
    /// so there is no hardware gamma-encode-on-write path
    /// to route through in the first place.
    ///
    /// Panics (before recording any GPU work) under the same conditions as
    /// `image_to_image_with` (formats not `compatible`, `Filter::Linear`
    /// requested for an integer-kind `src`, a region fails its fit check),
    /// plus misaligned byte offset/strides (see `buffer_to_buffer_with`).
    pub fn image_to_buffer_with<const SRC: usize, const DST: usize>(
        &self,
        encoder: &mut CommandEncoder,
        src: impl AsImageSlice<SRC>,
        dst: impl AsBufferSlice,
        dst_shape: TexelShape<DST>,
        opts: impl Into<CvtOptions<SRC>>,
    ) {
        let CvtOptions {
            axes,
            swizzle,
            resize,
            address_mode,
        } = opts.into();
        let resample = resize.is_some();

        assert!(
            (2..=4).contains(&SRC),
            "Cvt: image_to_buffer requires 2 <= SRC <= 4 (Image endpoints need at least X/Y)"
        );
        assert!(
            DST <= 4,
            "Cvt: image<->buffer destination rank must be <= 4"
        );

        let src = src.as_image_slice();
        if address_mode.is_none() {
            src.assert_fits_image();
        }
        let (src_view, src_offset, src_extent) = resolve_image_slice(&self.device, &src);

        let dst_slice = dst.as_buffer_slice();

        let dst_byte_offset = dst_slice.offset();
        let dst_format = dst_shape.format;
        let dst_strides = dst_shape.byte_stride();
        let dst_extent = dst_shape.extent;

        let src_format = src_view.format();
        assert!(
            cvt_compatible(src_format, dst_format),
            "Cvt: src format {:?} is not compatible with dst format {:?}",
            src_format,
            dst_format
        );

        let src_layout = describe(src_format).expect("Cvt: src format is not describable");
        let dst_layout = describe(dst_format).expect("Cvt: dst format is not describable");

        match resize {
            None => Self::assert_exact_extents(axes, src_extent.0, dst_extent.0),
            Some((min, mag)) => assert!(
                src_layout.kind == CanonicalKind::Float
                    || (min == Filter::Nearest && mag == Filter::Nearest),
                "Cvt: only Filter::Nearest is supported for integer-kind (Uint/Sint) image src pixel formats"
            ),
        }

        let align = component_byte_size(&dst_layout);
        assert_eq!(
            dst_byte_offset % align,
            0,
            "Cvt: dst_byte_offset must be a multiple of {align} (dst's component byte size)"
        );
        for &s in &dst_strides {
            assert_eq!(
                s % align,
                0,
                "Cvt: dst strides must be a multiple of {align} (dst's component byte size)"
            );
        }
        assert_eq!(
            dst_byte_offset % 4,
            0,
            "Cvt: dst_byte_offset must be a multiple of 4 (word-aligned) - required by the tiled non-atomic destination-write dispatch"
        );

        assert_fits_buffer_slice(
            "dst",
            &dst_slice,
            &dst_strides,
            &dst_extent.0,
            dst_layout.byte_size,
        );

        let dst_tight_len =
            classify_dst_tight_chain(&dst_extent.0, &dst_strides, dst_layout.byte_size);

        let wrap = !resample && address_mode.is_some();
        let address_mode = address_mode.unwrap_or([AddressMode::ClampToEdge; SRC]);

        let src_endpoint = EndpointDesc {
            kind: EndpointKind::Image,
            format: src_format,
        };
        let dst_endpoint = EndpointDesc {
            kind: EndpointKind::Buffer,
            format: dst_format,
        };

        let key = CvtComputeKey {
            src: src_endpoint,
            dst: dst_endpoint,
            swizzle,
            src_rank: SRC as u8,
            dst_rank: DST as u8,
            axes,
            resample,
            wrap,
            dst_tight_len: dst_tight_len as u8,
        };

        let src_image_size = src_view.extent().into_2d();
        let total_dst_elements: u32 = dst_extent.0.iter().product();

        let mut dst_stride4 = [0u32; 4];
        for d in 0..DST {
            dst_stride4[d] = dst_strides[d] as u32;
        }

        let constants = CvtConstantsImageToBuffer {
            dst_extent: pad4_u32(dst_extent.0),
            dst_byte_offset: dst_byte_offset as u32,
            dst_stride: dst_stride4,
            src_offset: pad4_i32(src_offset.0),
            src_image_size: [src_image_size.width(), src_image_size.height()],
            total_dst_elements,
            src_extent: pad4_f32(src_extent.0.map(|v| v as f32)),
            address_mode: pad4_addr_mode(address_mode),
        };

        let pipeline = {
            let mut pipelines = self.compute_pipelines.lock();
            pipelines
                .entry(key)
                .or_insert_with(|| {
                    make_compute_pipeline_nd(
                        &self.device,
                        src_endpoint,
                        dst_endpoint,
                        swizzle,
                        SRC,
                        DST,
                        axes,
                        resample,
                        wrap,
                        dst_tight_len,
                    )
                })
                .clone()
        };

        let groups = dst_tile_dispatch_extent(DstTileUnits::new(DST, dst_tight_len), &dst_extent.0);

        let mut compute = encoder.compute();
        compute.with_pipeline(&pipeline);
        compute.with_constants(&constants);

        if resample && src_layout.kind == CanonicalKind::Float {
            let (min_filter, mag_filter) = resize.expect("resample implies resize.is_some()");
            let sampler_desc = SamplerDesc {
                min_filter,
                mag_filter,
                mip_map_mode: MipMapMode::Nearest,
                address_mode: sampler_addr_mode3(address_mode),
                anisotropy: None,
                min_lod: 0.0,
                max_lod: 0.0,
                normalized: true,
            };
            let sampler = self.sampler_for(sampler_desc);
            compute.with_arguments(
                0,
                &CvtImageToBufferResampleArguments {
                    src: src_view.clone(),
                    sampler,
                    dst: dst_slice.buffer().clone(),
                },
            );
        } else {
            compute.with_arguments(
                0,
                &CvtImageToBufferArguments {
                    src: src_view.clone(),
                    dst: dst_slice.buffer().clone(),
                },
            );
        }
        compute.dispatch(groups);
    }

    /// Converts `src` into `dst`, `src` a `Buffer` and `dst` an `Image`
    /// of matching rank `N`, with default options: exact copy,
    /// identity axis map and swizzle, out-of-bounds source reads denied.
    ///
    /// Use [`Cvt::buffer_to_image_with`] for anything else.
    pub fn buffer_to_image<const N: usize>(
        &self,
        encoder: &mut CommandEncoder,
        src: impl AsBufferSlice,
        src_shape: TexelShape<N>,
        dst: impl AsImageSlice<N>,
    ) {
        self.buffer_to_image_with::<N, N>(encoder, src, src_shape, dst, CvtOptions::default());
    }

    /// Converts `src` (a `Buffer`, described by `src_format` with per-axis
    /// byte strides `src_strides` - `src_strides[0]` is the X-axis stride,
    /// the byte offset from one element to the next; it need not equal
    /// the format's tightly-packed element size) into a rectangular region
    /// of `dst` (an `Image`), per `opts`
    /// (see [`CvtOptions`], including the values accepted directly via `Into`).
    ///
    /// Mirror of [`Cvt::buffer_to_buffer_with`]; see its docs
    /// for the `resize`/`address_mode` semantics.
    ///
    /// Routes through the sRGB render pipeline when `dst`'s format is sRGB.
    ///
    /// Panics (before recording any GPU work) under the same conditions as
    /// `buffer_to_buffer_with` (formats not `compatible`, a region fails its
    /// fit check, misaligned byte offset/strides).
    pub fn buffer_to_image_with<const SRC: usize, const DST: usize>(
        &self,
        encoder: &mut CommandEncoder,
        src: impl AsBufferSlice,
        src_shape: TexelShape<SRC>,
        dst: impl AsImageSlice<DST>,
        opts: impl Into<CvtOptions<SRC>>,
    ) {
        #![allow(clippy::needless_range_loop)]

        let CvtOptions {
            axes,
            swizzle,
            resize,
            address_mode,
        } = opts.into();

        if let Some((min, mag)) = resize {
            assert!(
                min == Filter::Nearest && mag == Filter::Nearest,
                "Cvt: buffer source has no sampler, filter must be Filter::Nearest"
            );
        }
        let resample = resize.is_some();

        assert!(SRC <= 4, "Cvt: buffer<->image source rank must be <= 4");
        assert!(
            (2..=4).contains(&DST),
            "Cvt: buffer_to_image requires 2 <= DST <= 4 (Image endpoints need at least X/Y)"
        );

        let dst = dst.as_image_slice();
        let (dst_view, dst_offset, dst_extent) = resolve_image_slice(&self.device, &dst);

        let src_slice = src.as_buffer_slice();

        let src_byte_offset = src_slice.offset();
        let src_format = src_shape.format;
        let src_strides = src_shape.byte_stride();
        let src_extent = src_shape.extent;

        let dst_format = dst_view.format();
        assert!(
            cvt_compatible(src_format, dst_format),
            "Cvt: src format {:?} is not compatible with dst format {:?}",
            src_format,
            dst_format
        );

        if !resample {
            Self::assert_exact_extents(axes, src_extent.0, dst_extent.0);
        }

        let src_layout = describe(src_format).expect("Cvt: src format is not describable");
        let dst_layout = describe(dst_format).expect("Cvt: dst format is not describable");
        let align = component_byte_size(&src_layout);
        assert_eq!(
            src_byte_offset % align,
            0,
            "Cvt: src_byte_offset must be a multiple of {align} (src's component byte size)"
        );
        for &s in &src_strides {
            assert_eq!(
                s % align,
                0,
                "Cvt: src strides must be a multiple of {align} (src's component byte size)"
            );
        }

        if address_mode.is_none() {
            assert_fits_buffer_slice(
                "src",
                &src_slice,
                &src_strides,
                &src_extent.0,
                src_layout.byte_size,
            );
        }

        let wrap = !resample && address_mode.is_some();
        let address_mode = address_mode.unwrap_or([AddressMode::ClampToEdge; SRC]);

        let src_endpoint = EndpointDesc {
            kind: EndpointKind::Buffer,
            format: src_format,
        };
        let dst_endpoint = EndpointDesc {
            kind: EndpointKind::Image,
            format: dst_format,
        };

        let key = CvtKey {
            src: src_endpoint,
            dst: dst_endpoint,
            swizzle,
            src_rank: SRC as u8,
            dst_rank: DST as u8,
            axes,
            resample,
            wrap,
        };

        let dst_image_size = dst_view.extent().into_2d();
        let total_dst_elements: u32 = dst_extent.0.iter().product();

        let mut src_stride4 = [0u32; 4];
        for s in 0..SRC {
            src_stride4[s] = src_strides[s] as u32;
        }
        let mut src_extent4 = [0.0f32; 4];
        for s in 0..SRC {
            src_extent4[s] = src_extent.0[s] as f32;
        }

        let constants = CvtConstantsBufferToImage {
            dst_extent: pad4_u32(dst_extent.0),
            dst_offset: pad4_i32(dst_offset.0),
            dst_image_size: [dst_image_size.width(), dst_image_size.height()],
            src_byte_offset: src_byte_offset as u32,
            src_stride: src_stride4,
            total_dst_elements,
            src_extent: src_extent4,
            address_mode: pad4_addr_mode(address_mode),
        };

        if cvt_use_render_pipeline(&dst_layout, dst_format, dst_view.usage()) {
            let pipeline = {
                let mut pipelines = self.render_pipelines.lock();
                pipelines
                    .entry(key)
                    .or_insert_with(|| {
                        make_render_pipeline(
                            &self.device,
                            src_endpoint,
                            dst_format,
                            swizzle,
                            axes,
                            SRC,
                            resample,
                            wrap,
                        )
                    })
                    .clone()
            };

            let viewport_offset = Offset3::new(
                dst_offset.0[0].min(dst_image_size.width() as i32) as f32,
                dst_offset.0[1].min(dst_image_size.height() as i32) as f32,
                0.0,
            );
            let viewport_extent = Extent3::new(
                dst_extent.0[0].min(dst_image_size.width()) as f32,
                dst_extent.0[1].min(dst_image_size.height()) as f32,
                1.0,
            );

            let vx0 = dst_offset.0[0].max(0);
            let vy0 = dst_offset.0[1].max(0);
            let vx1 = (dst_offset.0[0] + dst_extent.0[0] as i32).min(dst_image_size.width() as i32);
            let vy1 =
                (dst_offset.0[1] + dst_extent.0[1] as i32).min(dst_image_size.height() as i32);

            if vx1 <= vx0 || vy1 <= vy0 {
                return;
            }

            let mut render = encoder.render(RenderPassDesc {
                name: "cvt_srgb",
                color_attachments: &[AttachmentDesc::new(&dst_view)],
                depth_stencil_attachment: None,
            });

            render.with_pipeline(&pipeline);
            render.with_constants(&constants);
            render.with_viewport(viewport_offset, viewport_extent);
            render.with_scissor(
                Offset2::new(vx0, vy0),
                Extent2::new((vx1 - vx0) as u32, (vy1 - vy0) as u32),
            );
            render.with_arguments(
                0,
                &CvtBufferSrcRenderArguments {
                    src: src_slice.buffer().clone(),
                },
            );
            render.draw(0..3, 0..1);
        } else {
            // `dst` is `Image` here, so the tiled non-atomic scheme doesn't
            // apply - `dst_tight_len` is meaningless, always `0`.
            let compute_key = CvtComputeKey {
                src: src_endpoint,
                dst: dst_endpoint,
                swizzle,
                src_rank: SRC as u8,
                dst_rank: DST as u8,
                axes,
                resample,
                wrap,
                dst_tight_len: 0,
            };
            let pipeline = {
                let mut pipelines = self.compute_pipelines.lock();
                pipelines
                    .entry(compute_key)
                    .or_insert_with(|| {
                        make_compute_pipeline_nd(
                            &self.device,
                            src_endpoint,
                            dst_endpoint,
                            swizzle,
                            SRC,
                            DST,
                            axes,
                            resample,
                            wrap,
                            0,
                        )
                    })
                    .clone()
            };

            let groups = Extent3::new(total_dst_elements.div_ceil(64), 1, 1);

            let mut compute = encoder.compute();
            compute.with_pipeline(&pipeline);
            compute.with_constants(&constants);
            compute.with_arguments(
                0,
                &CvtBufferToImageArguments {
                    src: src_slice.buffer().clone(),
                    dst: dst_view.clone(),
                },
            );
            compute.dispatch(groups);
        }
    }
}

// ============================================================================
// Tests: pure codegen/classification logic (no GPU device needed).
//
// The tiled dispatch's actual read-back correctness is verified
// on real hardware by `examples/cvt_smoke.rs`.
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_dst_tight_chain_cases() {
        // Case 1: "all aligned to u32" - a dense, word-sized dst (e.g.
        // Rgba8Unorm, byte_size=4) is naturally word-aligned at every
        // level, so `aligned` (checked first) wins over the numerically
        // tight equality: T == 0, not the whole rank.
        let t = classify_dst_tight_chain(&[4, 3], &[4, 16], 4);
        assert_eq!(
            t, 0,
            "dense word-sized dst should classify axis 0 as aligned, not tight"
        );

        // Case 2: rows tight (byte_size=1, no gap), planes aligned with an
        // explicit padding gap - the scenario the tight/aligned distinction
        // exists for.
        let t = classify_dst_tight_chain(&[3, 2], &[1, 4], 1);
        assert_eq!(
            t, 1,
            "byte_size==1 packed row + word-aligned padded plane stride should give T=1"
        );

        // Case 4: fully tight (no padding anywhere, sub-word format).
        let t = classify_dst_tight_chain(&[3, 2], &[1, 3], 1);
        assert_eq!(
            t, 2,
            "fully packed sub-word dst (no gaps) should give T == rank"
        );

        // General no-aliasing precondition, axis 0: `stride == 0` must be
        // rejected (this is the real gap in the old
        // `stride % component_byte_size == 0` check the plan calls out -
        // `0 % x == 0` let it slip through).
        let result = std::panic::catch_unwind(|| classify_dst_tight_chain(&[3], &[0], 1));
        assert!(result.is_err(), "dst_stride[0] < byte_size must panic");

        // General no-aliasing precondition, axis >= 1.
        let result = std::panic::catch_unwind(|| classify_dst_tight_chain(&[3, 2], &[1, 2], 1));
        assert!(
            result.is_err(),
            "dst_stride[1] < extent[0]*stride[0] must panic"
        );

        // "Bad" axis: neither word-aligned nor tight relative to its
        // predecessor - must panic with a clear message, not silently
        // produce wrong output.
        let result = std::panic::catch_unwind(|| classify_dst_tight_chain(&[3, 2], &[1, 5], 1));
        let err = result.expect_err("a non-word-aligned, non-tight stride must panic");
        let msg = err
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        assert!(
            msg.contains("neither word-aligned") && msg.contains("unsupported"),
            "panic message should clearly explain the unsupported alignment, got: {msg:?}"
        );
    }

    /// A 2D padded-row destination (`T=1`: tight rows, aligned-with-a-gap
    /// planes - the scenario the tight/aligned distinction exists for)
    /// generates the expected dispatch shape: the tight chain rides
    /// `gid.x`, the one remaining aligned axis rides `gid.y` independently
    /// (no wrap between them), and no `gid.z` unit is needed at rank 2.
    #[test]
    fn dst_buffer_tiled_body_2d_padded_row_shape() {
        let src = EndpointDesc {
            kind: EndpointKind::Buffer,
            format: PixelFormat::R8Uint,
        };
        let dst_layout = describe(PixelFormat::R8Uint).unwrap();
        let src_layout = describe(PixelFormat::R8Uint).unwrap();

        let units = DstTileUnits::new(2, 1);
        let body = gen_dst_buffer_tiled_body(
            src,
            &dst_layout,
            &src_layout,
            Swizzle::IDENTITY,
            AxisSwizzle::IDENTITY,
            2,
            2,
            false,
            false,
            units,
        );

        println!("{body}");

        assert!(body.contains("gid.x"), "tight chain unit should ride gid.x");
        assert!(
            body.contains("gid.y"),
            "the one remaining aligned axis should ride gid.y"
        );
        assert!(
            !body.contains("gid.z"),
            "no third dispatch dimension is needed for rank 2"
        );
        assert!(
            body.contains("dst_c1_v1_"),
            "axis 1 should be handled as its own independent aligned unit"
        );
        assert!(
            body.contains("dst_c0_v0_"),
            "axis 0 should be handled by the tight-chain (x) unit"
        );
    }

    /// A rank-4 destination with no tight chain at all (`T == 0`) needs the
    /// full `gid.x`/`gid.y`/`gid.z`-folded shape (3 remaining aligned axes
    /// after axis 0 claims `gid.x`, so axes 2 and 3 fold into `gid.z`).
    #[test]
    fn dst_buffer_tiled_body_rank4_all_aligned_folds_into_z() {
        let src = EndpointDesc {
            kind: EndpointKind::Buffer,
            format: PixelFormat::Rgba8Unorm,
        };
        let dst_layout = describe(PixelFormat::Rgba8Unorm).unwrap();
        let src_layout = describe(PixelFormat::Rgba8Unorm).unwrap();

        let units = DstTileUnits::new(4, 0);
        assert_eq!(units.z_unit(), ZUnit::Folded(2, 3));

        let body = gen_dst_buffer_tiled_body(
            src,
            &dst_layout,
            &src_layout,
            Swizzle::IDENTITY,
            AxisSwizzle::IDENTITY,
            4,
            4,
            false,
            false,
            units,
        );

        assert!(body.contains("group_zb"), "axis 2 should fold into gid.z");
        assert!(body.contains("group_zc"), "axis 3 should fold into gid.z");
        assert!(
            body.contains("dst_c0_v0_"),
            "axis 0 alone should ride gid.x (T == 0, not a chain)"
        );
        assert!(
            body.contains("dst_c1_v1_"),
            "axis 1 should ride gid.y directly"
        );
    }
}
