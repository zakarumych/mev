use std::hash::{Hash, Hasher};

/// Filter to use when sampling the texture.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Filter {
    /// Sample from nearest texel.
    #[default]
    Nearest,

    /// Sample by linear interpolation of the four nearest texels.
    Linear,
}

/// Mip-map mode to use when sampling the texture.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MipMapMode {
    /// Sample from nearest mip-map level.
    #[default]
    Nearest,

    /// Sample by linear interpolation of the two nearest mip-map levels.
    Linear,
}

/// Address mode to use when sampling the texture.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AddressMode {
    /// Repeat the texture.
    #[default]
    Repeat,

    /// Repeat the texture with mirroring.
    MirrorRepeat,

    /// Sample closes edge texel.
    ClampToEdge,
}

/// Describes how to sample the texture.
#[derive(Clone, Copy, Debug)]
pub struct SamplerDesc {
    /// Filter to use when sampling the texture with pixels smaller than fragment.
    pub min_filter: Filter,

    /// Filter to use when sampling the texture with pixels larger than fragment.
    pub mag_filter: Filter,

    /// Mip-map mode to use when sampling the texture.
    pub mip_map_mode: MipMapMode,

    /// Address mode to use when sampling the texture, for each dimension.
    pub address_mode: [AddressMode; 3],

    /// Maximum anisotropy level to use when sampling the texture.
    pub anisotropy: Option<f32>,

    /// Minimum level of detail to use when sampling the texture.
    pub min_lod: f32,

    /// Maximum level of detail to use when sampling the texture.
    pub max_lod: f32,

    /// Whether to normalize the texture coordinates.
    /// If true, 0.0 and 1.0 are treated as edges of the texture.
    /// Otherwise 1.0 is size of one texel.
    pub normalized: bool,
}

impl PartialEq for SamplerDesc {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn eq(&self, other: &Self) -> bool {
        self.min_filter == other.min_filter
            && self.mag_filter == other.mag_filter
            && self.mip_map_mode == other.mip_map_mode
            && self.address_mode == other.address_mode
            && match (self.anisotropy, other.anisotropy) {
                (Some(a), Some(b)) => f32::total_cmp(&a, &b).is_eq(),
                (None, None) => true,
                _ => false,
            }
            && f32::total_cmp(&self.min_lod, &other.min_lod).is_eq()
            && f32::total_cmp(&self.max_lod, &other.max_lod).is_eq()
            && self.normalized == other.normalized
    }
}

impl Eq for SamplerDesc {}

impl Hash for SamplerDesc {
    #[cfg_attr(feature = "inline-more", inline(always))]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.min_filter.hash(state);
        self.mag_filter.hash(state);
        self.mip_map_mode.hash(state);
        self.address_mode.hash(state);
        self.anisotropy.map(|v| v.to_bits().hash(state));
        self.min_lod.to_bits().hash(state);
        self.max_lod.to_bits().hash(state);
        self.normalized.hash(state);
    }
}

impl SamplerDesc {
    pub const fn new() -> Self {
        SamplerDesc {
            min_filter: Filter::Nearest,
            mag_filter: Filter::Nearest,
            mip_map_mode: MipMapMode::Nearest,
            address_mode: [AddressMode::Repeat; 3],
            anisotropy: None,
            min_lod: 0.0,
            max_lod: f32::INFINITY,
            normalized: true,
        }
    }
}

impl Default for SamplerDesc {
    fn default() -> Self {
        SamplerDesc::new()
    }
}
