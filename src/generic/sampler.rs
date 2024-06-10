use std::hash::{Hash, Hasher};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Filter {
    Nearest,
    Linear,
}

impl Default for Filter {
    fn default() -> Self {
        Filter::Nearest
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MipMapMode {
    Nearest,
    Linear,
}

impl Default for MipMapMode {
    fn default() -> Self {
        MipMapMode::Nearest
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AddressMode {
    Repeat,
    MirrorRepeat,
    ClampToEdge,
}

impl Default for AddressMode {
    fn default() -> Self {
        AddressMode::Repeat
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SamplerDesc {
    pub min_filter: Filter,
    pub mag_filter: Filter,
    pub mip_map_mode: MipMapMode,
    pub address_mode: [AddressMode; 3],
    pub anisotropy: Option<f32>,

    pub min_lod: f32,
    pub max_lod: f32,
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
