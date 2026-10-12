use super::{
    AuxiliaryDesc, AuxiliaryMode, NnError, Precision, checked_add, checked_bytes, checked_mul,
};

/// One resolution level of a three-dimensional factorized hash grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HashGridLevel {
    /// Number of cells along each axis. Each plane has `resolution + 1` vertices per axis.
    pub resolution: u32,
    /// Number of features produced by this level.
    pub features: u32,
    /// Maximum number of feature entries in each of the three plane tables.
    pub table_size: u32,
}

/// Validated three-dimensional Fact-Hash input encoding.
///
/// Each level bilinearly interpolates independent XY, YZ, and ZX plane tables,
/// multiplies their corresponding features, then concatenates the level outputs.
/// This implements the core factorization described in the
/// [Fact-Hash paper](https://arxiv.org/pdf/2604.02836), without a checkpoint format contract.
/// Finite input coordinates are clamped to `[0, 1]`.
///
/// Parameters use `f32` auxiliary storage, ordered by level, plane (XY, YZ, ZX),
/// entry, then feature. Each plane contains
/// `min((resolution + 1)^2, table_size)` entries. A plane uses dense indexing when
/// all vertices fit, with the first plane axis varying fastest. Otherwise the entry is
/// `(a ^ (b * 2654435761)) % entries`, using wrapping `u32` multiplication.
/// The caller initializes these parameters.
/// Initializing every plane to zero produces zero outputs and zero parameter gradients.
///
/// Half-precision inference converts table values to `f16` before interpolation.
/// Small feature products can underflow, and high resolutions have coarse half-precision
/// coordinates. Resolutions must be exactly representable in `f16` for such networks.
/// Training uses `f32`. A trainable grid with `P` parameters needs approximately `16 * P`
/// bytes for parameters, gradients, and Adam moments, in addition to dense-network state
/// and the sample tape. Adam traverses the entire trainable table on each training step.
#[derive(Clone, Debug)]
pub struct HashGrid {
    levels: Vec<HashGridLevel>,
    layouts: Vec<HashGridLevelLayout>,
    output_arity: u32,
    parameter_count: u32,
    parameter_bytes: usize,
}

#[derive(Clone, Debug)]
pub(super) struct HashGridLevelLayout {
    pub entries: u32,
    pub parameter_offset: u32,
    pub output_offset: u32,
    pub dense: bool,
}

impl HashGrid {
    /// Validates per-level dimensions and computes the packed parameter layout.
    ///
    /// At least one level is required. All fields must be nonzero, and resolutions
    /// must not exceed `2^24`, the exact consecutive-integer range of `f32`.
    pub fn new(levels: &[HashGridLevel]) -> Result<Self, NnError> {
        if levels.is_empty() {
            return Err(NnError::InvalidDescriptor(
                "hash grid requires at least one level".into(),
            ));
        }
        let mut layouts = Vec::with_capacity(levels.len());
        let mut output_arity = 0;
        let mut parameter_count = 0;
        for (index, level) in levels.iter().enumerate() {
            if level.resolution == 0 || level.resolution > 1 << 24 {
                return Err(NnError::InvalidDescriptor(format!(
                    "hash grid level {index} resolution must be in 1..=2^24"
                )));
            }
            if level.features == 0 || level.table_size == 0 {
                return Err(NnError::InvalidDescriptor(format!(
                    "hash grid level {index} features and table size must be nonzero"
                )));
            }
            let side = u64::from(level.resolution) + 1;
            let dense_entries = side * side;
            let entries = dense_entries.min(u64::from(level.table_size)) as u32;
            layouts.push(HashGridLevelLayout {
                entries,
                parameter_offset: parameter_count,
                output_offset: output_arity,
                dense: dense_entries <= u64::from(level.table_size),
            });
            let level_parameters = checked_mul(
                checked_mul(entries, level.features, "hash grid plane parameter count")?,
                3,
                "hash grid level parameter count",
            )?;
            parameter_count = checked_add(
                parameter_count,
                level_parameters,
                "hash grid parameter count",
            )?;
            output_arity = checked_add(output_arity, level.features, "hash grid output arity")?;
        }
        Ok(Self {
            levels: levels.to_vec(),
            layouts,
            output_arity,
            parameter_count,
            parameter_bytes: checked_bytes(parameter_count, "hash grid parameter bytes")?,
        })
    }

    pub fn levels(&self) -> &[HashGridLevel] {
        &self.levels
    }

    pub const fn input_arity(&self) -> u32 {
        3
    }

    pub const fn output_arity(&self) -> u32 {
        self.output_arity
    }

    /// Total number of `f32` values across all plane tables.
    pub const fn parameter_count(&self) -> u32 {
        self.parameter_count
    }

    /// Minimum byte size of the auxiliary parameter slice.
    pub const fn parameter_bytes(&self) -> usize {
        self.parameter_bytes
    }

    /// Declares the grid's `array<f32>` auxiliary binding.
    ///
    /// Trainable bindings have exactly [`Self::parameter_count`] elements and use Adam.
    /// Read-only bindings require at least [`Self::parameter_bytes`] bytes at invocation.
    /// A manually declared larger trainable binding is also valid; the grid uses its
    /// required prefix, while Adam updates every declared element, including the tail.
    pub fn auxiliary(&self, name: impl Into<String>, trainable: bool) -> AuxiliaryDesc {
        AuxiliaryDesc::new(
            name,
            "array<f32>",
            if trainable {
                AuxiliaryMode::Trainable {
                    elements: self.parameter_count,
                }
            } else {
                AuxiliaryMode::ReadOnly
            },
        )
    }

    pub(super) fn level_layouts(&self) -> &[HashGridLevelLayout] {
        &self.layouts
    }

    pub(super) fn validate_precision(&self, precision: Precision) -> Result<(), NnError> {
        if precision == Precision::F16 {
            for (index, level) in self.levels.iter().enumerate() {
                let resolution = level.resolution as f32;
                if resolution > 65504.0 || half::f16::from_f32(resolution).to_f32() != resolution {
                    return Err(NnError::InvalidDescriptor(format!(
                        "hash grid level {index} resolution must be exactly representable in f16 and at most 65504"
                    )));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_layout_uses_dense_and_capped_planes() -> Result<(), NnError> {
        let levels = [
            HashGridLevel {
                resolution: 1,
                features: 2,
                table_size: 64,
            },
            HashGridLevel {
                resolution: 3,
                features: 3,
                table_size: 8,
            },
        ];
        let grid = HashGrid::new(&levels)?;
        assert_eq!(grid.levels(), levels);
        assert_eq!(grid.input_arity(), 3);
        assert_eq!(grid.output_arity(), 5);
        assert_eq!(grid.parameter_count(), 96);
        assert_eq!(grid.parameter_bytes(), 384);
        let first = &grid.level_layouts()[0];
        assert_eq!(first.entries, 4);
        assert_eq!(first.parameter_offset, 0);
        assert_eq!(first.output_offset, 0);
        assert!(first.dense);
        let second = &grid.level_layouts()[1];
        assert_eq!(second.entries, 8);
        assert_eq!(second.parameter_offset, 24);
        assert_eq!(second.output_offset, 2);
        assert!(!second.dense);
        let exact = HashGrid::new(&[HashGridLevel {
            resolution: 3,
            features: 1,
            table_size: 16,
        }])?;
        assert!(exact.level_layouts()[0].dense);
        Ok(())
    }

    #[test]
    fn invalid_dimensions_and_parameter_overflow_are_rejected() -> Result<(), NnError> {
        assert!(HashGrid::new(&[]).is_err());
        for (resolution, features, table_size) in [
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
            ((1 << 24) + 1, 1, 1),
            (1, u32::MAX, 1),
            (1 << 24, 1, u32::MAX),
        ] {
            assert!(
                HashGrid::new(&[HashGridLevel {
                    resolution,
                    features,
                    table_size,
                }])
                .is_err()
            );
        }
        assert!(
            HashGrid::new(
                &[HashGridLevel {
                    resolution: 1,
                    features: u32::MAX / 3,
                    table_size: 1,
                }; 2]
            )
            .is_err()
        );
        let capped = HashGrid::new(&[HashGridLevel {
            resolution: 1 << 24,
            features: 2,
            table_size: 7,
        }])?;
        assert_eq!(capped.parameter_count(), 42);
        assert!(!capped.level_layouts()[0].dense);
        Ok(())
    }

    #[test]
    fn half_precision_requires_representable_resolution() -> Result<(), NnError> {
        for resolution in [1, 2048, 2050, 32768, 65504] {
            let grid = HashGrid::new(&[HashGridLevel {
                resolution,
                features: 1,
                table_size: 1,
            }])?;
            grid.validate_precision(Precision::F16)?;
        }
        for resolution in [2049, 65505, 65536, 1 << 24] {
            let grid = HashGrid::new(&[HashGridLevel {
                resolution,
                features: 1,
                table_size: 1,
            }])?;
            assert!(grid.validate_precision(Precision::F16).is_err());
            grid.validate_precision(Precision::F32)?;
        }
        Ok(())
    }

    #[test]
    fn auxiliary_helper_declares_exact_trainable_size() -> Result<(), NnError> {
        let grid = HashGrid::new(&[HashGridLevel {
            resolution: 2,
            features: 3,
            table_size: 4,
        }])?;
        for trainable in [false, true] {
            let auxiliary = grid.auxiliary("grid_parameters", trainable);
            assert_eq!(auxiliary.name, "grid_parameters");
            assert_eq!(auxiliary.wgsl_type, "array<f32>");
            assert_eq!(
                auxiliary.mode,
                if trainable {
                    AuxiliaryMode::Trainable { elements: 36 }
                } else {
                    AuxiliaryMode::ReadOnly
                }
            );
        }
        Ok(())
    }
}
