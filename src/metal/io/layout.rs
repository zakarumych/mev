use std::range::Range;
use crate::{Extent, ImageExtent, IoError, Offset, PixelFormat, TexelShape};

/// Validates a single-layer file-to-image load and computes block-aware pitches.
pub(super) fn validate_io_image<const N: usize>(
    shape: TexelShape<N>,
    format: PixelFormat,
    image_extent: ImageExtent,
    image_layers: u32,
    image_levels: u32,
    offset: Offset<N>,
    extent: Extent<N>,
    layers: Range<u32>,
    level: u32,
) -> Result<(usize, usize, usize), IoError> {
    let invalid = |message: &str| IoError::InvalidInput(message.into());
    if !(1..=3).contains(&N) || image_extent.dimensions() != N {
        return Err(invalid("image dimension mismatch"));
    }
    if shape.format != format || shape.extent != extent {
        return Err(invalid("source format or extent differs from destination"));
    }
    if level >= image_levels || level >= 32 {
        return Err(invalid("mip level out of bounds"));
    }
    if layers.start >= image_layers || layers.end.checked_sub(layers.start) != Some(1) {
        return Err(invalid("I/O image loads require one valid layer"));
    }
    let mip = image_extent.at_level(level).into_nd::<N>();
    let block = format.block_extent();
    let blocks = [block.width(), block.height(), 1];
    for axis in 0..N {
        let start = u32::try_from(offset.0[axis]).map_err(|_| invalid("negative image offset"))?;
        let end = start
            .checked_add(extent.0[axis])
            .ok_or_else(|| invalid("image extent overflow"))?;
        if extent.0[axis] == 0 || end > mip.0[axis] {
            return Err(invalid("image region out of bounds or empty"));
        }
        if start % blocks[axis] != 0 || (extent.0[axis] % blocks[axis] != 0 && end != mip.0[axis]) {
            return Err(invalid("image region is not block aligned"));
        }
    }
    let block_size = format.block_size();
    if shape.stride[0].is_some_and(|stride| stride != block_size) {
        return Err(invalid(
            "element stride must equal the pixel format block size",
        ));
    }
    let width = extent.0[0].div_ceil(blocks[0]) as usize;
    let height = if N > 1 {
        extent.0[1].div_ceil(blocks[1]) as usize
    } else {
        1
    };
    let depth = if N > 2 { extent.0[2] as usize } else { 1 };
    let row_size = width
        .checked_mul(block_size)
        .ok_or_else(|| invalid("row size overflow"))?;
    let row = if N > 1 {
        shape.stride[1].unwrap_or(row_size)
    } else {
        row_size
    };
    if row < row_size || row % block_size != 0 {
        return Err(invalid("row stride is too small or not block aligned"));
    }
    let plane_size = row
        .checked_mul(height)
        .ok_or_else(|| invalid("plane size overflow"))?;
    let plane = if N > 2 {
        shape.stride[2].unwrap_or(plane_size)
    } else {
        plane_size
    };
    if plane < plane_size || plane % row != 0 {
        return Err(invalid("plane stride is too small or not row aligned"));
    }
    let span = plane
        .checked_mul(depth - 1)
        .and_then(|n| {
            row.checked_mul(height - 1)
                .and_then(|rows| n.checked_add(rows))
        })
        .and_then(|n| n.checked_add(row_size))
        .ok_or_else(|| invalid("source byte span overflow"))?;
    Ok((row, plane, span))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Extent2, Extent3, Offset2, Offset3};

    fn layout(
        shape: TexelShape<2>,
        offset: Offset2,
        extent: Extent2,
        image_extent: Extent2,
        layer: Range<u32>,
        level: u32,
    ) -> Result<(usize, usize, usize), IoError> {
        validate_io_image(
            shape,
            PixelFormat::Bc1RgbaUnorm,
            image_extent.into(),
            2,
            3,
            offset,
            extent,
            layer,
            level,
        )
    }

    #[test]
    fn compressed_pitches_count_blocks_and_edge_blocks() {
        let extent = Extent2::new(7, 5);
        let shape = TexelShape::new(PixelFormat::Bc1RgbaUnorm, extent);
        assert_eq!(
            layout(shape, Offset::ZERO, extent, extent, (1..2).into(), 0),
            Ok((16, 32, 32))
        );
        let extent = Extent2::new(3, 2);
        let shape = TexelShape::new(PixelFormat::Bc1RgbaUnorm, extent);
        assert_eq!(
            layout(
                shape,
                Offset::ZERO,
                extent,
                Extent2::new(7, 5),
                (0..1).into(),
                1
            ),
            Ok((8, 8, 8))
        );
    }

    #[test]
    fn invalid_regions_and_layouts_are_rejected() {
        let extent = Extent2::new(8, 8);
        let shape = TexelShape::new(PixelFormat::Bc1RgbaUnorm, extent);
        for (offset, layers, level) in [
            (Offset2::new(-1, 0), (0..1).into(), 0),
            (Offset2::new(4, 0), (0..1).into(), 0),
            (Offset::ZERO, (0..2).into(), 0),
            (Offset::ZERO, (2..3).into(), 0),
            (Offset::ZERO, (0..1).into(), 3),
        ] {
            assert!(layout(shape, offset, extent, extent, layers, level).is_err());
        }
        assert!(
            layout(
                shape.with_row_stride(8),
                Offset::ZERO,
                extent,
                extent,
                (0..1).into(),
                0
            )
            .is_err()
        );
        assert!(
            layout(
                shape.with_texel_stride(1),
                Offset::ZERO,
                extent,
                extent,
                (0..1).into(),
                0
            )
            .is_err()
        );
        assert!(
            layout(
                TexelShape::new(PixelFormat::R8Unorm, extent),
                Offset::ZERO,
                extent,
                extent,
                (0..1).into(),
                0
            )
            .is_err()
        );
        let small = Extent2::new(3, 4);
        assert!(
            layout(
                TexelShape::new(shape.format, small),
                Offset::ZERO,
                small,
                extent,
                (0..1).into(),
                0
            )
            .is_err()
        );
    }

    #[test]
    fn padded_planes_and_overflow_are_checked() {
        let extent = Extent3::new(4, 3, 2);
        let shape = TexelShape::new(PixelFormat::R8Unorm, extent)
            .with_row_stride(8)
            .with_plane_stride(32);
        let check = |shape| {
            validate_io_image(
                shape,
                PixelFormat::R8Unorm,
                extent.into(),
                1,
                1,
                Offset3::ZERO,
                extent,
                (0..1).into(),
                0,
            )
        };
        assert_eq!(check(shape), Ok((8, 32, 52)));
        assert!(check(shape.with_plane_stride(16)).is_err());
        assert!(check(shape.with_row_stride(usize::MAX)).is_err());
    }
}
