//! Headless verification for the `Cvt` kernel (`src/kernels/cvt.rs`).
//!
//! Run with `cargo run --example cvt_smoke`. Creates a device with no
//! surface/window, exercises the masked-atomic buffer-write pattern in
//! isolation, then each `Cvt` op family, asserting results on the CPU
//! after reading buffers back.

use mev::kernels::{Axis, AxisMap, AxisSwizzle, Cvt, CvtOptions};
use mev::{
    AddressMode, Arguments, Buffer, BufferInitDesc, BufferUsage, ComputePipelineDesc, DeviceDesc,
    Extent1, Extent2, Extent3, Features, Filter, LibraryDesc, LibraryInput, Offset2,
    PipelineStages, PixelFormat, Swizzle, TexelShape, VertexFormat,
};

#[derive(Arguments)]
struct AtomicSmokeArguments {
    #[mev(storage)]
    #[mev(compute)]
    buf: Buffer,
}

/// Standalone smoke test for the masked-atomic `atomicAnd`/`atomicOr`
/// sub-word buffer-write pattern `Cvt`'s buffer encode path relies on,
/// run through naga's actual SPIR-V lowering and this machine's Vulkan
/// backend (not just compiled - executed and checked).
fn atomic_smoke_test(device: &mev::Device, queue: &mut mev::Queue) {
    let code = r#"
@group(0) @binding(0) var<storage, read_write> buf: array<atomic<u32>>;

@compute @workgroup_size(16, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let byte_offset = gid.x;
    let word_index = byte_offset / 4u;
    let bit_offset = (byte_offset % 4u) * 8u;
    let mask = 0xFFu << bit_offset;
    atomicAnd(&buf[word_index], ~mask);
    atomicOr(&buf[word_index], (byte_offset << bit_offset) & mask);
}
"#;

    let library = device
        .new_shader_library(LibraryDesc {
            name: "atomic_smoke",
            input: LibraryInput::wgsl(code),
        })
        .expect("atomic smoke shader failed to compile");

    let pipeline = device
        .new_compute_pipeline(ComputePipelineDesc {
            name: "atomic_smoke",
            shader: library.entry("main"),
            work_group_size: [16, 1, 1],
            constants: 0,
            arguments: &[AtomicSmokeArguments::LAYOUT],
        })
        .expect("atomic smoke pipeline failed to build");

    let mut buf = device.new_buffer_init(BufferInitDesc {
        data: &[0u8; 16],
        usage: BufferUsage::STORAGE | BufferUsage::HOST_READ | BufferUsage::HOST_WRITE,
        name: "atomic_smoke_buf",
    });

    let mut encoder = queue.new_command_encoder();
    {
        let mut compute = encoder.compute();
        compute.with_pipeline(&pipeline);
        compute.with_arguments(0, &AtomicSmokeArguments { buf: buf.clone() });
        compute.dispatch(Extent3::new(1, 1, 1));
    }
    let cbuf = encoder.finish();
    queue.submit_checkpoint([cbuf]).unwrap();
    queue.wait_idle().unwrap();

    let mut out = [0u8; 16];
    buf.read(0, &mut out).unwrap();
    for i in 0..16u8 {
        assert_eq!(out[i as usize], i, "atomic smoke test: byte {i} mismatch");
    }
    println!("[PASS] masked-atomic buffer-write smoke test");
}

fn make_readback_buffer(device: &mev::Device, data: &[u8], name: &str) -> Buffer {
    device.new_buffer_init(BufferInitDesc {
        data,
        usage: BufferUsage::STORAGE | BufferUsage::HOST_READ | BufferUsage::HOST_WRITE,
        name,
    })
}

fn submit_and_wait(queue: &mut mev::Queue, cbuf: mev::CommandBuffer) {
    queue.submit_checkpoint([cbuf]).unwrap();
    queue.wait_idle().unwrap();
}

fn test_cross_kind_unorm8_to_uint8(device: &mev::Device, queue: &mut mev::Queue, cvt: &Cvt) {
    // Raw Unorm8 byte 128 decodes to canonical 128/255; encoding that back
    // to Uint8 (raw_dst = round(canonical * 255)) must land on 128 exactly.
    let src = make_readback_buffer(device, &[128u8], "src_u8");
    let dst = make_readback_buffer(device, &[0u8], "dst_u8");

    let mut encoder = queue.new_command_encoder();
    cvt.buffer_to_buffer::<1>(
        &mut encoder,
        &src,
        TexelShape::new(VertexFormat::Unorm8.into(), Extent1::new(1)),
        &dst,
        TexelShape::new(VertexFormat::Uint8.into(), Extent1::new(1)),
    );
    submit_and_wait(queue, encoder.finish());

    let mut out = [0u8; 1];
    let mut dst = dst;
    dst.read(0, &mut out).unwrap();
    assert_eq!(
        out[0], 128,
        "Unorm8 0.5 (raw 128) -> Uint8 should be 128, got {}",
        out[0]
    );
    println!("[PASS] cross-kind conversion: Unorm8 128 -> Uint8 128");
}

fn test_transpose_mirror_2d(device: &mev::Device, queue: &mut mev::Queue, cvt: &Cvt) {
    // src is a 2x2 row-major Uint8 grid: [ [0,1], [2,3] ].
    // axis0 (x) -> dst axis1 (y), mirrored; axis1 (y) -> dst axis0 (x), not mirrored.
    // Expected dst (2x2 row-major): [1,3,0,2] (worked out by hand in the
    // task's verification notes; see report).
    let src_data = [0u8, 1, 2, 3];
    let src = make_readback_buffer(device, &src_data, "src_transpose");
    let dst = make_readback_buffer(device, &[0u8; 4], "dst_transpose");

    let axes = AxisSwizzle {
        x: AxisMap::Mirror(Axis::Y),
        y: AxisMap::Direct(Axis::X),
        z: AxisMap::Zero,
        w: AxisMap::Zero,
    };

    let mut encoder = queue.new_command_encoder();
    cvt.buffer_to_buffer_with::<2, 2>(
        &mut encoder,
        &src,
        TexelShape::new(VertexFormat::Uint8.into(), Extent2::new(2, 2)),
        &dst,
        TexelShape::new(VertexFormat::Uint8.into(), Extent2::new(2, 2)),
        axes,
    );
    submit_and_wait(queue, encoder.finish());

    let mut out = [0u8; 4];
    let mut dst = dst;
    dst.read(0, &mut out).unwrap();
    assert_eq!(out, [1, 3, 0, 2], "transpose+mirror mismatch: {:?}", out);
    println!("[PASS] Buffer->Buffer 2D transpose+mirror");
}

fn test_diagonal_read(device: &mev::Device, queue: &mut mev::Queue, cvt: &Cvt) {
    // src is a 3x3 row-major Uint8 grid 0..9; both src axes map to the
    // single dst axis, reading the diagonal: dst[i] = src[i][i].
    let src_data: [u8; 9] = std::array::from_fn(|i| i as u8);
    let src = make_readback_buffer(device, &src_data, "src_diag");
    let dst = make_readback_buffer(device, &[0u8; 3], "dst_diag");

    let axes = AxisSwizzle {
        x: AxisMap::Direct(Axis::X),
        y: AxisMap::Direct(Axis::X),
        z: AxisMap::Zero,
        w: AxisMap::Zero,
    };

    let mut encoder = queue.new_command_encoder();
    cvt.buffer_to_buffer_with::<2, 1>(
        &mut encoder,
        &src,
        TexelShape::new(VertexFormat::Uint8.into(), Extent2::new(3, 3)),
        &dst,
        TexelShape::new(VertexFormat::Uint8.into(), Extent1::new(3)),
        axes,
    );
    submit_and_wait(queue, encoder.finish());

    let mut out = [0u8; 3];
    let mut dst = dst;
    dst.read(0, &mut out).unwrap();
    assert_eq!(out, [0, 4, 8], "diagonal-read mismatch: {:?}", out);
    println!("[PASS] Buffer->Buffer diagonal read (SRC=2, DST=1)");
}

fn test_splat(device: &mev::Device, queue: &mut mev::Queue, cvt: &Cvt) {
    // src is a 1D Uint8 buffer [10,20,30]; dst is 3x2, dst axis1 (y) is a
    // pure splat (broadcast) axis, so every row of dst equals src.
    let src_data = [10u8, 20, 30];
    let src = make_readback_buffer(device, &src_data, "src_splat");
    let dst = make_readback_buffer(device, &[0u8; 6], "dst_splat");

    let axes = AxisSwizzle {
        x: AxisMap::Direct(Axis::X),
        y: AxisMap::Zero,
        z: AxisMap::Zero,
        w: AxisMap::Zero,
    };

    let mut encoder = queue.new_command_encoder();
    cvt.buffer_to_buffer_with::<1, 2>(
        &mut encoder,
        &src,
        TexelShape::new(VertexFormat::Uint8.into(), Extent1::new(3)),
        &dst,
        TexelShape::new(VertexFormat::Uint8.into(), Extent2::new(3, 2)),
        axes,
    );
    submit_and_wait(queue, encoder.finish());

    let mut out = [0u8; 6];
    let mut dst = dst;
    dst.read(0, &mut out).unwrap();
    assert_eq!(out, [10, 20, 30, 10, 20, 30], "splat mismatch: {:?}", out);
    println!("[PASS] Buffer->Buffer splat (SRC=1, DST=2)");
}

fn test_buffer_in_place(device: &mev::Device, queue: &mut mev::Queue, cvt: &Cvt) {
    let buf = make_readback_buffer(device, &[128u8], "in_place_buf");

    let mut encoder = queue.new_command_encoder();
    cvt.buffer_in_place(
        &mut encoder,
        &buf,
        VertexFormat::Unorm8.into(),
        VertexFormat::Uint8.into(),
        1,
        Swizzle::IDENTITY,
    );
    submit_and_wait(queue, encoder.finish());

    let mut out = [0u8; 1];
    let mut buf = buf;
    buf.read(0, &mut out).unwrap();
    assert_eq!(
        out[0], 128,
        "in-place Unorm8 128 -> Uint8 should be 128, got {}",
        out[0]
    );
    println!("[PASS] cvt_buffer_in_place");
}

fn test_image_to_image(device: &mev::Device, queue: &mut mev::Queue, cvt: &Cvt) {
    use mev::{ImageDesc, ImageExtent, ImageUsage};

    let extent = mev::Extent2::new(2, 2);

    let src_image = device.new_image(ImageDesc::new(
        ImageExtent::D2(extent),
        PixelFormat::Rgba8Unorm,
        ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
    ));
    let dst_image = device.new_image(ImageDesc::new(
        ImageExtent::D2(extent),
        PixelFormat::Rgba8Unorm,
        ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
    ));

    // 4 RGBA8 texels, distinct per channel so an R<->B swizzle swap is
    // observable in the readback.
    let src_pixels: [u8; 16] = [
        10, 20, 30, 255, // (0,0)
        40, 50, 60, 255, // (1,0)
        70, 80, 90, 255, // (0,1)
        100, 110, 120, 255, // (1,1)
    ];
    let staging = device.new_buffer_init(BufferInitDesc {
        data: &src_pixels,
        usage: BufferUsage::TRANSFER_SRC | BufferUsage::HOST_WRITE,
        name: "src_stage",
    });
    let readback = device.new_buffer(mev::BufferDesc {
        size: 16,
        usage: BufferUsage::TRANSFER_DST | BufferUsage::HOST_READ,
        name: "dst_stage",
    });

    let mut encoder = queue.new_command_encoder();
    encoder.init_image(
        PipelineStages::empty(),
        PipelineStages::TRANSFER,
        &src_image,
    );
    encoder.init_image(
        PipelineStages::empty(),
        PipelineStages::TRANSFER,
        &dst_image,
    );
    {
        let mut copy = encoder.copy();
        copy.copy_buffer_to_image(
            &staging,
            TexelShape::new(PixelFormat::Rgba8Unorm, extent).with_row_stride(2 * 4),
            &src_image,
        );
        copy.barrier(PipelineStages::TRANSFER, PipelineStages::COMPUTE_SHADER);
    }

    let swizzle = Swizzle {
        r: mev::ComponentMap::B,
        g: mev::ComponentMap::G,
        b: mev::ComponentMap::R,
        a: mev::ComponentMap::A,
    };

    cvt.image_to_image_with::<2, 2>(&mut encoder, &src_image, &dst_image, swizzle);

    {
        let mut copy = encoder.copy();
        copy.barrier(PipelineStages::COMPUTE_SHADER, PipelineStages::TRANSFER);
        copy.copy_image_to_buffer(
            &dst_image,
            &readback,
            TexelShape::new(PixelFormat::Rgba8Unorm, extent).with_row_stride(2 * 4),
        );
    }

    let cbuf = encoder.finish();
    submit_and_wait(queue, cbuf);

    let mut out = [0u8; 16];
    let mut readback = readback;
    readback.read(0, &mut out).unwrap();

    let mut expected = src_pixels;
    for px in 0..4 {
        expected[px * 4] = src_pixels[px * 4 + 2];
        expected[px * 4 + 2] = src_pixels[px * 4];
    }
    assert_eq!(
        out, expected,
        "Image->Image R<->B swizzle mismatch: {:?} != {:?}",
        out, expected
    );
    println!("[PASS] Image->Image format+swizzle (R<->B channel swap)");
}

/// Standard sRGB OETF (linear -> sRGB-encoded), used to compute the
/// expected CPU-side result independent of the GPU's own hardware encoder.
fn srgb_encode(linear: f32) -> f32 {
    if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

/// `cvt_image_to_image` into an sRGB destination must route through the
/// render pipeline (WGSL storage textures have no sRGB write format, so the
/// compute path can't target one at all): the fragment shader outputs a
/// linear-space value decoded from the (non-sRGB) `Rgba8Unorm` source, and
/// the hardware ROP applies the sRGB encode automatically on write to the
/// sRGB color attachment - so the output bytes are the sRGB-encoded curve
/// applied to the source's linear Unorm8 values, not a raw copy.
fn test_image_to_image_srgb_dst(device: &mev::Device, queue: &mut mev::Queue, cvt: &Cvt) {
    use mev::{ImageDesc, ImageExtent, ImageUsage};

    let extent = mev::Extent2::new(2, 2);

    let src_image = device.new_image(ImageDesc::new(
        ImageExtent::D2(extent),
        PixelFormat::Rgba8Unorm,
        ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
    ));
    // sRGB destinations are written via a render pass color attachment
    // (TARGET), not a storage image (STORAGE) - that's exactly the
    // constraint that forces the render-pipeline path in the first place.
    let dst_image = device.new_image(ImageDesc::new(
        ImageExtent::D2(extent),
        PixelFormat::Rgba8Srgb,
        ImageUsage::TARGET | ImageUsage::TRANSFER_SRC,
    ));

    let src_pixels: [u8; 16] = [
        10, 20, 30, 255, // (0,0)
        60, 90, 120, 255, // (1,0)
        150, 180, 210, 255, // (0,1)
        30, 128, 255, 255, // (1,1)
    ];
    let staging = device.new_buffer_init(BufferInitDesc {
        data: &src_pixels,
        usage: BufferUsage::TRANSFER_SRC | BufferUsage::HOST_WRITE,
        name: "src_stage_srgb",
    });
    let readback = device.new_buffer(mev::BufferDesc {
        size: 16,
        usage: BufferUsage::TRANSFER_DST | BufferUsage::HOST_READ,
        name: "dst_stage_srgb",
    });

    let mut encoder = queue.new_command_encoder();
    encoder.init_image(
        PipelineStages::empty(),
        PipelineStages::TRANSFER,
        &src_image,
    );
    encoder.init_image(
        PipelineStages::empty(),
        PipelineStages::COLOR_OUTPUT,
        &dst_image,
    );
    {
        let mut copy = encoder.copy();
        copy.copy_buffer_to_image(
            &staging,
            TexelShape::new(PixelFormat::Rgba8Unorm, extent).with_row_stride(2 * 4),
            &src_image,
        );
        copy.barrier(PipelineStages::TRANSFER, PipelineStages::COLOR_OUTPUT);
    }

    cvt.image_to_image::<2>(&mut encoder, &src_image, &dst_image);

    {
        let mut copy = encoder.copy();
        copy.barrier(PipelineStages::COLOR_OUTPUT, PipelineStages::TRANSFER);
        copy.copy_image_to_buffer(
            &dst_image,
            &readback,
            TexelShape::new(PixelFormat::Rgba8Srgb, extent).with_row_stride(2 * 4),
        );
    }

    let cbuf = encoder.finish();
    submit_and_wait(queue, cbuf);

    let mut out = [0u8; 16];
    let mut readback = readback;
    readback.read(0, &mut out).unwrap();

    for i in 0..16 {
        if i % 4 == 3 {
            // Alpha is never gamma-encoded.
            assert_eq!(
                out[i], src_pixels[i],
                "sRGB dst: alpha byte {i} should pass through unencoded: {} != {}",
                out[i], src_pixels[i]
            );
            continue;
        }
        let linear = src_pixels[i] as f32 / 255.0;
        let expected = (srgb_encode(linear) * 255.0).round().clamp(0.0, 255.0) as i32;
        let actual = out[i] as i32;
        assert!(
            (actual - expected).abs() <= 1,
            "sRGB dst: byte {i} = {actual}, expected ~{expected} (src {} gamma-encoded)",
            src_pixels[i]
        );
    }
    println!("[PASS] Image->Image into sRGB destination (render-pipeline gamma encode)");
}

/// `cvt_image_to_image_resample` into an sRGB destination combines both
/// extensions at once: mismatched src/dst extents (nearest-neighbor
/// upscale, same 2x2 -> 4x4 quadrant mapping as
/// `test_image_to_image_resample`) *and* a render-pipeline sRGB
/// destination (same hardware gamma-encode-on-write as
/// `test_image_to_image_srgb_dst`). Each dst quadrant reads whichever src
/// texel its nearest-index mapping lands on, then that texel's linear
/// Unorm8 value must come out sRGB-OETF-encoded in the readback.
fn test_image_to_image_srgb_dst_resample(device: &mev::Device, queue: &mut mev::Queue, cvt: &Cvt) {
    use mev::{ImageDesc, ImageExtent, ImageUsage};

    let src_extent = mev::Extent2::new(2, 2);
    let dst_extent = mev::Extent2::new(4, 4);

    let src_image = device.new_image(ImageDesc::new(
        ImageExtent::D2(src_extent),
        PixelFormat::Rgba8Unorm,
        ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
    ));
    // sRGB destinations are written via a render pass color attachment
    // (TARGET), not a storage image (STORAGE) - same constraint as
    // `test_image_to_image_srgb_dst`.
    let dst_image = device.new_image(ImageDesc::new(
        ImageExtent::D2(dst_extent),
        PixelFormat::Rgba8Srgb,
        ImageUsage::TARGET | ImageUsage::TRANSFER_SRC,
    ));

    // Row-major 2x2 grid: A=(0,0), B=(1,0), C=(0,1), D=(1,1).
    let a = [10u8, 20, 30, 255];
    let b = [40u8, 50, 60, 255];
    let c = [70u8, 80, 90, 255];
    let d = [100u8, 110, 120, 255];
    let src_pixels: [u8; 16] = std::array::from_fn(|i| {
        let px = i / 4;
        let ch = i % 4;
        [a, b, c, d][px][ch]
    });

    let staging = device.new_buffer_init(BufferInitDesc {
        data: &src_pixels,
        usage: BufferUsage::TRANSFER_SRC | BufferUsage::HOST_WRITE,
        name: "i2i_srgb_resample_stage",
    });
    let readback = device.new_buffer(mev::BufferDesc {
        size: 64,
        usage: BufferUsage::TRANSFER_DST | BufferUsage::HOST_READ,
        name: "i2i_srgb_resample_readback",
    });

    let mut encoder = queue.new_command_encoder();
    encoder.init_image(
        PipelineStages::empty(),
        PipelineStages::TRANSFER,
        &src_image,
    );
    encoder.init_image(
        PipelineStages::empty(),
        PipelineStages::COLOR_OUTPUT,
        &dst_image,
    );
    {
        let mut copy = encoder.copy();
        copy.copy_buffer_to_image(
            &staging,
            TexelShape::new(PixelFormat::Rgba8Unorm, src_extent).with_row_stride(2 * 4),
            &src_image,
        );
        copy.barrier(PipelineStages::TRANSFER, PipelineStages::COLOR_OUTPUT);
    }

    cvt.image_to_image_with::<2, 2>(&mut encoder, &src_image, &dst_image, Filter::Nearest);

    {
        let mut copy = encoder.copy();
        copy.barrier(PipelineStages::COLOR_OUTPUT, PipelineStages::TRANSFER);
        copy.copy_image_to_buffer(
            &dst_image,
            &readback,
            TexelShape::new(PixelFormat::Rgba8Srgb, dst_extent).with_row_stride(4 * 4),
        );
    }

    let cbuf = encoder.finish();
    submit_and_wait(queue, cbuf);

    let mut out = [0u8; 64];
    let mut readback = readback;
    readback.read(0, &mut out).unwrap();

    // Same quadrant mapping as `test_image_to_image_resample`: rows 0-1
    // (dy < 2) read src row 0 (A, B); rows 2-3 (dy >= 2) read src row 1
    // (C, D). Columns 0-1 (dx < 2) read the row's first src texel; columns
    // 2-3 (dx >= 2) read the second.
    let mut expected_src: [[u8; 4]; 16] = [[0; 4]; 16];
    for dy in 0..4usize {
        let row: [[u8; 4]; 4] = if dy < 2 { [a, a, b, b] } else { [c, c, d, d] };
        for dx in 0..4usize {
            expected_src[dy * 4 + dx] = row[dx];
        }
    }

    for px in 0..16 {
        let src_px = expected_src[px];
        for ch in 0..4 {
            let i = px * 4 + ch;
            if ch == 3 {
                // Alpha is never gamma-encoded.
                assert_eq!(
                    out[i], src_px[ch],
                    "sRGB resample dst: alpha byte {i} should pass through unencoded: {} != {}",
                    out[i], src_px[ch]
                );
                continue;
            }
            let linear = src_px[ch] as f32 / 255.0;
            let expected = (srgb_encode(linear) * 255.0).round().clamp(0.0, 255.0) as i32;
            let actual = out[i] as i32;
            assert!(
                (actual - expected).abs() <= 1,
                "sRGB resample dst: byte {i} = {actual}, expected ~{expected} (src {} gamma-encoded)",
                src_px[ch]
            );
        }
    }
    println!(
        "[PASS] Image->Image resample into sRGB destination (2x2 -> 4x4 nearest upscale, render-pipeline gamma encode)"
    );
}

fn test_buffer_to_image(device: &mev::Device, queue: &mut mev::Queue, cvt: &Cvt) {
    use mev::{ImageDesc, ImageExtent, ImageUsage};

    let extent = mev::Extent2::new(2, 2);

    let dst_image = device.new_image(ImageDesc::new(
        ImageExtent::D2(extent),
        PixelFormat::Rgba8Unorm,
        ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
    ));

    let src_pixels: [u8; 16] = std::array::from_fn(|i| (i * 7) as u8);
    let src = make_readback_buffer(device, &src_pixels, "b2i_src");
    let readback = device.new_buffer(mev::BufferDesc {
        size: 16,
        usage: BufferUsage::TRANSFER_DST | BufferUsage::HOST_READ,
        name: "b2i_readback",
    });

    let mut encoder = queue.new_command_encoder();
    encoder.init_image(
        PipelineStages::empty(),
        PipelineStages::COMPUTE_SHADER,
        &dst_image,
    );

    // `src`'s rank-3 shape's z axis is a degenerate extent-1 axis (purely so
    // `TexelShape` carries a plane stride); it maps to `Zero` since `dst` (a
    // 2D image) has no third axis to receive it - unlike x/y, it must not be
    // `Direct` (there is no destination axis index 2 to write into).
    let axes = AxisSwizzle {
        x: AxisMap::Direct(Axis::X),
        y: AxisMap::Direct(Axis::Y),
        z: AxisMap::Zero,
        w: AxisMap::Zero,
    };

    cvt.buffer_to_image_with::<3, 2>(
        &mut encoder,
        &src,
        TexelShape::new(PixelFormat::Rgba8Unorm, extent.to_3d()),
        &dst_image,
        axes,
    );

    {
        let mut copy = encoder.copy();
        copy.barrier(PipelineStages::COMPUTE_SHADER, PipelineStages::TRANSFER);
        copy.copy_image_to_buffer(
            &dst_image,
            &readback,
            TexelShape::new(PixelFormat::Rgba8Unorm, extent).with_row_stride(2 * 4),
        );
    }

    submit_and_wait(queue, encoder.finish());

    let mut out = [0u8; 16];
    let mut readback = readback;
    readback.read(0, &mut out).unwrap();
    assert_eq!(
        out, src_pixels,
        "Buffer->Image identity mismatch: {:?} != {:?}",
        out, src_pixels
    );
    println!("[PASS] Buffer->Image identity copy");
}

fn test_image_to_buffer(device: &mev::Device, queue: &mut mev::Queue, cvt: &Cvt) {
    use mev::{ImageDesc, ImageExtent, ImageUsage};

    let extent = mev::Extent2::new(2, 2);

    let src_image = device.new_image(ImageDesc::new(
        ImageExtent::D2(extent),
        PixelFormat::Rgba8Unorm,
        ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
    ));

    let src_pixels: [u8; 16] = std::array::from_fn(|i| (i * 11 + 3) as u8);
    let staging = device.new_buffer_init(BufferInitDesc {
        data: &src_pixels,
        usage: BufferUsage::TRANSFER_SRC | BufferUsage::HOST_WRITE,
        name: "i2b_stage",
    });
    let dst = make_readback_buffer(device, &[0u8; 16], "i2b_dst");

    let mut encoder = queue.new_command_encoder();
    encoder.init_image(
        PipelineStages::empty(),
        PipelineStages::TRANSFER,
        &src_image,
    );
    {
        let mut copy = encoder.copy();
        copy.copy_buffer_to_image(
            &staging,
            TexelShape::new(PixelFormat::Rgba8Unorm, extent).with_row_stride(2 * 4),
            &src_image,
        );
        copy.barrier(PipelineStages::TRANSFER, PipelineStages::COMPUTE_SHADER);
    }

    cvt.image_to_buffer_with::<2, 3>(
        &mut encoder,
        &src_image,
        &dst,
        TexelShape::new(PixelFormat::Rgba8Unorm, extent.to_3d()),
        CvtOptions::default(),
    );

    submit_and_wait(queue, encoder.finish());

    let mut out = [0u8; 16];
    let mut dst = dst;
    dst.read(0, &mut out).unwrap();
    assert_eq!(
        out, src_pixels,
        "Image->Buffer identity mismatch: {:?} != {:?}",
        out, src_pixels
    );
    println!("[PASS] Image->Buffer identity copy");
}

/// Non-resample entry points must panic (before recording any GPU work) on
/// src/dst extent mismatch, never silently resample.
fn test_extent_mismatch_panics(device: &mev::Device, queue: &mut mev::Queue, cvt: &Cvt) {
    use mev::{ImageDesc, ImageExtent, ImageUsage};
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let src_image = device.new_image(ImageDesc::new(
        ImageExtent::D2(mev::Extent2::new(2, 2)),
        PixelFormat::Rgba8Unorm,
        ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
    ));
    let dst_image = device.new_image(ImageDesc::new(
        ImageExtent::D2(mev::Extent2::new(4, 4)),
        PixelFormat::Rgba8Unorm,
        ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
    ));

    let mut encoder = queue.new_command_encoder();
    let result = catch_unwind(AssertUnwindSafe(|| {
        cvt.image_to_image::<2>(&mut encoder, &src_image, &dst_image);
    }));
    assert!(
        result.is_err(),
        "cvt_image_to_image must panic on src/dst extent mismatch, not silently resample"
    );
    // The panic fires before any GPU work is recorded, so `encoder` is
    // still a valid (empty) recording - finish and submit it rather than
    // dropping it, so its command buffer/pool is returned for tracking
    // instead of leaking (the `Queue` asserts all command buffers were
    // deallocated on drop).
    submit_and_wait(queue, encoder.finish());
    println!("[PASS] cvt_image_to_image panics on extent mismatch (no silent resample)");
}

/// `cvt_image_to_image_resample` upscales a 2x2 `Rgba8Unorm` source into a
/// 4x4 destination with `Filter::Nearest`. Verified against a hand-computed
/// expected nearest-neighbor mapping: `dst` texel `(dx, dy)` samples `src`
/// texel `(dx * 2 / 4, dy * 2 / 4)`, so each 2x2 quadrant of `dst` is a
/// solid block equal to one `src` texel.
fn test_image_to_image_resample(device: &mev::Device, queue: &mut mev::Queue, cvt: &Cvt) {
    use mev::{ImageDesc, ImageExtent, ImageUsage};

    let src_extent = mev::Extent2::new(2, 2);
    let dst_extent = mev::Extent2::new(4, 4);

    let src_image = device.new_image(ImageDesc::new(
        ImageExtent::D2(src_extent),
        PixelFormat::Rgba8Unorm,
        ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
    ));
    let dst_image = device.new_image(ImageDesc::new(
        ImageExtent::D2(dst_extent),
        PixelFormat::Rgba8Unorm,
        ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
    ));

    // Row-major 2x2 grid: A=(0,0), B=(1,0), C=(0,1), D=(1,1).
    let a = [10u8, 20, 30, 255];
    let b = [40u8, 50, 60, 255];
    let c = [70u8, 80, 90, 255];
    let d = [100u8, 110, 120, 255];
    let src_pixels: [u8; 16] = std::array::from_fn(|i| {
        let px = i / 4;
        let ch = i % 4;
        [a, b, c, d][px][ch]
    });

    let staging = device.new_buffer_init(BufferInitDesc {
        data: &src_pixels,
        usage: BufferUsage::TRANSFER_SRC | BufferUsage::HOST_WRITE,
        name: "i2i_resample_stage",
    });
    let readback = device.new_buffer(mev::BufferDesc {
        size: 64,
        usage: BufferUsage::TRANSFER_DST | BufferUsage::HOST_READ,
        name: "i2i_resample_readback",
    });

    let mut encoder = queue.new_command_encoder();
    encoder.init_image(
        PipelineStages::empty(),
        PipelineStages::TRANSFER,
        &src_image,
    );
    encoder.init_image(
        PipelineStages::empty(),
        PipelineStages::COMPUTE_SHADER,
        &dst_image,
    );
    {
        let mut copy = encoder.copy();
        copy.copy_buffer_to_image(
            &staging,
            TexelShape::new(PixelFormat::Rgba8Unorm, src_extent).with_row_stride(2 * 4),
            &src_image,
        );
        copy.barrier(PipelineStages::TRANSFER, PipelineStages::COMPUTE_SHADER);
    }

    cvt.image_to_image_with::<2, 2>(&mut encoder, &src_image, &dst_image, Filter::Nearest);

    {
        let mut copy = encoder.copy();
        copy.barrier(PipelineStages::COMPUTE_SHADER, PipelineStages::TRANSFER);
        copy.copy_image_to_buffer(
            &dst_image,
            &readback,
            TexelShape::new(PixelFormat::Rgba8Unorm, dst_extent).with_row_stride(4 * 4),
        );
    }

    submit_and_wait(queue, encoder.finish());

    let mut out = [0u8; 64];
    let mut readback = readback;
    readback.read(0, &mut out).unwrap();

    // Row 0-1 (dy < 2): quadrant columns A,A,B,B. Row 2-3 (dy >= 2): C,C,D,D.
    let mut expected = [0u8; 64];
    let rows: [[u8; 4]; 4] = [a, a, b, b];
    for dy in 0..2usize {
        for dx in 0..4usize {
            expected[(dy * 4 + dx) * 4..(dy * 4 + dx) * 4 + 4].copy_from_slice(&rows[dx]);
        }
    }
    let rows: [[u8; 4]; 4] = [c, c, d, d];
    for dy in 2..4usize {
        for dx in 0..4usize {
            expected[(dy * 4 + dx) * 4..(dy * 4 + dx) * 4 + 4].copy_from_slice(&rows[dx]);
        }
    }

    assert_eq!(
        out, expected,
        "Image->Image resample (2x2 -> 4x4 nearest upscale) mismatch:\n  got:      {:?}\n  expected: {:?}",
        out, expected
    );
    println!("[PASS] Image->Image resample (2x2 -> 4x4 nearest upscale)");
}

/// `cvt_buffer_to_buffer_resample` has no `filter` parameter at all (buffers
/// have no hardware sampler, so resampling is always nearest-index with
/// `address_mode` wrapping - see its doc comment); this exercises that
/// nearest-index mapping is correct: a 1D length-2 `Uint8` src `[10, 20]`
/// upscaled to a length-4 dst maps dst indices `{0,1} -> src[0]`,
/// `{2,3} -> src[1]` (same `(p+0.5)/dst_extent * src_extent` formula as the
/// image case).
fn test_buffer_to_buffer_resample(device: &mev::Device, queue: &mut mev::Queue, cvt: &Cvt) {
    let src_data = [10u8, 20];
    let src = make_readback_buffer(device, &src_data, "src_bb_resample");
    let dst = make_readback_buffer(device, &[0u8; 4], "dst_bb_resample");

    let mut encoder = queue.new_command_encoder();
    cvt.buffer_to_buffer_with::<1, 1>(
        &mut encoder,
        &src,
        TexelShape::new(VertexFormat::Uint8.into(), Extent1::new(2)),
        &dst,
        TexelShape::new(VertexFormat::Uint8.into(), Extent1::new(4)),
        CvtOptions::default()
            .filter(Filter::Nearest)
            .address_mode([AddressMode::ClampToEdge; 1]),
    );
    submit_and_wait(queue, encoder.finish());

    let mut out = [0u8; 4];
    let mut dst = dst;
    dst.read(0, &mut out).unwrap();
    assert_eq!(
        out,
        [10, 10, 20, 20],
        "buffer<->buffer resample mismatch: {:?}",
        out
    );
    println!("[PASS] Buffer->Buffer resample (1D length-2 -> length-4 nearest upscale)");
}

/// A `Buffer` src has no hardware sampler (see `CvtOptions::resize`'s doc):
/// `buffer_to_buffer_with` and `buffer_to_image_with` must panic (before
/// recording any GPU work) when given a non-`Nearest` filter in either
/// `resize` slot, rather than silently falling back to nearest.
fn test_buffer_resample_non_nearest_filter_panics(
    device: &mev::Device,
    queue: &mut mev::Queue,
    cvt: &Cvt,
) {
    use mev::{ImageDesc, ImageExtent, ImageUsage};
    use std::panic::{AssertUnwindSafe, catch_unwind};

    {
        let src = make_readback_buffer(device, &[10u8, 20], "b2b_resample_linear_src");
        let dst = make_readback_buffer(device, &[0u8; 4], "b2b_resample_linear_dst");
        let mut encoder = queue.new_command_encoder();
        let result = catch_unwind(AssertUnwindSafe(|| {
            cvt.buffer_to_buffer_with::<1, 1>(
                &mut encoder,
                &src,
                TexelShape::new(VertexFormat::Uint8.into(), Extent1::new(2)),
                &dst,
                TexelShape::new(VertexFormat::Uint8.into(), Extent1::new(4)),
                CvtOptions::default().filter(Filter::Linear),
            );
        }));
        assert!(
            result.is_err(),
            "cvt_buffer_to_buffer_with must panic when a non-Nearest filter is requested for a Buffer src"
        );
        // See `test_extent_mismatch_panics`: finish/submit the still-valid
        // (empty) recording instead of dropping it, to avoid leaking its
        // command buffer/pool.
        submit_and_wait(queue, encoder.finish());
    }

    {
        let src = make_readback_buffer(device, &[0u8; 16], "b2i_resample_linear_src");
        let dst_image = device.new_image(ImageDesc::new(
            ImageExtent::D2(mev::Extent2::new(2, 2)),
            PixelFormat::Rgba8Unorm,
            ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
        ));
        let mut encoder = queue.new_command_encoder();
        let result = catch_unwind(AssertUnwindSafe(|| {
            cvt.buffer_to_image_with::<2, 2>(
                &mut encoder,
                &src,
                TexelShape::new(PixelFormat::Rgba8Unorm, mev::Extent2::new(2, 2)),
                &dst_image,
                CvtOptions::default().filter(Filter::Linear),
            );
        }));
        assert!(
            result.is_err(),
            "cvt_buffer_to_image_with must panic when a non-Nearest filter is requested for a Buffer src"
        );
        submit_and_wait(queue, encoder.finish());
    }

    println!(
        "[PASS] buffer_to_buffer_with/buffer_to_image_with panic on non-Nearest filter (buffer src has no sampler)"
    );
}

/// The tiled non-atomic dispatch processes 4 destination elements per
/// thread; a destination element count that isn't a multiple of 4
/// exercises the tail-block bounds check within the tight chain (group 1
/// covers only element 4 of `{4,5,6,7}` - sub-ids 1..4 in that group must
/// be skipped, not read/written out of bounds).
fn test_tail_block_non_multiple_of_4(device: &mev::Device, queue: &mut mev::Queue, cvt: &Cvt) {
    let src_data = [10u8, 20, 30, 40, 50];
    let src = make_readback_buffer(device, &src_data, "tail_src");
    let dst = make_readback_buffer(device, &[0u8; 5], "tail_dst");

    let mut encoder = queue.new_command_encoder();
    cvt.buffer_to_buffer::<1>(
        &mut encoder,
        &src,
        TexelShape::new(VertexFormat::Uint8.into(), Extent1::new(5)),
        &dst,
        TexelShape::new(VertexFormat::Uint8.into(), Extent1::new(5)),
    );
    submit_and_wait(queue, encoder.finish());

    let mut out = [0u8; 5];
    let mut dst = dst;
    dst.read(0, &mut out).unwrap();
    assert_eq!(
        out, src_data,
        "tail-block (5 elements, not a multiple of 4) mismatch: {:?}",
        out
    );
    println!(
        "[PASS] Buffer->Buffer tail-block handling (5 elements, byte_size==1, not a multiple of 4)"
    );
}

/// A 2D destination with an explicitly padded row stride (rows tight,
/// planes word-aligned with a gap - the exact scenario the tight/aligned
/// dispatch distinction exists for). The destination is pre-filled with a
/// sentinel byte so padding bytes, which no thread should ever touch, are
/// directly observable in the readback alongside the real, correctly
/// converted element bytes.
fn test_padded_dst_stride_sentinel(device: &mev::Device, queue: &mut mev::Queue, cvt: &Cvt) {
    let src_data: [u8; 6] = [1, 2, 3, 4, 5, 6]; // row-major 3x2, tightly packed.
    let src = make_readback_buffer(device, &src_data, "padded_src");
    let dst = make_readback_buffer(device, &[0xAAu8; 8], "padded_dst");

    let mut encoder = queue.new_command_encoder();
    cvt.buffer_to_buffer::<2>(
        &mut encoder,
        &src,
        TexelShape::new(VertexFormat::Uint8.into(), Extent2::new(3, 2)),
        &dst,
        TexelShape {
            stride: [Some(1), Some(4)],
            ..TexelShape::new(VertexFormat::Uint8.into(), Extent2::new(3, 2))
        },
    );
    submit_and_wait(queue, encoder.finish());

    let mut out = [0u8; 8];
    let mut dst = dst;
    dst.read(0, &mut out).unwrap();
    let expected = [1u8, 2, 3, 0xAA, 4, 5, 6, 0xAA];
    assert_eq!(
        out, expected,
        "padded dst_stride mismatch: {:?} != {:?}",
        out, expected
    );
    println!("[PASS] Buffer->Buffer padded dst_stride (2D, sentinel padding bytes untouched)");
}

/// The two new destination-stride preconditions the tiled non-atomic
/// dispatch's safety argument depends on must panic (before any GPU work)
/// rather than silently produce wrong output: (a) `dst_stride[0] >=
/// byte_size` - `stride == 0` slips past the older `stride %
/// component_byte_size == 0` check (`0 % x == 0`) but is a real aliasing
/// hazard; (b) a stride that is neither word-aligned nor tight against its
/// predecessor ("bad", unsupported).
fn test_new_dst_stride_preconditions_panic(
    device: &mev::Device,
    queue: &mut mev::Queue,
    cvt: &Cvt,
) {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    {
        let src = make_readback_buffer(device, &[1u8], "bad_stride_zero_src");
        let dst = make_readback_buffer(device, &[0u8], "bad_stride_zero_dst");
        let mut encoder = queue.new_command_encoder();
        let result = catch_unwind(AssertUnwindSafe(|| {
            cvt.buffer_to_buffer::<1>(
                &mut encoder,
                &src,
                TexelShape::new(VertexFormat::Uint8.into(), Extent1::new(1)),
                &dst,
                TexelShape {
                    stride: [Some(0)],
                    ..TexelShape::new(VertexFormat::Uint8.into(), Extent1::new(1))
                },
            );
        }));
        assert!(
            result.is_err(),
            "cvt_buffer_to_buffer must panic when dst_stride[0] < the format's element byte size"
        );
        submit_and_wait(queue, encoder.finish());
    }

    {
        let src = make_readback_buffer(device, &[0u8; 6], "bad_stride_axis1_src");
        let dst = make_readback_buffer(device, &[0u8; 10], "bad_stride_axis1_dst");
        let mut encoder = queue.new_command_encoder();
        let result = catch_unwind(AssertUnwindSafe(|| {
            cvt.buffer_to_buffer::<2>(
                &mut encoder,
                &src,
                TexelShape::new(VertexFormat::Uint8.into(), Extent2::new(3, 2)),
                &dst,
                TexelShape {
                    stride: [Some(1), Some(5)],
                    ..TexelShape::new(VertexFormat::Uint8.into(), Extent2::new(3, 2))
                },
            );
        }));
        assert!(
            result.is_err(),
            "cvt_buffer_to_buffer must panic on a dst stride that is neither word-aligned nor tight"
        );
        submit_and_wait(queue, encoder.finish());
    }

    println!(
        "[PASS] cvt_buffer_to_buffer panics on the new dst-stride preconditions (stride < byte_size; bad/unsupported alignment)"
    );
}

/// Rank-4 destination with every axis classified `aligned` (`T == 0`,
/// fully packed `Uint32` - dense word-sized strides are always
/// word-aligned, so `aligned` wins over the numerically tight equality at
/// every level): axis 0 rides `gid.x`, axis 1 rides `gid.y` directly, and
/// axes 2/3 fold together into `gid.z` - the one dispatch shape that needs
/// a secondary flat unravel, only reachable at rank 4. Exercises that shape
/// on real hardware, not just its generated-WGSL structure.
fn test_rank4_all_aligned_z_fold(device: &mev::Device, queue: &mut mev::Queue, cvt: &Cvt) {
    let values: [u32; 16] = std::array::from_fn(|i| 1000 + i as u32);
    let src_data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let src = make_readback_buffer(device, &src_data, "rank4_src");
    let dst = make_readback_buffer(device, &vec![0u8; 64], "rank4_dst");

    let extent = mev::Extent([2, 2, 2, 2]);

    let mut encoder = queue.new_command_encoder();
    cvt.buffer_to_buffer::<4>(
        &mut encoder,
        &src,
        TexelShape::new(VertexFormat::Uint32.into(), extent),
        &dst,
        TexelShape::new(VertexFormat::Uint32.into(), extent),
    );
    submit_and_wait(queue, encoder.finish());

    let mut out = vec![0u8; 64];
    let mut dst = dst;
    dst.read(0, &mut out).unwrap();
    assert_eq!(
        out, src_data,
        "rank-4 all-aligned (T=0) z-folded dispatch mismatch"
    );
    println!("[PASS] Buffer->Buffer rank-4 all-aligned (T=0), z-folded dispatch shape");
}

/// Exact-index (non-resize) `Image`->`Buffer` copy at a source offset that
/// reads past the source image's real physical bounds: with
/// `CvtOptions::default()` (`address_mode: None`, out-of-bounds
/// source reads denied), this must panic before any GPU work is recorded -
/// `image_to_buffer_with` asserts the source region fits the real image
/// (`ImageSlice::assert_fits_image`) up front rather than letting the
/// exact-index kernel read past the texture.
fn test_image_to_buffer_oob_denied_panics(device: &mev::Device, queue: &mut mev::Queue, cvt: &Cvt) {
    use mev::{ImageDesc, ImageExtent, ImageUsage};
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let src_image = device.new_image(ImageDesc::new(
        ImageExtent::D2(mev::Extent2::new(2, 2)),
        PixelFormat::Rgba8Unorm,
        ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
    ));
    let dst = make_readback_buffer(device, &[0u8; 16], "i2b_oob_denied_dst");

    // offset=(1,0), extent=(2,2) read against a 2x2 physical image: the x
    // axis runs 1..3, past the image's real width of 2.
    let src_slice = src_image.slice((Offset2::new(1, 0), mev::Extent2::new(2, 2)));

    let mut encoder = queue.new_command_encoder();
    let result = catch_unwind(AssertUnwindSafe(|| {
        cvt.image_to_buffer::<2>(
            &mut encoder,
            &src_slice,
            &dst,
            TexelShape::new(PixelFormat::Rgba8Unorm, mev::Extent2::new(2, 2)),
        );
    }));
    assert!(
        result.is_err(),
        "image_to_buffer must panic (before GPU work) when the source region overruns the image and address_mode is unset"
    );
    // See `test_extent_mismatch_panics`: finish/submit the still-valid
    // (empty) recording instead of dropping it, to avoid leaking its
    // command buffer/pool.
    submit_and_wait(queue, encoder.finish());
    println!(
        "[PASS] Image->Buffer exact-index panics on out-of-bounds source region (address_mode: None)"
    );
}

/// Same overrunning source region as `test_image_to_buffer_oob_denied_panics`,
/// but with `address_mode: Some([AddressMode::ClampToEdge; 2])`: must not
/// panic, must execute, and the out-of-range x=2 column (past the image's
/// real width of 2) must clamp to the image's real edge column (x=1) rather
/// than being denied. Verified against a hand-computed expected array, not
/// just "it didn't panic".
fn test_image_to_buffer_oob_clamp_wraps(device: &mev::Device, queue: &mut mev::Queue, cvt: &Cvt) {
    use mev::{ImageDesc, ImageExtent, ImageUsage};

    let src_image = device.new_image(ImageDesc::new(
        ImageExtent::D2(mev::Extent2::new(2, 2)),
        PixelFormat::Rgba8Unorm,
        ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
    ));

    // Physical 2x2 grid, row-major: p00=(0,0), p10=(1,0), p01=(0,1), p11=(1,1).
    let p00 = [1u8, 2, 3, 255];
    let p10 = [4u8, 5, 6, 255];
    let p01 = [7u8, 8, 9, 255];
    let p11 = [10u8, 11, 12, 255];
    let src_pixels: [u8; 16] = std::array::from_fn(|i| {
        let px = i / 4;
        let ch = i % 4;
        [p00, p10, p01, p11][px][ch]
    });

    let staging = device.new_buffer_init(BufferInitDesc {
        data: &src_pixels,
        usage: BufferUsage::TRANSFER_SRC | BufferUsage::HOST_WRITE,
        name: "i2b_oob_clamp_stage",
    });
    let dst = make_readback_buffer(device, &[0u8; 16], "i2b_oob_clamp_dst");

    let mut encoder = queue.new_command_encoder();
    encoder.init_image(
        PipelineStages::empty(),
        PipelineStages::TRANSFER,
        &src_image,
    );
    {
        let mut copy = encoder.copy();
        copy.copy_buffer_to_image(
            &staging,
            TexelShape::new(PixelFormat::Rgba8Unorm, mev::Extent2::new(2, 2)).with_row_stride(2 * 4),
            &src_image,
        );
        copy.barrier(PipelineStages::TRANSFER, PipelineStages::COMPUTE_SHADER);
    }

    // Same overrunning offset=(1,0), extent=(2,2) as the panic test above.
    let src_slice = src_image.slice((Offset2::new(1, 0), mev::Extent2::new(2, 2)));

    cvt.image_to_buffer_with::<2, 2>(
        &mut encoder,
        &src_slice,
        &dst,
        TexelShape::new(PixelFormat::Rgba8Unorm, mev::Extent2::new(2, 2)),
        CvtOptions::default().address_mode([AddressMode::ClampToEdge; 2]),
    );

    submit_and_wait(queue, encoder.finish());

    let mut out = [0u8; 16];
    let mut dst = dst;
    dst.read(0, &mut out).unwrap();

    // Logical (lx, ly) reads physical (1+lx, ly): lx=0 -> physical x=1
    // (real column), lx=1 -> physical x=2, clamped to the real edge x=1.
    // So both dst columns read the image's real x=1 column: row ly=0 reads
    // p10 twice, row ly=1 reads p11 twice.
    let expected_src = [p10, p10, p11, p11];
    let expected: [u8; 16] = std::array::from_fn(|i| {
        let px = i / 4;
        let ch = i % 4;
        expected_src[px][ch]
    });
    assert_eq!(
        out, expected,
        "Image->Buffer exact-index out-of-bounds clamp mismatch: {:?} != {:?}",
        out, expected
    );
    println!(
        "[PASS] Image->Buffer exact-index out-of-bounds source clamps to real image edge (address_mode: Some(ClampToEdge))"
    );
}

fn main() {
    mev::match_backend! {
        metal => { println!("Metal backend"); }
        vulkan => { println!("Vulkan backend"); }
        webgpu => { println!("WebGPU backend"); }
    }

    let instance = mev::Instance::load().expect("Failed to init graphics");

    let (device, mut queues) = instance
        .new_device(DeviceDesc {
            idx: 0,
            queues: &[0],
            features: Features::empty(),
        })
        .expect("Failed to create device");

    let mut queue = queues.remove(0);

    atomic_smoke_test(&device, &mut queue);

    let cvt = Cvt::new(&device);

    test_cross_kind_unorm8_to_uint8(&device, &mut queue, &cvt);
    test_transpose_mirror_2d(&device, &mut queue, &cvt);
    test_diagonal_read(&device, &mut queue, &cvt);
    test_splat(&device, &mut queue, &cvt);
    test_buffer_in_place(&device, &mut queue, &cvt);
    test_image_to_image(&device, &mut queue, &cvt);
    test_image_to_image_srgb_dst(&device, &mut queue, &cvt);
    test_image_to_image_srgb_dst_resample(&device, &mut queue, &cvt);
    test_buffer_to_image(&device, &mut queue, &cvt);
    test_image_to_buffer(&device, &mut queue, &cvt);
    test_extent_mismatch_panics(&device, &mut queue, &cvt);
    test_image_to_image_resample(&device, &mut queue, &cvt);
    test_buffer_to_buffer_resample(&device, &mut queue, &cvt);
    test_buffer_resample_non_nearest_filter_panics(&device, &mut queue, &cvt);
    test_tail_block_non_multiple_of_4(&device, &mut queue, &cvt);
    test_padded_dst_stride_sentinel(&device, &mut queue, &cvt);
    test_new_dst_stride_preconditions_panic(&device, &mut queue, &cvt);
    test_rank4_all_aligned_z_fold(&device, &mut queue, &cvt);
    test_image_to_buffer_oob_denied_panics(&device, &mut queue, &cvt);
    test_image_to_buffer_oob_clamp_wraps(&device, &mut queue, &cvt);

    println!("All cvt_smoke checks passed.");
}
