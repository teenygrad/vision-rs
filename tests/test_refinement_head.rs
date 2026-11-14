// FoundationStereo refinement-head blocks (`core/update.py`), each traced into a graph,
// compiled with `RuntimeModel`, loaded from PyTorch weights by `state_dict` key and
// compared against the real PyTorch module.
//
// Blocks run at reduced width so the committed weights stay small; structure is
// identical to the full model's. Fixtures use seeded random weights, scaled up where the
// default init would leave a branch negligible.
//
// Runs on whichever backend teeny-runtime was built for: CUDA, or RISC-V under qemu
// (`--no-default-features --features qemu`). The RISC-V runtime looks for `qemu-riscv64`
// on PATH; with only the static build installed, set
// `TEENYC_QEMU_RISCV64=/usr/bin/qemu-riscv64-static`.

#![cfg(any(feature = "cuda", feature = "qemu"))]

mod common;

use common::stereo_block::check_block;
use serial_test::serial;
use vision_rs::models::stereo::update::{
    disp_head::disp_head,
    gru::selective_conv_gru,
    motion_encoder::{MotionEncoderDims, basic_motion_encoder},
};

/// `SelectiveConvGRU(hidden 8, input 8 + 8)` on a 6x10 map.
#[test]
#[serial]
fn selective_conv_gru_matches_pytorch() -> anyhow::Result<()> {
    check_block(
        "selective_conv_gru",
        "selective_conv_gru",
        &[
            ("att", [1, 1, 6, 10]),
            ("h", [1, 8, 6, 10]),
            ("x1", [1, 8, 6, 10]),
            ("x2", [1, 8, 6, 10]),
        ],
        |i| {
            selective_conv_gru::<f32>(8, 16)(&i[0], &i[1], &[&i[2], &i[3]]);
        },
        1e-5,
    )?;
    Ok(())
}

/// `BasicMotionEncoder` with `cor_planes 18, cor 32, disp 16, out 24` on a 6x10 map.
#[test]
#[serial]
fn basic_motion_encoder_matches_pytorch() -> anyhow::Result<()> {
    let dims = MotionEncoderDims {
        cor_planes: 18,
        cor: 32,
        disp: 16,
        out: 24,
    };
    check_block(
        "basic_motion_encoder",
        "basic_motion_encoder",
        &[("disp", [1, 1, 6, 10]), ("corr", [1, 18, 6, 10])],
        |i| {
            basic_motion_encoder::<f32>(dims)(&i[0], &i[1]);
        },
        1e-5,
    )?;
    Ok(())
}

/// `DispHead(input_dim 8, output_dim 1)`, including two `EdgeNextConvEncoder`s, on a
/// 9x11 map (larger than the 7x7 depthwise kernel in both directions).
#[test]
#[serial]
fn disp_head_matches_pytorch() -> anyhow::Result<()> {
    check_block(
        "disp_head",
        "disp_head",
        &[("x", [1, 8, 9, 11])],
        |i| {
            disp_head::<f32>(8, 1)(&i[0]);
        },
        1e-5,
    )?;
    Ok(())
}
