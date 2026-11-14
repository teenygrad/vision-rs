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
    update_block::{UpdateBlockDims, UpdateInputs, basic_selective_multi_update_block},
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
            vec![(
                "",
                selective_conv_gru::<f32>(8, 16)(&i[0], &i[1], &[&i[2], &i[3]]),
            )]
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
        |i| vec![("", basic_motion_encoder::<f32>(dims)(&i[0], &i[1]))],
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
        |i| vec![("", disp_head::<f32>(8, 1)(&i[0]))],
        1e-5,
    )?;
    Ok(())
}

/// `BasicSelectiveMultiUpdateBlock(n_gru_layers=3)` at hidden width 8 over a
/// 12x16 / 6x8 / 3x4 pyramid: all three GRU levels, the motion encoder, `DispHead` and
/// the mask head in one graph. Checks every output, including the updated 1/4-level
/// state that also feeds the two heads.
#[test]
#[serial]
fn update_block_matches_pytorch() -> anyhow::Result<()> {
    let dims = UpdateBlockDims {
        hidden: 8,
        motion: MotionEncoderDims {
            cor_planes: 18,
            cor: 16,
            disp: 8,
            out: 8,
        },
        mask_mid: 6,
        mask_out: 4,
    };
    check_block(
        "update_block",
        "update_block",
        &[
            ("net0", [1, 8, 12, 16]),
            ("net1", [1, 8, 6, 8]),
            ("net2", [1, 8, 3, 4]),
            ("inp0", [1, 8, 12, 16]),
            ("inp1", [1, 8, 6, 8]),
            ("inp2", [1, 8, 3, 4]),
            ("corr", [1, 18, 12, 16]),
            ("disp", [1, 1, 12, 16]),
            ("att0", [1, 1, 12, 16]),
            ("att1", [1, 1, 6, 8]),
            ("att2", [1, 1, 3, 4]),
        ],
        |i| {
            let out = basic_selective_multi_update_block::<f32>(dims)(&UpdateInputs {
                net: [&i[0], &i[1], &i[2]],
                inp: [&i[3], &i[4], &i[5]],
                corr: &i[6],
                disp: &i[7],
                att: [&i[8], &i[9], &i[10]],
            });
            let [n0, n1, n2] = out.net;
            vec![
                ("net0", n0),
                ("net1", n1),
                ("net2", n2),
                ("mask", out.mask),
                ("delta_disp", out.delta_disp),
            ]
        },
        1e-5,
    )?;
    Ok(())
}

/// The update block at full width with the real `23-51-11` checkpoint weights, on inputs
/// captured from a forward pass over FoundationStereo's sample stereo pair (iterations
/// 0 and 8). The reference is the PyTorch block run in FP64 on the CPU: PyTorch on
/// CUDA defaults to TF32 convolutions, which alone is ~1e-2 off at this width, while
/// plain FP32 PyTorch lands within ~3e-5 of FP64. The fixtures are ~86 MB per
/// iteration, so
/// they live outside the repo: set `FS_REAL_FIXTURES` to the directory holding
/// `update_block_real_it{0,8}/` and run with `--ignored`.
#[test]
#[serial]
#[ignore = "needs FS_REAL_FIXTURES (real-checkpoint fixtures, not committed)"]
fn update_block_matches_real_checkpoint() -> anyhow::Result<()> {
    let root = std::env::var("FS_REAL_FIXTURES")
        .expect("set FS_REAL_FIXTURES to the real-checkpoint fixture directory");
    for it in [0, 8] {
        let dir = format!("{root}/update_block_real_it{it}");
        let shapes_txt = std::fs::read_to_string(format!("{dir}/shapes.txt"))?;
        let inputs: Vec<(String, [usize; 4])> = shapes_txt
            .lines()
            .map(|l| {
                let mut f = l.split_whitespace();
                let name = f.next().unwrap().to_string();
                let d: Vec<usize> = f.map(|v| v.parse().unwrap()).collect();
                (name, [d[0], d[1], d[2], d[3]])
            })
            .collect();
        let inputs: Vec<(&str, [usize; 4])> =
            inputs.iter().map(|(n, s)| (n.as_str(), *s)).collect();
        check_block(
            &format!("update_block_real_it{it}"),
            &dir,
            &inputs,
            |i| {
                let out = basic_selective_multi_update_block::<f32>(
                    UpdateBlockDims::foundation_stereo(),
                )(&UpdateInputs {
                    net: [&i[0], &i[1], &i[2]],
                    inp: [&i[3], &i[4], &i[5]],
                    corr: &i[6],
                    disp: &i[7],
                    att: [&i[8], &i[9], &i[10]],
                });
                let [n0, n1, n2] = out.net;
                vec![
                    ("net0", n0),
                    ("net1", n1),
                    ("net2", n2),
                    ("mask", out.mask),
                    ("delta_disp", out.delta_disp),
                ]
            },
            1e-4,
        )?;
    }
    Ok(())
}
