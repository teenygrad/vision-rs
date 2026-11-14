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

use common::stereo_block::{Block, check_block, compare, fixture};
use serial_test::serial;
use vision_rs::models::stereo::geometry::{
    GeoPyramid, correlation_volume, geo_lookup, geo_pyramid,
};
use vision_rs::models::stereo::update::{
    disp_head::disp_head,
    gru::selective_conv_gru,
    motion_encoder::{MotionEncoderDims, basic_motion_encoder},
    refinement::{StepInputs, refinement_step},
    update_block::{UpdateBlockDims, UpdateInputs, basic_selective_multi_update_block},
    upsample::upsample_disp,
};

/// `SelectiveConvGRU(hidden 8, input 8 + 8)` on a 6x10 map.
#[test]
#[serial]
fn selective_conv_gru_matches_pytorch() -> anyhow::Result<()> {
    check_block(
        "selective_conv_gru",
        "selective_conv_gru",
        &[
            ("att", &[1, 1, 6, 10]),
            ("h", &[1, 8, 6, 10]),
            ("x1", &[1, 8, 6, 10]),
            ("x2", &[1, 8, 6, 10]),
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
        &[("disp", &[1, 1, 6, 10]), ("corr", &[1, 18, 6, 10])],
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
        &[("x", &[1, 8, 9, 11])],
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
            ("net0", &[1, 8, 12, 16]),
            ("net1", &[1, 8, 6, 8]),
            ("net2", &[1, 8, 3, 4]),
            ("inp0", &[1, 8, 12, 16]),
            ("inp1", &[1, 8, 6, 8]),
            ("inp2", &[1, 8, 3, 4]),
            ("corr", &[1, 18, 12, 16]),
            ("disp", &[1, 1, 12, 16]),
            ("att0", &[1, 1, 12, 16]),
            ("att1", &[1, 1, 6, 8]),
            ("att2", &[1, 1, 3, 4]),
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
        let inputs: Vec<(String, Vec<usize>)> = shapes_txt
            .lines()
            .map(|l| {
                let mut f = l.split_whitespace();
                let name = f.next().unwrap().to_string();
                (name, f.map(|v| v.parse().unwrap()).collect())
            })
            .collect();
        let inputs: Vec<(&str, &[usize])> = inputs
            .iter()
            .map(|(n, s)| (n.as_str(), s.as_slice()))
            .collect();
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

/// The geometry-encoding lookup, from feature maps to the encoder's input:
/// `correlation_volume`, then `geo_pyramid` (two levels), then `geo_lookup` (radius 2),
/// against `Combined_Geo_Encoding_Volume`. Features 12 channels, a 3-channel cost
/// volume over 12 disparities, 3x10 pixels; the disparities put about a third of the
/// cost-volume samples and half the correlation samples outside their axis, so the
/// zero padding is exercised at both ends.
#[test]
#[serial]
fn geo_lookup_matches_pytorch() -> anyhow::Result<()> {
    check_block(
        "geo_lookup",
        "geo_lookup",
        &[
            ("fmap1", &[1, 12, 3, 10]),
            ("fmap2", &[1, 12, 3, 10]),
            ("geo", &[1, 3, 12, 3, 10]),
            ("disp", &[1, 1, 3, 10]),
        ],
        |i| {
            let corr = correlation_volume::<f32>(&i[0], &i[1], 16);
            let pyramid = geo_pyramid::<f32>(&i[2], &corr, 2);
            vec![("", geo_lookup::<f32>(&pyramid, &i[3], 2))]
        },
        1e-5,
    )?;
    Ok(())
}

/// Four iterations of FoundationStereo's refinement loop, against
/// `FoundationStereo.forward`'s loop over the real `Combined_Geo_Encoding_Volume` and
/// `BasicSelectiveMultiUpdateBlock` (hidden 8; 3-channel cost volume over 12
/// disparities; radius 2; 12x16 / 6x8 / 3x4 pyramid).
///
/// Two graphs, as the full model will run them: an init graph (correlation and the
/// geometry pyramid) run once, and a step graph (lookup, update block, `disp + delta`)
/// run once per iteration with its hidden states and disparity fed back. Checks the
/// disparity after every iteration, then the final hidden states and mask.
#[test]
#[serial]
fn refinement_loop_matches_pytorch() -> anyhow::Result<()> {
    const DIR: &str = "refinement_loop";
    const ITERS: usize = 4;
    let f = |n: &str| fixture(DIR, &format!("{n}.bin"));
    let device = teeny_test::model::runtime::open()?;

    // Init: correlation volume and pyramid, once per image pair.
    let mut init = Block::new(
        &device,
        "refinement_loop/init",
        DIR,
        &[
            ("fmap1", &[1, 12, 12, 16]),
            ("fmap2", &[1, 12, 12, 16]),
            ("geo", &[1, 3, 12, 12, 16]),
        ],
        |i| {
            let corr = correlation_volume::<f32>(&i[0], &i[1], 16);
            let p = geo_pyramid::<f32>(&i[2], &corr, 2);
            vec![
                ("corr0", p.corr[0].clone()),
                ("geo1", p.geo[1].clone()),
                ("corr1", p.corr[1].clone()),
            ]
        },
    )?;
    let geo0 = f("geo");
    let pyr = init.run(&[&f("fmap1"), &f("fmap2"), &geo0])?;
    let (corr0, geo1, corr1) = (&pyr[0], &pyr[1], &pyr[2]);

    // Step: one iteration, loaded once.
    let dims = UpdateBlockDims {
        hidden: 8,
        motion: MotionEncoderDims {
            cor_planes: 40,
            cor: 16,
            disp: 8,
            out: 8,
        },
        mask_mid: 6,
        mask_out: 4,
    };
    let mut step = Block::new(
        &device,
        "refinement_loop/step",
        DIR,
        &[
            ("net0", &[1, 8, 12, 16]),
            ("net1", &[1, 8, 6, 8]),
            ("net2", &[1, 8, 3, 4]),
            ("inp0", &[1, 8, 12, 16]),
            ("inp1", &[1, 8, 6, 8]),
            ("inp2", &[1, 8, 3, 4]),
            ("att0", &[1, 1, 12, 16]),
            ("att1", &[1, 1, 6, 8]),
            ("att2", &[1, 1, 3, 4]),
            ("geo0", &[1, 3, 12, 12, 16]),
            ("geo1", &[1, 3, 6, 12, 16]),
            ("corr0", &[1, 12, 16, 16]),
            ("corr1", &[1, 12, 16, 8]),
            ("disp", &[1, 1, 12, 16]),
        ],
        |i| {
            let pyramid = GeoPyramid {
                geo: vec![i[9].clone(), i[10].clone()],
                corr: vec![i[11].clone(), i[12].clone()],
            };
            let out = refinement_step::<f32>(dims, 2)(&StepInputs {
                net: [&i[0], &i[1], &i[2]],
                inp: [&i[3], &i[4], &i[5]],
                att: [&i[6], &i[7], &i[8]],
                pyramid: &pyramid,
                disp: &i[13],
            });
            let [n0, n1, n2] = out.net;
            vec![
                ("net0", n0),
                ("net1", n1),
                ("net2", n2),
                ("mask", out.mask),
                ("disp", out.disp),
            ]
        },
    )?;

    let (inp, att) = (
        [f("inp0"), f("inp1"), f("inp2")],
        [f("att0"), f("att1"), f("att2")],
    );
    let mut net = vec![f("net0"), f("net1"), f("net2")];
    let mut disp = f("disp");
    let mut mask = Vec::new();
    for it in 0..ITERS {
        let out = step.run(&[
            &net[0], &net[1], &net[2], &inp[0], &inp[1], &inp[2], &att[0], &att[1], &att[2], &geo0,
            geo1, corr0, corr1, &disp,
        ])?;
        let mut out = out.into_iter();
        net = vec![
            out.next().unwrap(),
            out.next().unwrap(),
            out.next().unwrap(),
        ];
        mask = out.next().unwrap();
        disp = out.next().unwrap();
        compare(
            &format!("refinement_loop/disp_it{it}"),
            &disp,
            &f(&format!("expected_disp_it{it}")),
            1e-5,
        );
    }
    for (n, got) in ["net0", "net1", "net2"].iter().zip(&net) {
        compare(
            &format!("refinement_loop/{n}"),
            got,
            &f(&format!("expected_{n}")),
            1e-5,
        );
    }
    compare("refinement_loop/mask", &mask, &f("expected_mask"), 1e-5);
    Ok(())
}

/// `FoundationStereo.upsample_disp` at reduced width (`c` 4 instead of 32): a 6x8
/// disparity, the mask features and a 12x16 stem map, to a 24x32 disparity. Covers
/// both transposed convs (with and without bias), LeakyReLU, the channel softmax and
/// the fused convex upsampling, including its zero-padded border.
#[test]
#[serial]
fn upsample_disp_matches_pytorch() -> anyhow::Result<()> {
    check_block(
        "upsample_disp",
        "upsample_disp",
        &[
            ("disp", &[1, 1, 6, 8]),
            ("mask", &[1, 4, 6, 8]),
            ("stem", &[1, 4, 12, 16]),
        ],
        |i| vec![("", upsample_disp::<f32>(4)(&i[0], &i[1], &i[2]))],
        1e-5,
    )?;
    Ok(())
}
