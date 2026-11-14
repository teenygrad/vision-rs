// interpolate_bilinear_forward test.
//
// Matches F.interpolate(x, size=(OH, OW), mode='bilinear') for BOTH align_corners
// settings, which FoundationStereo uses at different sites. The kernel itself is
// align_corners-agnostic: the caller supplies src = dst * scale + offset.
//
// B=2, C=3, 4x5 -> 7x9: an upsample with a non-integer, non-matching ratio on both
// axes, so the weights are never trivially 0 or 1.

#![cfg(feature = "cuda")]

use dotenv::dotenv;
use std::path::PathBuf;
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;
use teeny_cuda::compiler::{compile_kernel, target::Target};
use teeny_cuda::errors::Result;
use teeny_test::cuda as testing;
use vision_rs::models::stereo::kernels::interpolate_bilinear::InterpolateBilinearForward;

const B: usize = 2;
const C: usize = 3;
const IH: usize = 4;
const IW: usize = 5;
const OH: usize = 7;
const OW: usize = 9;
const BLOCK_P: i32 = 4; // OW = 9: three column tiles per row, the last partial (tail mask)

fn load(rel: &str) -> Vec<f32> {
    let p = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    std::fs::read(&p)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// `align_corners=true` → scale = (IN-1)/(OUT-1), offset = 0.
/// `align_corners=false` → scale = IN/OUT, offset = 0.5*scale - 0.5.
fn mapping(ins: usize, outs: usize, align_corners: bool) -> (f32, f32) {
    if align_corners {
        let s = (ins as f32 - 1.0) / (outs as f32 - 1.0);
        (s, 0.0)
    } else {
        let s = ins as f32 / outs as f32;
        (s, 0.5 * s - 0.5)
    }
}

fn run(align_corners: bool, fixture: &str) -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let x = load("interpolate_bilinear/x.bin");
    let expected = load(fixture);
    assert_eq!(x.len(), B * C * IH * IW);
    assert_eq!(expected.len(), B * C * OH * OW);

    let mut x_buf = device.buffer::<f32>(B * C * IH * IW)?;
    let out_buf = device.buffer::<f32>(B * C * OH * OW)?;
    x_buf.to_device(&x)?;

    let kernel = InterpolateBilinearForward::<f32>::new(BLOCK_P);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(PathBuf::from(compile_kernel(
        &kernel, &target, true, false,
    )?))?;
    let program = testing::load_program_from_ptx::<InterpolateBilinearForward<f32>>(&ptx)?;

    let (sy, oy) = mapping(IH, OH, align_corners);
    let (sx, ox) = mapping(IW, OW, align_corners);
    let tiles = OH * OW.div_ceil(BLOCK_P as usize);
    device.launch(
        &program,
        &testing::launch_config_with_grid(B * C * tiles, &program),
        (
            x_buf.as_device_ptr(),
            out_buf.as_device_ptr(),
            C as i32,
            IH as i32,
            IW as i32,
            OH as i32,
            OW as i32,
            sy,
            oy,
            sx,
            ox,
        ),
    )?;

    let mut out = vec![0.0f32; B * C * OH * OW];
    out_buf.to_host(&mut out)?;

    let mut max_abs = 0.0f32;
    for i in 0..out.len() {
        let d = (out[i] - expected[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
        assert!(
            d < 1e-5,
            "align_corners={align_corners} mismatch at {i}: gpu={} torch={}",
            out[i],
            expected[i]
        );
    }
    println!(
        "interpolate_bilinear (align_corners={align_corners}): max |gpu - torch| = {max_abs:e}"
    );
    Ok(())
}

#[test]
fn test_interpolate_bilinear_align_corners_true_cuda() -> Result<()> {
    run(true, "interpolate_bilinear/expected_ac_true.bin")
}

#[test]
fn test_interpolate_bilinear_align_corners_false_cuda() -> Result<()> {
    run(false, "interpolate_bilinear/expected_ac_false.bin")
}
