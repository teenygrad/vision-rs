// unfold3x3_forward test.
//
// Matches F.unfold(x, kernel_size=3, dilation=1, padding=1) reshaped to [B, C*9, H, W],
// as used by FoundationStereo's convex upsampling (core/submodule.py:459).
// B=2, C=3, H=5, W=7. W is deliberately not a multiple of BLOCK_W so the tail mask runs.

#![cfg(feature = "cuda")]

use dotenv::dotenv;
use std::path::PathBuf;
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;
use teeny_cuda::compiler::{compile_kernel, target::Target};
use teeny_cuda::errors::Result;
use teeny_test::cuda as testing;
use vision_rs::models::stereo::kernels::unfold3x3::Unfold3x3Forward;

const B: usize = 2;
const C: usize = 3;
const H: usize = 5;
const W: usize = 7;
const BLOCK_W: i32 = 8; // H*W = 35 is not a multiple of 8: exercises tiling and the tail mask

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

#[test]
fn test_unfold3x3_forward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let x = load("unfold3x3/x.bin");
    let expected = load("unfold3x3/expected.bin");
    assert_eq!(x.len(), B * C * H * W);
    assert_eq!(expected.len(), B * C * 9 * H * W);

    let mut x_buf = device.buffer::<f32>(B * C * H * W)?;
    let out_buf = device.buffer::<f32>(B * C * 9 * H * W)?;
    x_buf.to_device(&x)?;

    let kernel = Unfold3x3Forward::<f32>::new(BLOCK_W);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(PathBuf::from(compile_kernel(
        &kernel, &target, true, false,
    )?))?;
    let program = testing::load_program_from_ptx::<Unfold3x3Forward<f32>>(&ptx)?;

    // Lanes cover flattened spatial positions; one CTA per (b, c, tap, tile).
    let tiles = (H * W).div_ceil(BLOCK_W as usize);
    device.launch(
        &program,
        &testing::launch_config_with_grid(B * C * 9 * tiles, &program),
        (
            x_buf.as_device_ptr(),
            out_buf.as_device_ptr(),
            C as i32,
            H as i32,
            W as i32,
        ),
    )?;

    let mut out = vec![0.0f32; B * C * 9 * H * W];
    out_buf.to_host(&mut out)?;

    let mut max_abs = 0.0f32;
    for i in 0..out.len() {
        let d = (out[i] - expected[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
        assert!(
            d < 1e-6,
            "mismatch at {i}: gpu={} torch={}",
            out[i],
            expected[i]
        );
    }
    println!("unfold3x3: max |gpu - torch| = {max_abs:e}");
    Ok(())
}
