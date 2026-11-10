// bilinear_sampler_1d_forward test.
//
// The stereo special case of grid_sample: height-1 input, constant y grid, so the
// operation is a linear interpolation between two adjacent columns.
//
// Fixtures come from PyTorch's F.grid_sample via exactly the call FoundationStereo's
// core/utils/utils.py::bilinear_sampler makes (align_corners=True, padding_mode="zeros").
// N=6, C=5, W=11, K=9, with 22 of 54 taps deliberately out of range.

#![cfg(feature = "cuda")]

use dotenv::dotenv;
use std::path::PathBuf;
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;
use teeny_cuda::compiler::{compile_kernel, target::Target};
use teeny_cuda::errors::Result;
use teeny_test::cuda as testing;
use vision_rs::models::stereo::kernels::bilinear_sampler_1d::BilinearSampler1dForward;

const N: usize = 6;
const C: usize = 5;
const W: usize = 11;
const K: usize = 9;
const BLOCK_K: i32 = 16; // next pow2 >= K

fn load(rel: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing {path}: {e}"));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[test]
fn test_bilinear_sampler_1d_forward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let img = load("bilinear_sampler_1d/img.bin");
    let coords = load("bilinear_sampler_1d/coords.bin");
    let expected = load("bilinear_sampler_1d/expected.bin");

    assert_eq!(img.len(), N * C * W);
    assert_eq!(coords.len(), N * K);
    assert_eq!(expected.len(), N * C * K);

    let mut img_buf = device.buffer::<f32>(N * C * W)?;
    let mut coords_buf = device.buffer::<f32>(N * K)?;
    let out_buf = device.buffer::<f32>(N * C * K)?;
    img_buf.to_device(&img)?;
    coords_buf.to_device(&coords)?;

    let kernel = BilinearSampler1dForward::<f32>::new(BLOCK_K);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(PathBuf::from(compile_kernel(
        &kernel, &target, true, false,
    )?))?;
    let program = testing::load_program_from_ptx::<BilinearSampler1dForward<f32>>(&ptx)?;

    let cfg = testing::launch_config_with_grid(N * C, &program);
    device.launch(
        &program,
        &cfg,
        (
            img_buf.as_device_ptr(),
            coords_buf.as_device_ptr(),
            out_buf.as_device_ptr(),
            C as i32,
            W as i32,
            K as i32,
        ),
    )?;

    let mut out_host = vec![0.0f32; N * C * K];
    out_buf.to_host(&mut out_host)?;

    let mut max_abs = 0.0f32;
    for i in 0..N * C * K {
        let d = (out_host[i] - expected[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
        assert!(
            d < 1e-5,
            "mismatch at {i}: gpu={} expected={} (|d|={d})",
            out_host[i],
            expected[i]
        );
    }
    println!("bilinear_sampler_1d: max |gpu - torch| = {max_abs:e}");
    Ok(())
}
