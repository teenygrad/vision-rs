#![cfg(feature = "cuda")]
use dotenv::dotenv;
use std::path::PathBuf;
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_cuda::compiler::{compile_kernel, target::Target};
use teeny_cuda::errors::Result;
use teeny_test::cuda as testing;
use vision_rs::models::stereo::kernels::avg_pool2d_padded::{
    AvgPool2dPaddedForward, pooled_extent,
};

const B: usize = 2;
const C: usize = 3;
// Odd extents: padding bites on every edge and column tiles end part-way.
const IH: usize = 13;
const IW: usize = 19;
const BLOCK_W: i32 = 8;

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

/// Runs one (K, S, P) configuration against its PyTorch fixture.
fn check(k: usize, s: usize, p: usize) -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;
    let oh = pooled_extent(IH, k, s, p);
    let ow = pooled_extent(IW, k, s, p);
    let x = load("avg_pool2d_padded/x.bin");
    let expected = load(&format!("avg_pool2d_padded/k{k}s{s}p{p}.bin"));
    assert_eq!(x.len(), B * C * IH * IW);
    assert_eq!(expected.len(), B * C * oh * ow);

    let mut x_buf = device.buffer::<f32>(x.len())?;
    let out_buf = device.buffer::<f32>(expected.len())?;
    x_buf.to_device(&x)?;

    let kernel = AvgPool2dPaddedForward::<f32>::new(BLOCK_W);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(PathBuf::from(compile_kernel(
        &kernel, &target, true, false,
    )?))?;
    let program = testing::load_program_from_ptx::<AvgPool2dPaddedForward<f32>>(&ptx)?;

    let tiles = ow.div_ceil(BLOCK_W as usize);
    device.launch(
        &program,
        &testing::launch_config_with_grid(B * C * oh * tiles, &program),
        (
            x_buf.as_device_ptr(),
            out_buf.as_device_ptr(),
            IH as i32,
            IW as i32,
            oh as i32,
            ow as i32,
            k as i32,
            s as i32,
            p as i32,
        ),
    )?;

    let mut out = vec![0.0f32; expected.len()];
    out_buf.to_host(&mut out)?;
    let mut max_abs = 0.0f32;
    for i in 0..out.len() {
        assert!(out[i].is_finite(), "non-finite at {i}");
        let d = (out[i] - expected[i]).abs();
        max_abs = max_abs.max(d);
        assert!(
            d < 1e-6,
            "k{k}s{s}p{p}: mismatch at {i}: gpu={} torch={}",
            out[i],
            expected[i]
        );
    }
    println!("avg_pool2d_padded k{k}s{s}p{p}: max |gpu - torch| = {max_abs:e}");
    Ok(())
}

/// FoundationStereo's `pool2x` (`core/update.py:72`).
#[test]
fn test_avg_pool2d_padded_k3s2p1_cuda() -> Result<()> {
    check(3, 2, 1)
}

/// FoundationStereo's `pool4x` (`core/update.py:75`).
#[test]
fn test_avg_pool2d_padded_k5s4p1_cuda() -> Result<()> {
    check(5, 4, 1)
}
