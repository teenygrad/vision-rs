#![cfg(feature = "cuda")]
use dotenv::dotenv;
use std::path::PathBuf;
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;
use teeny_cuda::compiler::{compile_kernel, target::Target};
use teeny_cuda::errors::Result;
use teeny_test::cuda as testing;
use vision_rs::models::stereo::kernels::l2_normalize::L2NormalizeForward;

const B: usize = 2;
const C: usize = 12;
const S: usize = 15; // H*W1 = 3*5
const BLOCK_C: i32 = 16;

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
fn test_l2_normalize_forward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;
    let x = load("correlation/fmap1.bin");
    let expected = load("correlation/n1.bin");
    assert_eq!(x.len(), B * C * S);

    let mut x_buf = device.buffer::<f32>(B * C * S)?;
    let out_buf = device.buffer::<f32>(B * C * S)?;
    x_buf.to_device(&x)?;

    let kernel = L2NormalizeForward::<f32>::new(BLOCK_C);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(PathBuf::from(compile_kernel(
        &kernel, &target, true, false,
    )?))?;
    let program = testing::load_program_from_ptx::<L2NormalizeForward<f32>>(&ptx)?;

    device.launch(
        &program,
        &testing::launch_config_with_grid(B * S, &program),
        (
            x_buf.as_device_ptr(),
            out_buf.as_device_ptr(),
            C as i32,
            S as i32,
        ),
    )?;

    let mut out = vec![0.0f32; B * C * S];
    out_buf.to_host(&mut out)?;
    let mut max_abs = 0.0f32;
    for i in 0..out.len() {
        assert!(out[i].is_finite(), "non-finite at {i}");
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
    println!("l2_normalize: max |gpu - torch| = {max_abs:e}");
    Ok(())
}
