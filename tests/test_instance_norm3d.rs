#![cfg(feature = "cuda")]
use dotenv::dotenv;
use std::path::PathBuf;
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_cuda::compiler::{compile_kernel, target::Target};
use teeny_cuda::errors::Result;
use teeny_test::cuda as testing;
use vision_rs::models::stereo::kernels::instance_norm3d::InstanceNorm3dForward;

const B: usize = 2;
const C: usize = 3;
const N: usize = 140; // D*H*W = 4*5*7: two full 64-wide chunks and a partial one
const BLOCK: i32 = 64;

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
fn test_instance_norm3d_forward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;
    let x = load("instance_norm3d/x.bin");
    let expected = load("instance_norm3d/expected.bin");
    assert_eq!(x.len(), B * C * N);

    let mut x_buf = device.buffer::<f32>(B * C * N)?;
    let out_buf = device.buffer::<f32>(B * C * N)?;
    x_buf.to_device(&x)?;

    let kernel = InstanceNorm3dForward::<f32>::new(BLOCK);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(PathBuf::from(compile_kernel(
        &kernel, &target, true, false,
    )?))?;
    let program = testing::load_program_from_ptx::<InstanceNorm3dForward<f32>>(&ptx)?;

    device.launch(
        &program,
        &testing::launch_config_with_grid(B * C, &program),
        (x_buf.as_device_ptr(), out_buf.as_device_ptr(), N as i32),
    )?;

    let mut out = vec![0.0f32; B * C * N];
    out_buf.to_host(&mut out)?;
    let mut max_abs = 0.0f32;
    for i in 0..out.len() {
        assert!(out[i].is_finite(), "non-finite at {i}");
        let d = (out[i] - expected[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
        assert!(
            d < 1e-5,
            "mismatch at {i}: gpu={} torch={}",
            out[i],
            expected[i]
        );
    }
    println!("instance_norm3d: max |gpu - torch| = {max_abs:e}");
    Ok(())
}
