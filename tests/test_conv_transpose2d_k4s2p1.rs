// conv_transpose2d_k4s2p1_forward test.
//
// Matches nn.ConvTranspose2d(Cin, Cout, kernel_size=4, stride=2, padding=1), the only
// transposed-conv shape FoundationStereo uses (core/foundation_stereo.py:166).
// B=2, Cin=3, Cout=2, 4x5 -> 8x10. Fixtures come from a real PyTorch module including
// its bias, so the bias path is covered.

#![cfg(feature = "cuda")]

use dotenv::dotenv;
use std::path::PathBuf;
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;
use teeny_cuda::compiler::{compile_kernel, target::Target};
use teeny_cuda::errors::Result;
use teeny_test::cuda as testing;
use vision_rs::models::stereo::kernels::conv_transpose2d_k4s2p1::ConvTranspose2dK4s2p1Forward;

const B: usize = 2;
const CIN: usize = 3;
const COUT: usize = 2;
const IH: usize = 4;
const IW: usize = 5;
const OH: usize = IH * 2;
const OW: usize = IW * 2;
const BLOCK_CI: i32 = 4; // >= CIN, not equal: exercises the channel mask

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
fn test_conv_transpose2d_k4s2p1_forward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let x = load("conv_transpose2d_k4s2p1/x.bin");
    let w = load("conv_transpose2d_k4s2p1/w.bin");
    let bias = load("conv_transpose2d_k4s2p1/b.bin");
    let expected = load("conv_transpose2d_k4s2p1/expected.bin");
    assert_eq!(x.len(), B * CIN * IH * IW);
    assert_eq!(w.len(), CIN * COUT * 16);
    assert_eq!(bias.len(), COUT);
    assert_eq!(expected.len(), B * COUT * OH * OW);

    let mut x_buf = device.buffer::<f32>(x.len())?;
    let mut w_buf = device.buffer::<f32>(w.len())?;
    let mut b_buf = device.buffer::<f32>(bias.len())?;
    let out_buf = device.buffer::<f32>(expected.len())?;
    x_buf.to_device(&x)?;
    w_buf.to_device(&w)?;
    b_buf.to_device(&bias)?;

    let kernel = ConvTranspose2dK4s2p1Forward::<f32>::new(BLOCK_CI);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(PathBuf::from(compile_kernel(
        &kernel, &target, true, false,
    )?))?;
    let program = testing::load_program_from_ptx::<ConvTranspose2dK4s2p1Forward<f32>>(&ptx)?;

    device.launch(
        &program,
        &testing::launch_config_with_grid(B * COUT * OH * OW, &program),
        (
            x_buf.as_device_ptr(),
            w_buf.as_device_ptr(),
            b_buf.as_device_ptr(),
            out_buf.as_device_ptr(),
            CIN as i32,
            COUT as i32,
            IH as i32,
            IW as i32,
        ),
    )?;

    let mut out = vec![0.0f32; expected.len()];
    out_buf.to_host(&mut out)?;

    let mut max_abs = 0.0f32;
    for i in 0..out.len() {
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
    println!("conv_transpose2d_k4s2p1: max |gpu - torch| = {max_abs:e}");
    Ok(())
}
