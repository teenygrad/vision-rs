// correlation_forward test.
//
// Fused L2-normalise + all-pairs cosine correlation, matching FoundationStereo
// core/geometry.py:74 exactly:
//
//   torch.einsum('aijk,aijh->ajkh', F.normalize(fmap1, dim=1), F.normalize(fmap2, dim=1))
//
// B=2, C=12, H=3, W1=5, W2=7. Fixtures include one all-zero feature column in each
// feature map, which exercises F.normalize's eps=1e-12 clamp (PyTorch yields 0 there,
// not NaN).

#![cfg(feature = "cuda")]

use dotenv::dotenv;
use std::path::PathBuf;
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;
use teeny_cuda::compiler::{compile_kernel, target::Target};
use teeny_cuda::errors::Result;
use teeny_test::cuda as testing;
use vision_rs::models::stereo::kernels::correlation::CorrelationForward;
use vision_rs::models::stereo::kernels::l2_normalize::L2NormalizeForward;

const B: usize = 2;
const C: usize = 12;
const H: usize = 3;
const W1: usize = 5;
const W2: usize = 7;
const BLOCK_C: i32 = 16; // next pow2 >= C

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
fn test_correlation_forward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let fmap1 = load("correlation/n1.bin");
    let fmap2 = load("correlation/n2.bin");
    let expected = load("correlation/expected.bin");

    assert_eq!(fmap1.len(), B * C * H * W1);
    assert_eq!(fmap2.len(), B * C * H * W2);
    assert_eq!(expected.len(), B * H * W1 * W2);

    let mut f1_buf = device.buffer::<f32>(B * C * H * W1)?;
    let mut f2_buf = device.buffer::<f32>(B * C * H * W2)?;
    let out_buf = device.buffer::<f32>(B * H * W1 * W2)?;
    f1_buf.to_device(&fmap1)?;
    f2_buf.to_device(&fmap2)?;

    let kernel = CorrelationForward::<f32>::new(BLOCK_C);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(PathBuf::from(compile_kernel(
        &kernel, &target, true, false,
    )?))?;
    let program = testing::load_program_from_ptx::<CorrelationForward<f32>>(&ptx)?;

    let cfg = testing::launch_config_with_grid(B * H * W1, &program);
    device.launch(
        &program,
        &cfg,
        (
            f1_buf.as_device_ptr(),
            f2_buf.as_device_ptr(),
            out_buf.as_device_ptr(),
            C as i32,
            H as i32,
            W1 as i32,
            W2 as i32,
        ),
    )?;

    let mut out_host = vec![0.0f32; B * H * W1 * W2];
    out_buf.to_host(&mut out_host)?;

    let mut max_abs = 0.0f32;
    for i in 0..B * H * W1 * W2 {
        assert!(
            out_host[i].is_finite(),
            "non-finite at {i}: {}",
            out_host[i]
        );
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
    println!("correlation: max |gpu - torch| = {max_abs:e}");
    Ok(())
}

/// The full FoundationStereo expression: l2_normalize on both maps, then correlation.
/// Runs the two kernels back to back on device, output of the first feeding the second.
#[test]
fn test_l2_normalize_then_correlation_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let fmap1 = load("correlation/fmap1.bin");
    let fmap2 = load("correlation/fmap2.bin");
    let n1_ref = load("correlation/n1.bin");
    let n2_ref = load("correlation/n2.bin");
    let expected = load("correlation/expected.bin");

    let target = Target::new(env.capability);

    // --- pass 1: l2_normalize along channels, S = H*W ---
    let norm_k = L2NormalizeForward::<f32>::new(BLOCK_C);
    let norm_ptx = std::fs::read(PathBuf::from(compile_kernel(
        &norm_k, &target, true, false,
    )?))?;
    let norm_prog = testing::load_program_from_ptx::<L2NormalizeForward<f32>>(&norm_ptx)?;

    let mut f1_buf = device.buffer::<f32>(B * C * H * W1)?;
    let mut f2_buf = device.buffer::<f32>(B * C * H * W2)?;
    let n1_buf = device.buffer::<f32>(B * C * H * W1)?;
    let n2_buf = device.buffer::<f32>(B * C * H * W2)?;
    f1_buf.to_device(&fmap1)?;
    f2_buf.to_device(&fmap2)?;

    let s1 = (H * W1) as i32;
    device.launch(
        &norm_prog,
        &testing::launch_config_with_grid(B * H * W1, &norm_prog),
        (f1_buf.as_device_ptr(), n1_buf.as_device_ptr(), C as i32, s1),
    )?;
    let s2 = (H * W2) as i32;
    device.launch(
        &norm_prog,
        &testing::launch_config_with_grid(B * H * W2, &norm_prog),
        (f2_buf.as_device_ptr(), n2_buf.as_device_ptr(), C as i32, s2),
    )?;

    let mut n1_host = vec![0.0f32; B * C * H * W1];
    let mut n2_host = vec![0.0f32; B * C * H * W2];
    n1_buf.to_host(&mut n1_host)?;
    n2_buf.to_host(&mut n2_host)?;
    for i in 0..n1_host.len() {
        assert!(
            (n1_host[i] - n1_ref[i]).abs() < 1e-6,
            "normalize mismatch at {i}: {} vs {}",
            n1_host[i],
            n1_ref[i]
        );
    }
    for i in 0..n2_host.len() {
        assert!(
            (n2_host[i] - n2_ref[i]).abs() < 1e-6,
            "normalize mismatch at {i}"
        );
    }

    // --- pass 2: correlation on the normalised maps ---
    let corr_k = CorrelationForward::<f32>::new(BLOCK_C);
    let corr_ptx = std::fs::read(PathBuf::from(compile_kernel(
        &corr_k, &target, true, false,
    )?))?;
    let corr_prog = testing::load_program_from_ptx::<CorrelationForward<f32>>(&corr_ptx)?;
    let out_buf = device.buffer::<f32>(B * H * W1 * W2)?;
    device.launch(
        &corr_prog,
        &testing::launch_config_with_grid(B * H * W1, &corr_prog),
        (
            n1_buf.as_device_ptr(),
            n2_buf.as_device_ptr(),
            out_buf.as_device_ptr(),
            C as i32,
            H as i32,
            W1 as i32,
            W2 as i32,
        ),
    )?;

    let mut out_host = vec![0.0f32; B * H * W1 * W2];
    out_buf.to_host(&mut out_host)?;

    let mut max_abs = 0.0f32;
    for i in 0..out_host.len() {
        assert!(out_host[i].is_finite(), "non-finite at {i}");
        let d = (out_host[i] - expected[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
        assert!(
            d < 1e-5,
            "mismatch at {i}: gpu={} torch={}",
            out_host[i],
            expected[i]
        );
    }
    println!("l2_normalize + correlation: max |gpu - torch| = {max_abs:e}");
    Ok(())
}
