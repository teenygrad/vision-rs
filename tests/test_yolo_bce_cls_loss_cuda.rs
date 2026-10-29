// BCE classification loss forward and backward tests.
//
// Verifies yolo_bce_cls_loss_forward and yolo_bce_cls_loss_backward kernels
// against PyTorch-generated reference outputs (fixtures/yolo_bce_cls_loss/).
//
// Pred / target layout: [C, N] channels-first.
// Forward output:  [N] per-anchor BCE loss summed over C classes.
// Backward output: [C, N] per-element gradient w.r.t. pred logits.

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use dotenv::dotenv;
use insta::assert_debug_snapshot;
use teeny_cuda::compiler::{compile_kernel, target::{Capability, Target}};
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;

#[cfg(feature = "cuda")]
use teeny_cuda::errors::Result;
#[cfg(feature = "cuda")]
use teeny_test::cuda as testing;

const N:       usize = 32;
const C:       usize = 80;
const BLOCK_N: i32   = 32;

fn load(rel: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing {path}: {e}"));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

// ── Forward snapshot ──────────────────────────────────────────────────────────

#[test]
fn test_yolo_bce_cls_loss_forward_snapshot() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let kernel = vision_rs::models::yolo::kernels::loss::cls::YoloBceClsLossForward::<f32>::new(BLOCK_N);
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true, false)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("yolo_bce_cls_loss_forward_source", kernel.source());
    assert_debug_snapshot!("yolo_bce_cls_loss_forward_mlir", mlir.trim());
    Ok(())
}

// ── Forward CUDA ──────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_yolo_bce_cls_loss_forward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let pred     = load("yolo_bce_cls_loss/pred.bin");
    let target   = load("yolo_bce_cls_loss/target.bin");
    let expected = load("yolo_bce_cls_loss/expected.bin");

    assert_eq!(pred.len(),     C * N);
    assert_eq!(target.len(),   C * N);
    assert_eq!(expected.len(), N);

    let mut pred_buf   = device.buffer::<f32>(C * N)?;
    let mut target_buf = device.buffer::<f32>(C * N)?;
    let loss_buf       = device.buffer::<f32>(N)?;

    pred_buf.to_device(&pred)?;
    target_buf.to_device(&target)?;

    let kernel = vision_rs::models::yolo::kernels::loss::cls::YoloBceClsLossForward::<f32>::new(BLOCK_N);
    let cuda_target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &cuda_target, true, false)?)?;
    let program = testing::load_program_from_ptx::<
        vision_rs::models::yolo::kernels::loss::cls::YoloBceClsLossForward<f32>
    >(&ptx)?;

    let n_tiles = N.div_ceil(BLOCK_N as usize);
    let cfg = testing::launch_config_with_grid(n_tiles, &program);
    device.launch(&program, &cfg, (
        pred_buf.as_device_ptr()   as *mut f32,
        target_buf.as_device_ptr() as *mut f32,
        loss_buf.as_device_ptr()   as *mut f32,
        N as i32,
        C as i32,
    ))?;

    let mut loss_host = vec![0.0f32; N];
    loss_buf.to_host(&mut loss_host)?;

    for i in 0..N {
        assert!(
            (loss_host[i] - expected[i]).abs() < 1e-3,
            "forward mismatch at anchor {i}: gpu={:.6}  expected={:.6}",
            loss_host[i], expected[i]
        );
    }
    Ok(())
}

// ── Backward snapshot ─────────────────────────────────────────────────────────

#[test]
fn test_yolo_bce_cls_loss_backward_snapshot() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let kernel = vision_rs::models::yolo::kernels::loss::cls::YoloBceClsLossBackward::<f32>::new(BLOCK_N);
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true, false)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("yolo_bce_cls_loss_backward_source", kernel.source());
    assert_debug_snapshot!("yolo_bce_cls_loss_backward_mlir", mlir.trim());
    Ok(())
}

// ── Backward CUDA ─────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_yolo_bce_cls_loss_backward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let dy            = load("yolo_bce_cls_loss/dy.bin");
    let pred          = load("yolo_bce_cls_loss/pred.bin");
    let target        = load("yolo_bce_cls_loss/target.bin");
    let expected_grad = load("yolo_bce_cls_loss/expected_dpred.bin");

    assert_eq!(dy.len(),            N);
    assert_eq!(pred.len(),      C * N);
    assert_eq!(target.len(),    C * N);
    assert_eq!(expected_grad.len(), C * N);

    let mut dy_buf     = device.buffer::<f32>(N)?;
    let mut pred_buf   = device.buffer::<f32>(C * N)?;
    let mut target_buf = device.buffer::<f32>(C * N)?;
    let d_pred_buf     = device.buffer::<f32>(C * N)?;

    dy_buf.to_device(&dy)?;
    pred_buf.to_device(&pred)?;
    target_buf.to_device(&target)?;

    let kernel = vision_rs::models::yolo::kernels::loss::cls::YoloBceClsLossBackward::<f32>::new(BLOCK_N);
    let cuda_target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &cuda_target, true, false)?)?;
    let program = testing::load_program_from_ptx::<
        vision_rs::models::yolo::kernels::loss::cls::YoloBceClsLossBackward<f32>
    >(&ptx)?;

    let n_tiles = N.div_ceil(BLOCK_N as usize);
    let cfg = testing::launch_config_with_grid(n_tiles, &program);
    device.launch(&program, &cfg, (
        dy_buf.as_device_ptr()     as *mut f32,
        pred_buf.as_device_ptr()   as *mut f32,
        target_buf.as_device_ptr() as *mut f32,
        d_pred_buf.as_device_ptr() as *mut f32,
        N as i32,
        C as i32,
    ))?;

    let mut d_pred_host = vec![0.0f32; C * N];
    d_pred_buf.to_host(&mut d_pred_host)?;

    for i in 0..(C * N) {
        assert!(
            (d_pred_host[i] - expected_grad[i]).abs() < 1e-5,
            "backward mismatch at index {i} (class {}, anchor {}): gpu={:.6}  expected={:.6}",
            i / N, i % N, d_pred_host[i], expected_grad[i]
        );
    }
    Ok(())
}
