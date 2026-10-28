/*
 * Copyright 2026 Teenygrad
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


// C3k2 backward CUDA test.
//
// Parameters: C3k2(32, 64, n=1, c3k=False, shortcut=False, e=0.25)
//   cv1:    Conv(32→32, 1×1) → BN → SiLU
//   m0_cv1: Conv(16→8,  3×3) → BN → SiLU
//   m0_cv2: Conv(8→16,  3×3) → BN → SiLU
//   cat([y0, y1, m0_out]) → 48 channels
//   cv2:    Conv(48→64, 1×1) → BN → SiLU
//
// The test loads pre-saved forward intermediates (pre_bn_nc, bn_mean, bn_rstd,
// pre_silu_nc for each stage) and runs the backward kernels in reverse order,
// comparing the final dx against PyTorch autograd reference (dx_expected.bin).
//
// Backward sequence (reverse):
//   dy (NCHW) → NC
//   cv2:    SiluBackward → BatchNormBackward → Conv2dBackwardDx → d_cat (NC)
//   cat:    ChannelCatBackward × 3 → [d_y0, d_y1_cat, d_m0out] (NC)
//   m0_cv2: SiluBackward → BatchNormBackward → Conv2dBackwardDx → d_m0cv1_silu (NCHW→NC)
//   m0_cv1: SiluBackward → BatchNormBackward → Conv2dBackwardDx → d_y1_bottleneck (NC)
//   VectorAdd(d_y1_cat, d_y1_bottleneck) → d_y1
//   ChannelChunkBackward(d_y0 → d_cv1_silu ch 0..16, d_y1 → d_cv1_silu ch 16..32)
//   cv1:    SiluBackward → BatchNormBackward → Conv2dBackwardDx → dx (NCHW)
//   Compare dx with dx_expected.bin.

#[cfg(feature = "cuda")]
mod cuda {
    use dotenv::dotenv;
    use serial_test::serial;
    use teeny_cuda::compiler::{compile_kernel, target::Target};
    use teeny_core::device::{Device, buffer::Buffer};
    use teeny_cuda::{device::CudaLaunchConfig, errors::Result};
    use teeny_test::cuda as testing;

    const BATCH: usize = 2;
    const H: usize = 4;
    const W: usize = 4;
    const C_IN: usize = 32;
    const C_OUT: usize = 64;
    const C: usize = 16;      // round(64 * 0.25)
    const C_INNER: usize = 8; // round(16 * 0.5)
    const C_CAT: usize = 3 * C;

    const N_SPATIAL: usize = BATCH * H * W; // 32

    const N_INPUT: usize = N_SPATIAL * C_IN;    // 1024
    const N_CV1: usize = N_SPATIAL * 2 * C;     // 1024  (32 channels)
    const N_HALF: usize = N_SPATIAL * C;         //  512  (16 channels)
    const N_M0CV1: usize = N_SPATIAL * C_INNER;  //  256  (8 channels)
    const N_M0CV2: usize = N_SPATIAL * C;         //  512  (16 channels)
    const N_CAT: usize = N_SPATIAL * C_CAT;      // 1536  (48 channels)
    const N_CV2: usize = N_SPATIAL * C_OUT;      // 2048  (64 channels)

    const BLOCK_OW: i32 = 4;
    const BLOCK_BN: i32 = 128;
    const BLOCK_SILU: i32 = 128;
    const BLOCK_CH: i32 = 128;
    const BLOCK_VADD: i32 = 128;
    const TOL: f32 = 1e-3;

    fn load(name: &str) -> Vec<f32> {
        let path = format!(
            "{}/tests/fixtures/c3k2_yolo26/backward/{}",
            env!("CARGO_MANIFEST_DIR"),
            name
        );
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    fn nchw_to_nc(src: &[f32], b: usize, c: usize, h: usize, w: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; b * h * w * c];
        for bi in 0..b {
            for ci in 0..c {
                for hi in 0..h {
                    for wi in 0..w {
                        let ni = bi * h * w + hi * w + wi;
                        out[ni * c + ci] = src[bi * c * h * w + ci * h * w + hi * w + wi];
                    }
                }
            }
        }
        out
    }

    fn nc_to_nchw(src: &[f32], b: usize, c: usize, h: usize, w: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; b * c * h * w];
        for bi in 0..b {
            for ci in 0..c {
                for hi in 0..h {
                    for wi in 0..w {
                        let ni = bi * h * w + hi * w + wi;
                        out[bi * c * h * w + ci * h * w + hi * w + wi] = src[ni * c + ci];
                    }
                }
            }
        }
        out
    }

    #[test]
    #[serial]
    fn test_c3k2_backward_cuda() -> Result<()> {
        dotenv().ok();
        let env = testing::setup_cuda_env()?;
        let device = env.device;
        let target = Target::new(env.capability);

        // ── Load forward intermediates ────────────────────────────────────────
        let dy_nchw        = load("dy.bin");               // [2,64,4,4]
        let dx_expected    = load("dx_expected.bin");      // [2,32,4,4]

        // Per-stage pre-BN (x to BN) and pre-SiLU (output of BN, input to SiLU),
        // saved as NC: shape (B*H*W, C) = (32, C)
        let cv1_pre_bn_nc   = load("cv1_pre_bn_nc.bin");   // [32,32]
        let cv1_bn_mean     = load("cv1_bn_mean.bin");      // [32]
        let cv1_bn_rstd     = load("cv1_bn_rstd.bin");      // [32]
        let cv1_pre_silu_nc = load("cv1_pre_silu_nc.bin"); // [32,32]
        let cv1_bn_w        = load("cv1_bn_w.bin");
        let cv1_conv_w      = load("cv1_conv_w.bin");       // [32,32,1,1]

        let m0cv1_pre_bn_nc   = load("m0_cv1_pre_bn_nc.bin");   // [32,8]
        let m0cv1_bn_mean     = load("m0_cv1_bn_mean.bin");
        let m0cv1_bn_rstd     = load("m0_cv1_bn_rstd.bin");
        let m0cv1_pre_silu_nc = load("m0_cv1_pre_silu_nc.bin");
        let m0cv1_bn_w        = load("m0_cv1_bn_w.bin");
        let m0cv1_conv_w      = load("m0_cv1_conv_w.bin");      // [8,16,3,3]

        let m0cv2_pre_bn_nc   = load("m0_cv2_pre_bn_nc.bin");   // [32,16]
        let m0cv2_bn_mean     = load("m0_cv2_bn_mean.bin");
        let m0cv2_bn_rstd     = load("m0_cv2_bn_rstd.bin");
        let m0cv2_pre_silu_nc = load("m0_cv2_pre_silu_nc.bin");
        let m0cv2_bn_w        = load("m0_cv2_bn_w.bin");
        let m0cv2_conv_w      = load("m0_cv2_conv_w.bin");      // [16,8,3,3]

        let cv2_pre_bn_nc   = load("cv2_pre_bn_nc.bin");   // [32,64]
        let cv2_bn_mean     = load("cv2_bn_mean.bin");
        let cv2_bn_rstd     = load("cv2_bn_rstd.bin");
        let cv2_pre_silu_nc = load("cv2_pre_silu_nc.bin");
        let cv2_bn_w        = load("cv2_bn_w.bin");
        let cv2_conv_w      = load("cv2_conv_w.bin");       // [64,48,1,1]

        // ── Compile kernels ───────────────────────────────────────────────────
        let silu_bwd_ptx = std::fs::read(compile_kernel(
            &teeny_kernels::nn::activation::sigmoid::SiluBackward::<f32>::new(BLOCK_SILU),
            &target, true, false
        )?)?;
        let silu_bwd = testing::load_program_from_ptx::<
            teeny_kernels::nn::activation::sigmoid::SiluBackward<f32>,
        >(&silu_bwd_ptx)?;

        let bn_bwd_ptx = std::fs::read(compile_kernel(
            &teeny_kernels::nn::norm::batchnorm::BatchNormBackward::<f32>::new(BLOCK_BN),
            &target, true, false
        )?)?;
        let bn_bwd = testing::load_program_from_ptx::<
            teeny_kernels::nn::norm::batchnorm::BatchNormBackward<f32>,
        >(&bn_bwd_ptx)?;

        // Conv1 backward dx (k=1,s=1,p=0): for cv1 and cv2
        let conv1_bwd_ptx = std::fs::read(compile_kernel(
            &teeny_kernels::nn::conv::conv2d::Conv2dBackwardDx::<f32>::new(1, 1, 1, 1, 0, 0, 1, BLOCK_OW),
            &target, true, false
        )?)?;
        let conv1_bwd = testing::load_program_from_ptx::<
            teeny_kernels::nn::conv::conv2d::Conv2dBackwardDx<f32>,
        >(&conv1_bwd_ptx)?;

        // Conv3 backward dx (k=3,s=1,p=1): for m0_cv1 and m0_cv2
        let conv3_bwd_ptx = std::fs::read(compile_kernel(
            &teeny_kernels::nn::conv::conv2d::Conv2dBackwardDx::<f32>::new(3, 3, 1, 1, 1, 1, 1, BLOCK_OW),
            &target, true, false
        )?)?;
        let conv3_bwd = testing::load_program_from_ptx::<
            teeny_kernels::nn::conv::conv2d::Conv2dBackwardDx<f32>,
        >(&conv3_bwd_ptx)?;

        let cat_bwd_ptx = std::fs::read(compile_kernel(
            &teeny_kernels::nn::tensor::channel_cat::ChannelCatBackward::<f32>::new(BLOCK_CH),
            &target, true, false
        )?)?;
        let cat_bwd = testing::load_program_from_ptx::<
            teeny_kernels::nn::tensor::channel_cat::ChannelCatBackward<f32>,
        >(&cat_bwd_ptx)?;

        let chunk_bwd_ptx = std::fs::read(compile_kernel(
            &teeny_kernels::nn::tensor::channel_chunk::ChannelChunkBackward::<f32>::new(BLOCK_CH),
            &target, true, false
        )?)?;
        let chunk_bwd = testing::load_program_from_ptx::<
            teeny_kernels::nn::tensor::channel_chunk::ChannelChunkBackward<f32>,
        >(&chunk_bwd_ptx)?;

        let vadd_ptx = std::fs::read(compile_kernel(
            &teeny_kernels::nn::tensor::elemwise_add::ElemwiseAddForward::<f32>::new(BLOCK_VADD),
            &target, true, false
        )?)?;
        let vadd = testing::load_program_from_ptx::<
            teeny_kernels::nn::tensor::elemwise_add::ElemwiseAddForward<f32>,
        >(&vadd_ptx)?;

        let ow_tiles = W.div_ceil(BLOCK_OW as usize);
        let bn_cfg = |c_ch: usize| CudaLaunchConfig {
            grid: [c_ch as u32, 1, 1], block: [1, 1, 1], cluster: [1, 1, 1],
        };
        let ch_cfg = CudaLaunchConfig {
            grid: [N_SPATIAL as u32, 1, 1],
            block: [BLOCK_CH as u32, 1, 1],
            cluster: [1, 1, 1],
        };

        // Macro: run SiluBackward(dy_nc, pre_silu_nc) → dx_nc
        // Args: (upstream_grad_buf NC, pre_silu_buf NC, n)
        macro_rules! silu_bwd {
            ($dy:expr, $x:expr, $n:expr) => {{
                let dx = device.buffer::<f32>($n)?;
                device.launch(&silu_bwd, &testing::launch_config($n, BLOCK_SILU), (
                    $dy.as_device_ptr() as *mut f32,
                    $x.as_device_ptr() as *mut f32,
                    dx.as_device_ptr() as *mut f32,
                    $n as i32,
                ))?;
                dx
            }};
        }

        // Macro: run BatchNormBackward(dy_nc, pre_bn_nc, w, mean, rstd) → dx_nc
        macro_rules! bn_bwd {
            ($dy:expr, $x:expr, $c:expr, $w:expr, $mean:expr, $rstd:expr) => {{
                let n = N_SPATIAL * $c;
                let dx = device.buffer::<f32>(n)?;
                let dw = device.buffer::<f32>($c)?;
                let db = device.buffer::<f32>($c)?;
                let mut w_buf = device.buffer::<f32>($c)?;
                let mut mean_buf = device.buffer::<f32>($c)?;
                let mut rstd_buf = device.buffer::<f32>($c)?;
                w_buf.to_device($w)?;
                mean_buf.to_device($mean)?;
                rstd_buf.to_device($rstd)?;
                device.launch(&bn_bwd, &bn_cfg($c), (
                    $dy.as_device_ptr() as *mut f32,
                    $x.as_device_ptr() as *mut f32,
                    dx.as_device_ptr() as *mut f32,
                    w_buf.as_device_ptr() as *mut f32,
                    mean_buf.as_device_ptr() as *mut f32,
                    rstd_buf.as_device_ptr() as *mut f32,
                    dw.as_device_ptr() as *mut f32,
                    db.as_device_ptr() as *mut f32,
                    N_SPATIAL as i32, $c as i32,
                ))?;
                dx
            }};
        }

        // ── Step 1: Convert dy from NCHW to NC ───────────────────────────────
        let dy_nc_host = nchw_to_nc(&dy_nchw, BATCH, C_OUT, H, W);

        // ── Step 2: cv2 SiLU backward ────────────────────────────────────────
        let mut dy_nc_buf = device.buffer::<f32>(N_CV2)?;
        let mut cv2_psilu_buf = device.buffer::<f32>(N_CV2)?;
        dy_nc_buf.to_device(&dy_nc_host)?;
        cv2_psilu_buf.to_device(&cv2_pre_silu_nc)?;
        let d_cv2_bn_out = silu_bwd!(dy_nc_buf, cv2_psilu_buf, N_CV2);

        // ── Step 3: cv2 BN backward ──────────────────────────────────────────
        let mut cv2_pbn_buf = device.buffer::<f32>(N_CV2)?;
        cv2_pbn_buf.to_device(&cv2_pre_bn_nc)?;
        let d_cv2_conv_out = bn_bwd!(
            d_cv2_bn_out, cv2_pbn_buf, C_OUT,
            &cv2_bn_w, &cv2_bn_mean, &cv2_bn_rstd
        );

        // ── Step 4: cv2 Conv backward dx (k=1) → d_cat (NCHW, zero-init) ────
        // dy is d_cv2_conv_out (NC), needs to be in NCHW for the conv kernel.
        // Conv backward dx: (dy_nchw, w, dx_nchw_zero, B, C_IN, C_OUT, H, W, OH, OW)
        let mut d_cv2_conv_host = vec![0f32; N_CV2];
        d_cv2_conv_out.to_host(&mut d_cv2_conv_host)?;
        let d_cv2_conv_nchw = nc_to_nchw(&d_cv2_conv_host, BATCH, C_OUT, H, W);
        let mut d_cv2_conv_nchw_buf = device.buffer::<f32>(N_CV2)?;
        let mut cv2_cw_buf = device.buffer::<f32>(C_OUT * C_CAT)?;
        let mut d_cat_nchw_buf = device.buffer::<f32>(N_CAT)?;
        let zeros = vec![0f32; N_CAT];
        d_cv2_conv_nchw_buf.to_device(&d_cv2_conv_nchw)?;
        cv2_cw_buf.to_device(&cv2_conv_w)?;
        d_cat_nchw_buf.to_device(&zeros)?; // atomic_add needs zero-init
        device.launch(&conv1_bwd, &CudaLaunchConfig {
            grid: [(BATCH * C_OUT * H * ow_tiles) as u32, 1, 1],
            block: [128, 1, 1], cluster: [1, 1, 1],
        }, (
            d_cv2_conv_nchw_buf.as_device_ptr() as *mut f32, // dy
            cv2_cw_buf.as_device_ptr() as *mut f32,          // w
            d_cat_nchw_buf.as_device_ptr() as *mut f32,      // dx (zero-init)
            BATCH as i32, C_CAT as i32, C_OUT as i32,
            H as i32, W as i32, H as i32, W as i32,
        ))?;

        // Convert d_cat from NCHW to NC
        let mut d_cat_nchw_host = vec![0f32; N_CAT];
        d_cat_nchw_buf.to_host(&mut d_cat_nchw_host)?;
        let d_cat_nc_host = nchw_to_nc(&d_cat_nchw_host, BATCH, C_CAT, H, W);
        let mut d_cat_nc_buf = device.buffer::<f32>(N_CAT)?;
        d_cat_nc_buf.to_device(&d_cat_nc_host)?;

        // ── Step 5: ChannelCat backward ──────────────────────────────────────
        // Extract three gradient slices from d_cat_nc.
        // ChannelCatBackward(dy_wide, dx_narrow, chunk_c, c_total, chunk_offset)
        //   dx[n*chunk_c + ci] = dy[n*c_total + chunk_offset + ci]
        let d_y0 = device.buffer::<f32>(N_HALF)?;
        let d_y1_from_cat = device.buffer::<f32>(N_HALF)?;
        let d_m0out = device.buffer::<f32>(N_M0CV2)?;
        device.launch(&cat_bwd, &ch_cfg, (
            d_cat_nc_buf.as_device_ptr() as *mut f32,
            d_y0.as_device_ptr() as *mut f32,
            C as i32, C_CAT as i32, 0i32,                     // y0 slice: ch [0, 16)
        ))?;
        device.launch(&cat_bwd, &ch_cfg, (
            d_cat_nc_buf.as_device_ptr() as *mut f32,
            d_y1_from_cat.as_device_ptr() as *mut f32,
            C as i32, C_CAT as i32, C as i32,                 // y1 slice: ch [16, 32)
        ))?;
        device.launch(&cat_bwd, &ch_cfg, (
            d_cat_nc_buf.as_device_ptr() as *mut f32,
            d_m0out.as_device_ptr() as *mut f32,
            C as i32, C_CAT as i32, (2 * C) as i32,           // m0_out slice: ch [32, 48)
        ))?;

        // ── Step 6: m0_cv2 SiLU backward ─────────────────────────────────────
        let mut m0cv2_psilu_buf = device.buffer::<f32>(N_M0CV2)?;
        m0cv2_psilu_buf.to_device(&m0cv2_pre_silu_nc)?;
        let d_m0cv2_bn_out = silu_bwd!(d_m0out, m0cv2_psilu_buf, N_M0CV2);

        // ── Step 7: m0_cv2 BN backward ───────────────────────────────────────
        let mut m0cv2_pbn_buf = device.buffer::<f32>(N_M0CV2)?;
        m0cv2_pbn_buf.to_device(&m0cv2_pre_bn_nc)?;
        let d_m0cv2_conv_out = bn_bwd!(
            d_m0cv2_bn_out, m0cv2_pbn_buf, C,
            &m0cv2_bn_w, &m0cv2_bn_mean, &m0cv2_bn_rstd
        );

        // ── Step 8: m0_cv2 Conv backward dx (k=3) → d_m0cv1_silu ────────────
        // m0_cv2: Conv(C_INNER=8 → C=16, 3×3).  dy: [B,16,H,W], dx: [B,8,H,W].
        let mut d_m0cv2_host = vec![0f32; N_M0CV2];
        d_m0cv2_conv_out.to_host(&mut d_m0cv2_host)?;
        let d_m0cv2_nchw = nc_to_nchw(&d_m0cv2_host, BATCH, C, H, W);
        let mut d_m0cv2_nchw_buf = device.buffer::<f32>(N_M0CV2)?;
        let mut m0cv2_cw_buf = device.buffer::<f32>(C * C_INNER * 9)?;
        let mut d_m0cv1_silu_nchw_buf = device.buffer::<f32>(N_M0CV1)?;
        let zeros_m0cv1 = vec![0f32; N_M0CV1];
        d_m0cv2_nchw_buf.to_device(&d_m0cv2_nchw)?;
        m0cv2_cw_buf.to_device(&m0cv2_conv_w)?;
        d_m0cv1_silu_nchw_buf.to_device(&zeros_m0cv1)?;
        device.launch(&conv3_bwd, &CudaLaunchConfig {
            grid: [(BATCH * C * H * ow_tiles) as u32, 1, 1],
            block: [128, 1, 1], cluster: [1, 1, 1],
        }, (
            d_m0cv2_nchw_buf.as_device_ptr() as *mut f32,       // dy: [B,C_OUT=16,H,W]
            m0cv2_cw_buf.as_device_ptr() as *mut f32,           // w:  [C_OUT=16, C_IN=8, 3,3]
            d_m0cv1_silu_nchw_buf.as_device_ptr() as *mut f32,  // dx: [B,C_IN=8,H,W]
            BATCH as i32, C_INNER as i32, C as i32,
            H as i32, W as i32, H as i32, W as i32,
        ))?;

        // Convert d_m0cv1_silu from NCHW to NC
        let mut d_m0cv1_silu_nchw_host = vec![0f32; N_M0CV1];
        d_m0cv1_silu_nchw_buf.to_host(&mut d_m0cv1_silu_nchw_host)?;
        let d_m0cv1_silu_nc_host = nchw_to_nc(&d_m0cv1_silu_nchw_host, BATCH, C_INNER, H, W);
        let mut d_m0cv1_silu_nc_buf = device.buffer::<f32>(N_M0CV1)?;
        d_m0cv1_silu_nc_buf.to_device(&d_m0cv1_silu_nc_host)?;

        // ── Step 9: m0_cv1 SiLU backward ─────────────────────────────────────
        let mut m0cv1_psilu_buf = device.buffer::<f32>(N_M0CV1)?;
        m0cv1_psilu_buf.to_device(&m0cv1_pre_silu_nc)?;
        let d_m0cv1_bn_out = silu_bwd!(d_m0cv1_silu_nc_buf, m0cv1_psilu_buf, N_M0CV1);

        // ── Step 10: m0_cv1 BN backward ──────────────────────────────────────
        let mut m0cv1_pbn_buf = device.buffer::<f32>(N_M0CV1)?;
        m0cv1_pbn_buf.to_device(&m0cv1_pre_bn_nc)?;
        let d_m0cv1_conv_out = bn_bwd!(
            d_m0cv1_bn_out, m0cv1_pbn_buf, C_INNER,
            &m0cv1_bn_w, &m0cv1_bn_mean, &m0cv1_bn_rstd
        );

        // ── Step 11: m0_cv1 Conv backward dx (k=3) → d_y1_bottleneck ─────────
        // m0_cv1: Conv(C=16 → C_INNER=8, 3×3).  dy: [B,8,H,W], dx: [B,16,H,W].
        let mut d_m0cv1_host = vec![0f32; N_M0CV1];
        d_m0cv1_conv_out.to_host(&mut d_m0cv1_host)?;
        let d_m0cv1_nchw = nc_to_nchw(&d_m0cv1_host, BATCH, C_INNER, H, W);
        let mut d_m0cv1_nchw_buf = device.buffer::<f32>(N_M0CV1)?;
        let mut m0cv1_cw_buf = device.buffer::<f32>(C_INNER * C * 9)?;
        let mut d_y1_bottleneck_nchw_buf = device.buffer::<f32>(N_HALF)?;
        let zeros_half = vec![0f32; N_HALF];
        d_m0cv1_nchw_buf.to_device(&d_m0cv1_nchw)?;
        m0cv1_cw_buf.to_device(&m0cv1_conv_w)?;
        d_y1_bottleneck_nchw_buf.to_device(&zeros_half)?;
        device.launch(&conv3_bwd, &CudaLaunchConfig {
            grid: [(BATCH * C_INNER * H * ow_tiles) as u32, 1, 1],
            block: [128, 1, 1], cluster: [1, 1, 1],
        }, (
            d_m0cv1_nchw_buf.as_device_ptr() as *mut f32,         // dy: [B,C_OUT=8,H,W]
            m0cv1_cw_buf.as_device_ptr() as *mut f32,             // w:  [C_OUT=8, C_IN=16,3,3]
            d_y1_bottleneck_nchw_buf.as_device_ptr() as *mut f32, // dx: [B,C_IN=16,H,W]
            BATCH as i32, C as i32, C_INNER as i32,
            H as i32, W as i32, H as i32, W as i32,
        ))?;

        // Convert d_y1_bottleneck to NC
        let mut d_y1_bn_host = vec![0f32; N_HALF];
        d_y1_bottleneck_nchw_buf.to_host(&mut d_y1_bn_host)?;
        let d_y1_bn_nc_host = nchw_to_nc(&d_y1_bn_host, BATCH, C, H, W);
        let mut d_y1_bottleneck_nc_buf = device.buffer::<f32>(N_HALF)?;
        d_y1_bottleneck_nc_buf.to_device(&d_y1_bn_nc_host)?;

        // ── Step 12: d_y1 = d_y1_from_cat + d_y1_bottleneck ──────────────────
        // y1 contributes gradient through two paths:
        //   a) directly in the cat (d_y1_from_cat)
        //   b) through the bottleneck (d_y1_bottleneck)
        let d_y1 = device.buffer::<f32>(N_HALF)?;
        device.launch(&vadd, &testing::launch_config(N_HALF, BLOCK_VADD), (
            d_y1_from_cat.as_device_ptr() as *mut f32,
            d_y1_bottleneck_nc_buf.as_device_ptr() as *mut f32,
            d_y1.as_device_ptr() as *mut f32,
            N_HALF as i32,
        ))?;

        // ── Step 13: ChannelChunk backward → d_cv1_silu ───────────────────────
        // Scatter d_y0 and d_y1 into their respective channel ranges of d_cv1_silu.
        // ChannelChunkBackward(dy_narrow, dx_wide, c_total, chunk_c, chunk_offset)
        //   dx[n*c_total + chunk_offset + ci] = dy[n*chunk_c + ci]
        let mut d_cv1_silu_nc_buf = device.buffer::<f32>(N_CV1)?;
        let zeros_cv1 = vec![0f32; N_CV1];
        d_cv1_silu_nc_buf.to_device(&zeros_cv1)?; // pre-zero; writes are non-overlapping
        device.launch(&chunk_bwd, &ch_cfg, (
            d_y0.as_device_ptr() as *mut f32,
            d_cv1_silu_nc_buf.as_device_ptr() as *mut f32,
            (2 * C) as i32, C as i32, 0i32,    // y0 → ch [0, 16)
        ))?;
        device.launch(&chunk_bwd, &ch_cfg, (
            d_y1.as_device_ptr() as *mut f32,
            d_cv1_silu_nc_buf.as_device_ptr() as *mut f32,
            (2 * C) as i32, C as i32, C as i32, // y1 → ch [16, 32)
        ))?;

        // ── Step 14: cv1 SiLU backward ────────────────────────────────────────
        let mut cv1_psilu_buf = device.buffer::<f32>(N_CV1)?;
        cv1_psilu_buf.to_device(&cv1_pre_silu_nc)?;
        let d_cv1_bn_out = silu_bwd!(d_cv1_silu_nc_buf, cv1_psilu_buf, N_CV1);

        // ── Step 15: cv1 BN backward ──────────────────────────────────────────
        let mut cv1_pbn_buf = device.buffer::<f32>(N_CV1)?;
        cv1_pbn_buf.to_device(&cv1_pre_bn_nc)?;
        let d_cv1_conv_out = bn_bwd!(
            d_cv1_bn_out, cv1_pbn_buf, 2 * C,
            &cv1_bn_w, &cv1_bn_mean, &cv1_bn_rstd
        );

        // ── Step 16: cv1 Conv backward dx (k=1) → dx_input ───────────────────
        // cv1: Conv(C_IN=32 → 2*C=32, 1×1).  dy: [B,32,H,W], dx: [B,32,H,W].
        let mut d_cv1_conv_host = vec![0f32; N_CV1];
        d_cv1_conv_out.to_host(&mut d_cv1_conv_host)?;
        let d_cv1_conv_nchw = nc_to_nchw(&d_cv1_conv_host, BATCH, 2 * C, H, W);
        let mut d_cv1_conv_nchw_buf = device.buffer::<f32>(N_CV1)?;
        let mut cv1_cw_buf = device.buffer::<f32>(2 * C * C_IN)?;
        let mut dx_nchw_buf = device.buffer::<f32>(N_INPUT)?;
        let zeros_input = vec![0f32; N_INPUT];
        d_cv1_conv_nchw_buf.to_device(&d_cv1_conv_nchw)?;
        cv1_cw_buf.to_device(&cv1_conv_w)?;
        dx_nchw_buf.to_device(&zeros_input)?;
        device.launch(&conv1_bwd, &CudaLaunchConfig {
            grid: [(BATCH * 2 * C * H * ow_tiles) as u32, 1, 1],
            block: [128, 1, 1], cluster: [1, 1, 1],
        }, (
            d_cv1_conv_nchw_buf.as_device_ptr() as *mut f32, // dy: [B,2C=32,H,W]
            cv1_cw_buf.as_device_ptr() as *mut f32,          // w:  [2C=32, C_IN=32,1,1]
            dx_nchw_buf.as_device_ptr() as *mut f32,         // dx: [B,C_IN=32,H,W]
            BATCH as i32, C_IN as i32, (2 * C) as i32,
            H as i32, W as i32, H as i32, W as i32,
        ))?;

        // ── Step 17: Compare dx with reference ───────────────────────────────
        let mut dx_gpu = vec![0f32; N_INPUT];
        dx_nchw_buf.to_host(&mut dx_gpu)?;

        // Both dx_gpu and dx_expected are NCHW; compare element-wise.
        for i in 0..N_INPUT {
            assert!(
                (dx_gpu[i] - dx_expected[i]).abs() < TOL,
                "backward dx mismatch at {i}: gpu={:.6} expected={:.6}",
                dx_gpu[i], dx_expected[i],
            );
        }
        Ok(())
    }
}
