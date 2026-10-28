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


// Softmax over the class-score dimension of a YOLOv8 P5 detection head.
//
// Layout: (batch=2, anchors=20×20) × (n_cols=128) — 80 COCO classes padded
// to the next power of 2, which the kernel requires (BLOCK_SIZE == n_cols).
// Grid: one CTA per row (per anchor); block: [BLOCK_SIZE, 1, 1].

#[cfg(feature = "cuda")]
mod cuda {
    use dotenv::dotenv;
    use serial_test::serial;
    use teeny_cuda::compiler::{compile_kernel, target::Target};
    use teeny_core::device::{Device, buffer::Buffer};
    use teeny_cuda::{device::CudaLaunchConfig, errors::Result};
    use teeny_test::cuda as testing;

    const N_ROWS: usize = 2 * 20 * 20; // batch × P5 anchor grid = 800
    const N_COLS: usize = 128; // 80 COCO classes → next power of 2
    const BLOCK_SIZE: i32 = 128; // must equal N_COLS

    fn load_fixture(rel: &str) -> Vec<f32> {
        let path = format!(
            "{}/tests/fixtures/softmax_yolo/{}",
            env!("CARGO_MANIFEST_DIR"),
            rel
        );
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    fn launch_cfg() -> CudaLaunchConfig {
        CudaLaunchConfig {
            grid: [N_ROWS as u32, 1, 1],
            block: [BLOCK_SIZE as u32, 1, 1],
            cluster: [1, 1, 1],
        }
    }

    #[test]
    #[serial]
    fn test_softmax_yolo_forward_cuda() -> Result<()> {
        dotenv().ok();
        let env = testing::setup_cuda_env()?;

        let x_host = load_fixture("x.bin");
        let expected = load_fixture("expected_forward.bin");
        let mut y_host = vec![0.0f32; N_ROWS * N_COLS];

        let mut x_buf = env.device.buffer::<f32>(N_ROWS * N_COLS)?;
        let y_buf = env.device.buffer::<f32>(N_ROWS * N_COLS)?;
        x_buf.to_device(&x_host)?;

        let kernel = teeny_kernels::nn::activation::softmax::SoftmaxForward::<f32>::new(BLOCK_SIZE);
        let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true, false)?)?;
        let program = testing::load_program_from_ptx::<
            teeny_kernels::nn::activation::softmax::SoftmaxForward<f32>,
        >(&ptx)?;
        env.device.launch(
            &program,
            &launch_cfg(),
            (
                x_buf.as_device_ptr() as *mut f32,
                y_buf.as_device_ptr() as *mut f32,
                N_ROWS as i32,
                N_COLS as i32,
            ),
        )?;
        y_buf.to_host(&mut y_host)?;

        for i in 0..N_ROWS * N_COLS {
            assert!(
                (y_host[i] - expected[i]).abs() < 1e-5,
                "softmax_yolo_forward mismatch at {i}: got={} expected={}",
                y_host[i],
                expected[i],
            );
        }

        // Each row of class probabilities must sum to 1.
        for r in 0..N_ROWS {
            let row_sum: f32 = y_host[r * N_COLS..(r + 1) * N_COLS].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-5,
                "row {r} sums to {row_sum}, expected 1.0",
            );
        }

        Ok(())
    }

    #[test]
    #[serial]
    fn test_softmax_yolo_backward_cuda() -> Result<()> {
        dotenv().ok();
        let env = testing::setup_cuda_env()?;

        // Softmax backward takes (dy, y, dx) — y is the saved forward output.
        let dy_host = load_fixture("dy.bin");
        let y_host = load_fixture("expected_forward.bin");
        let expected = load_fixture("expected_backward.bin");
        let mut dx_host = vec![0.0f32; N_ROWS * N_COLS];

        let mut dy_buf = env.device.buffer::<f32>(N_ROWS * N_COLS)?;
        let mut y_buf = env.device.buffer::<f32>(N_ROWS * N_COLS)?;
        let dx_buf = env.device.buffer::<f32>(N_ROWS * N_COLS)?;
        dy_buf.to_device(&dy_host)?;
        y_buf.to_device(&y_host)?;

        let kernel =
            teeny_kernels::nn::activation::softmax::SoftmaxBackward::<f32>::new(BLOCK_SIZE);
        let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true, false)?)?;
        let program = testing::load_program_from_ptx::<
            teeny_kernels::nn::activation::softmax::SoftmaxBackward<f32>,
        >(&ptx)?;
        env.device.launch(
            &program,
            &launch_cfg(),
            (
                dy_buf.as_device_ptr() as *mut f32,
                y_buf.as_device_ptr() as *mut f32,
                dx_buf.as_device_ptr() as *mut f32,
                N_ROWS as i32,
                N_COLS as i32,
            ),
        )?;
        dx_buf.to_host(&mut dx_host)?;

        for i in 0..N_ROWS * N_COLS {
            assert!(
                (dx_host[i] - expected[i]).abs() < 1e-5,
                "softmax_yolo_backward mismatch at {i}: got={} expected={}",
                dx_host[i],
                expected[i],
            );
        }

        Ok(())
    }
}
