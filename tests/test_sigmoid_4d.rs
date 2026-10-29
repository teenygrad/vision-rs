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


#[cfg(feature = "cuda")]
mod cuda {
    use dotenv::dotenv;
    use serial_test::serial;
    use teeny_cuda::compiler::{compile_kernel, target::Target};
    use teeny_core::device::{Device, buffer::Buffer};
    use teeny_cuda::errors::Result;
    use teeny_test::cuda as testing;

    // Shape: (N=2, C=8, H=16, W=16) stored as a flat contiguous f32 buffer.
    const N: usize = 2 * 8 * 16 * 16; // 4096
    const BLOCK_SIZE: i32 = 128;

    fn load_fixture(rel: &str) -> Vec<f32> {
        let path = format!(
            "{}/tests/fixtures/sigmoid_4d/{}",
            env!("CARGO_MANIFEST_DIR"),
            rel
        );
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    #[test]
    #[serial]
    fn test_sigmoid_4d_forward_cuda() -> Result<()> {
        dotenv().ok();
        let env = testing::setup_cuda_env()?;

        let x_host = load_fixture("x.bin");
        let expected = load_fixture("expected_forward.bin");
        let mut y_host = vec![0.0f32; N];

        let mut x_buf = env.device.buffer::<f32>(N)?;
        let y_buf = env.device.buffer::<f32>(N)?;
        x_buf.to_device(&x_host)?;

        let kernel = teeny_kernels::nn::activation::sigmoid::SigmoidForward::<f32>::new(BLOCK_SIZE);
        let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true, false)?)?;
        let program = testing::load_program_from_ptx::<
            teeny_kernels::nn::activation::sigmoid::SigmoidForward<f32>,
        >(&ptx)?;
        env.device.launch(
            &program,
            &testing::launch_config(N, BLOCK_SIZE),
            (
                x_buf.as_device_ptr() as *mut f32,
                y_buf.as_device_ptr() as *mut f32,
                N as i32,
            ),
        )?;
        y_buf.to_host(&mut y_host)?;

        for i in 0..N {
            assert!(
                (y_host[i] - expected[i]).abs() < 1e-5,
                "sigmoid_4d_forward mismatch at {i}: got={} expected={}",
                y_host[i],
                expected[i],
            );
        }
        Ok(())
    }

    #[test]
    #[serial]
    fn test_sigmoid_4d_backward_cuda() -> Result<()> {
        dotenv().ok();
        let env = testing::setup_cuda_env()?;

        // Sigmoid backward takes (dy, y, dx) — y is the forward output.
        let dy_host = load_fixture("dy.bin");
        let y_host = load_fixture("expected_forward.bin");
        let expected = load_fixture("expected_backward.bin");
        let mut dx_host = vec![0.0f32; N];

        let mut dy_buf = env.device.buffer::<f32>(N)?;
        let mut y_buf = env.device.buffer::<f32>(N)?;
        let dx_buf = env.device.buffer::<f32>(N)?;
        dy_buf.to_device(&dy_host)?;
        y_buf.to_device(&y_host)?;

        let kernel = teeny_kernels::nn::activation::sigmoid::SigmoidBackward::<f32>::new(BLOCK_SIZE);
        let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true, false)?)?;
        let program = testing::load_program_from_ptx::<
            teeny_kernels::nn::activation::sigmoid::SigmoidBackward<f32>,
        >(&ptx)?;
        env.device.launch(
            &program,
            &testing::launch_config(N, BLOCK_SIZE),
            (
                dy_buf.as_device_ptr() as *mut f32,
                y_buf.as_device_ptr() as *mut f32,
                dx_buf.as_device_ptr() as *mut f32,
                N as i32,
            ),
        )?;
        dx_buf.to_host(&mut dx_host)?;

        for i in 0..N {
            assert!(
                (dx_host[i] - expected[i]).abs() < 1e-5,
                "sigmoid_4d_backward mismatch at {i}: got={} expected={}",
                dx_host[i],
                expected[i],
            );
        }
        Ok(())
    }
}
