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


// Full backward-pass integration test for a YOLO26n-equivalent model.
//
// Pipeline under test:
//   1. Build a YOLO26n-equivalent graph (64×64, nc=4).
//   2. Compile all kernels with LoweringMode::Training.
//   3. Load; initialise all parameters to 0.01.
//   4. forward_train → boxes [1,4·84] + scores [1,4·84] retained in cache.
//   5. Yolo26Loss::compute_grads → (d_boxes, d_scores) on host.
//   6. Upload as TensorRef seeds; call backward_multi.
//   7. Assert at least one parameter gradient is non-zero.
//
// Architecture note:
//   The model is identical to YOLO26n except that layer 10 (C2PSA) is
//   replaced by a plain C3k2.  BatchNorm2dNchwInferenceRuntimeOp (used
//   inside the PSA attention decomposition) triggers a compiler-fork ICE
//   (codegen_loop_body_switch_as_scf_if).  C3k2 uses only training-mode BN
//   nodes which compile cleanly.  The backbone/head structure, channel widths,
//   and all fan-out skip connections are otherwise identical to YOLO26n.
//
// Requires TEENYC_PATH (and optionally TEENYC_CACHE_DIR) in the
// environment. Kernel compilation is cached; first run may be slow.

#[cfg(feature = "cuda")]
mod cuda {
    use dotenv::dotenv;
    use serial_test::serial;
    use teeny_compiler::compiler::backend::llvm::compiler::LlvmCompiler;
    use teeny_core::{
        graph::{DtypeRepr, SymTensor},
        model::LoweringMode,
    };
    use teeny_cuda::{compiler::{graph::CudaGraphCompiler, target::Target}, model::TensorRef};
    use teeny_test::cuda as testing;
    use teeny_kernels::graph::TritonLowering;
    use vision_rs::models::yolo::{
        loss::yolo26::Yolo26Loss,
        yolo26::blocks::{
            c3k2::c3k2,
            concat::concat,
            conv::conv,
            detect::{DetectHead, DetectOutput, detect},
            sppf::sppf,
            upsample::upsample,
        },
    };

    const BATCH: usize = 1;
    const NC: usize = 4; // small class count keeps the scores tensor compact
    const IMG_H: usize = 64;
    const IMG_W: usize = 64;
    // Anchors at 64×64 with strides [8,16,32]:
    //   8×8 + 4×4 + 2×2 = 64 + 16 + 4 = 84
    const N_ANCHORS: usize = 84;

    // ── YOLO26n-equivalent model (C2PSA replaced by C3k2) ────────────────────

    fn build_model(nc: usize) -> impl Fn(SymTensor) -> DetectOutput {
        // Channel widths: YOLO26n (width=0.25, capped at 1024)
        let (c0, c1, c2, c3, c4) = (16usize, 32, 64, 128, 256);
        let n = 1usize; // depth: max(round(2*0.5), 1) = 1

        // Backbone
        let l0 = conv::<f32>(3, c0, 3, 2);
        let l1 = conv::<f32>(c0, c1, 3, 2);
        let l2 = c3k2::<f32>(c1, c2, n, false, true, 0.25);
        let l3 = conv::<f32>(c2, c2, 3, 2);
        let l4 = c3k2::<f32>(c2, c3, n, false, true, 0.25); // → p3
        let l5 = conv::<f32>(c3, c3, 3, 2);
        let l6 = c3k2::<f32>(c3, c3, n, true, true, 0.5); // → p4
        let l7 = conv::<f32>(c3, c4, 3, 2);
        let l8 = c3k2::<f32>(c4, c4, n, true, true, 0.5);
        let l9 = sppf::<f32>(c4, c4, true);
        // Layer 10: C3k2 instead of C2PSA (avoids PSA attention BN compiler ICE)
        let l10 = c3k2::<f32>(c4, c4, n, true, true, 0.5); // → p5

        // Head
        let up = upsample(2, 2);
        let cat = concat();
        let l13 = c3k2::<f32>(c4 + c3, c3, n, true, true, 0.5); // neck4
        let l16 = c3k2::<f32>(c3 + c3, c2, n, true, true, 0.5); // p3_det
        let l17 = conv::<f32>(c2, c2, 3, 2);
        let l19 = c3k2::<f32>(c2 + c3, c3, n, true, true, 0.5); // p4_det
        let l20 = conv::<f32>(c3, c3, 3, 2);
        let l22 = c3k2::<f32>(c3 + c4, c4, 1, true, true, 0.5); // p5_det
        let head = detect::<f32>(nc, &[c2, c3, c4], DetectHead::OneToMany);

        move |x: SymTensor| {
            // Backbone
            let x = l0(x);
            let x = l1(x);
            let x = l2(x);
            let x = l3(x);
            let p3 = l4(x); // skip to top-down neck
            let x = l5(p3.clone());
            let p4 = l6(x); // skip to first neck concat
            let x = l7(p4.clone());
            let x = l8(x);
            let x = l9(x);
            let p5 = l10(x); // skip to last neck concat

            // Top-down neck
            let x = up(p5.clone());
            let x = cat(vec![x, p4]);
            let nk4 = l13(x); // skip to p4_det concat

            let x = up(nk4.clone());
            let x = cat(vec![x, p3]);
            let p3d = l16(x);

            // Bottom-up path
            let x = l17(p3d.clone());
            let x = cat(vec![x, nk4]);
            let p4d = l19(x);

            let x = l20(p4d.clone());
            let x = cat(vec![x, p5]);
            let p5d = l22(x);

            head(vec![p3d, p4d, p5d])
        }
    }

    // ── Integration test ──────────────────────────────────────────────────────

    #[test]
    #[serial]
    fn test_yolo26n_backward_grads_non_zero() -> anyhow::Result<()> {
        dotenv().ok();

        // ── CUDA setup ────────────────────────────────────────────────────────

        let env = testing::setup_cuda_env()?;
        let device = &env.device;
        let target = Target::new(env.capability);

        // ── 1. Trace the model graph ──────────────────────────────────────────

        let (input_sym, _graph_rc) = SymTensor::input(
            DtypeRepr::F32,
            vec![None, Some(3), Some(IMG_H), Some(IMG_W)],
        );
        let out = build_model(NC)(input_sym);
        let graph = out.boxes.graph.borrow();

        // ── 2. Compile all kernels ────────────────────────────────────────────

        let rustc_path =
            std::env::var("TEENYC_PATH").expect("TEENYC_PATH must be set to run this test");
        let cache_dir =
            std::env::var("TEENYC_CACHE_DIR").unwrap_or_else(|_| "/tmp/teenyc_cache".to_string());
        let compiler = LlvmCompiler::new(rustc_path, cache_dir)?;
        let graph_compiler = CudaGraphCompiler::new(compiler);
        let lowering = TritonLowering::new();
        println!("compiling YOLO26n-equivalent model (first run may take several minutes)...");
        let cuda_model = graph_compiler.compile_model(
            &graph,
            &lowering,
            &target,
            LoweringMode::Training,
            false,
        )?;
        println!("compilation complete ({} DAG nodes)", cuda_model.dag.len());

        // ── 3. Load + initialise parameters ──────────────────────────────────

        let mut model = cuda_model.load(device, BATCH)?;
        let param_info: Vec<(usize, Vec<Vec<usize>>)> = model
            .param_info()
            .map(|(idx, shapes)| (idx, shapes.to_vec()))
            .collect();
        for (idx, shapes) in &param_info {
            for (pi, shape) in shapes.iter().enumerate() {
                let n: usize = shape.iter().product();
                model.load_param_f32(*idx, pi, &vec![0.01f32; n])?;
            }
        }

        // ── 4. Forward pass ───────────────────────────────────────────────────

        let input_data = vec![1.0f32; BATCH * 3 * IMG_H * IMG_W];
        // input_ref ptr is cloned into the ActivationCache; freed on cache drop.
        let input_ref = TensorRef::from_host_f32(&input_data, vec![BATCH, 3, IMG_H, IMG_W])?;
        model.zero_grad();
        let (_, cache) = model.forward_train(device, BATCH, &[input_ref])?;

        // ── 5. Identify terminal nodes, copy predictions to host ──────────────

        let terminals = model.terminal_node_indices_sorted_by_size();
        assert_eq!(
            terminals.len(),
            2,
            "expected 2 terminal nodes (boxes + scores), got {}",
            terminals.len()
        );
        let (boxes_idx, scores_idx) = (terminals[0], terminals[1]);

        let boxes_host = cache.tensors[boxes_idx].as_ref().unwrap().to_host_f32()?;
        let scores_host = cache.tensors[scores_idx].as_ref().unwrap().to_host_f32()?;

        assert_eq!(boxes_host.len(), BATCH * 4 * N_ANCHORS, "wrong boxes size");
        assert_eq!(
            scores_host.len(),
            BATCH * NC * N_ANCHORS,
            "wrong scores size"
        );

        // ── 6. Compute CIoU + BCE loss gradients ─────────────────────────────

        let loss = Yolo26Loss::new(IMG_H, IMG_W, NC, env.capability);
        let gt_boxes = vec![vec![[32.0f32, 32.0, 20.0, 20.0]]];
        let gt_cls = vec![vec![0usize]];
        let (d_boxes, d_scores) =
            loss.compute_grads(device, &boxes_host, &scores_host, &gt_boxes, &gt_cls)?;

        // ── 7. Seed backward_multi ────────────────────────────────────────────

        let d_boxes_ref = TensorRef::from_host_f32(&d_boxes, vec![BATCH, 4 * N_ANCHORS])?;
        let d_scores_ref = TensorRef::from_host_f32(&d_scores, vec![BATCH, NC * N_ANCHORS])?;

        model.backward_multi(
            device,
            BATCH,
            &[
                (boxes_idx, d_boxes_ref.clone()),
                (scores_idx, d_scores_ref.clone()),
            ],
            &cache,
        )?;

        d_boxes_ref.free()?;
        d_scores_ref.free()?;
        drop(cache);

        // ── 8. Assert non-zero gradients ──────────────────────────────────────

        let any_nonzero = param_info.iter().any(|(idx, shapes)| {
            shapes.iter().enumerate().any(|(pi, _)| {
                model
                    .read_param_grad_f32(*idx, pi)
                    .map(|g| g.iter().any(|&v| v != 0.0))
                    .unwrap_or(false)
            })
        });
        assert!(
            any_nonzero,
            "all parameter gradients are zero — backward pass is not propagating"
        );

        let grad_nodes = param_info
            .iter()
            .filter(|(idx, shapes)| {
                shapes.iter().enumerate().any(|(pi, _)| {
                    model
                        .read_param_grad_f32(*idx, pi)
                        .map(|g| g.iter().any(|&v| v != 0.0))
                        .unwrap_or(false)
                })
            })
            .count();
        println!(
            "✓  YOLO26n backward: {grad_nodes}/{} param nodes have non-zero gradients",
            param_info.len()
        );

        Ok(())
    }
}
