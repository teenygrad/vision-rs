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


//! YOLO26 training loss: CIoU box regression + BCE classification.
//!
//! # Pipeline (per training step)
//!
//! 1. **Decode** raw LTRB predictions to XYWH using the anchor grid.
//! 2. **Assign** GT boxes to anchors (CPU-side TaskAlignedAssigner).
//! 3. **Box loss**: CIoU forward (saved activations) then backward.
//! 4. **Cls loss**: BCE cls backward on all anchors.
//! 5. **Decode backward**: d_xywh → d_ltrb through the linear decode.
//! 6. Return `d_boxes` `[B, 4*A]` and `d_scores` `[B, nc*A]` (host f32).

pub use super::{anchor::AnchorGrid, assign::TaskAlignedAssigner};

/// CPU reference CIoU forward — returns saved activations (iou, v, alpha).
pub fn ciou_forward_cpu(pred: &[f32], target: &[f32], n: usize)
    -> (Vec<f32>, Vec<f32>, Vec<f32>)
{
    const EPS: f32 = 1e-7;
    let pi2 = std::f32::consts::PI * std::f32::consts::PI;
    let mut iou   = vec![0.0f32; n];
    let mut v_vec = vec![0.0f32; n];
    let mut alpha = vec![0.0f32; n];

    for i in 0..n {
        let px = pred[i];           let py = pred[n + i];
        let pw = pred[2 * n + i];   let ph = pred[3 * n + i];
        let tx = target[i];         let ty = target[n + i];
        let tw = target[2 * n + i]; let th = target[3 * n + i];

        let px1 = px - pw * 0.5; let px2 = px + pw * 0.5;
        let py1 = py - ph * 0.5; let py2 = py + ph * 0.5;
        let tx1 = tx - tw * 0.5; let tx2 = tx + tw * 0.5;
        let ty1 = ty - th * 0.5; let ty2 = ty + th * 0.5;

        let inter = ((px2.min(tx2) - px1.max(tx1)).max(0.0))
                  * ((py2.min(ty2) - py1.max(ty1)).max(0.0));
        let union = pw * ph + tw * th - inter;
        let iou_i = inter / (union + EPS);

        let v_i = (4.0 / pi2)
            * ((tw / (th + EPS)).atan() - (pw / (ph + EPS)).atan()).powi(2);
        let alpha_i = v_i / (1.0 - iou_i + v_i + EPS);

        iou[i] = iou_i; v_vec[i] = v_i; alpha[i] = alpha_i;
    }
    (iou, v_vec, alpha)
}

// ── CUDA-only training loss ───────────────────────────────────────────────────

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use teeny_core::device::{Device, buffer::Buffer};
    use teeny_cuda::{
        compiler::{compile_kernel, target::Capability, target::Target},
        device::{CudaDevice, CudaLaunchConfig, program::CudaProgram},
        errors::Result,
    };
    use crate::models::yolo::kernels::loss::{
        ciou::{YoloCiouLossBackward, YoloCiouLossForward},
        cls::YoloBceClsLossBackward,
    };

    /// One CTA per `grid_x`, with block/cluster dims from the compiled kernel's metadata.
    fn launch_config_with_grid<K: teeny_core::device::program::Kernel>(
        grid_x: usize,
        program: &CudaProgram<'_, K>,
    ) -> CudaLaunchConfig {
        CudaLaunchConfig {
            grid: [grid_x as u32, 1, 1],
            block: [program.threads_per_block().max(1), 1, 1],
            cluster: [program.num_ctas().max(1), 1, 1],
        }
    }

    /// CUDA training loss for YOLO26: computes d_boxes and d_scores.
    pub struct Yolo26Loss {
        /// Anchor grid (points + strides) for the model's input resolution.
        pub grid:         AnchorGrid,
        /// One2many assigner (top_k=10) — used during `compute_grads` and the o2m head of `compute_grads_dual`.
        pub assigner:     TaskAlignedAssigner,
        /// One2one assigner (top_k=1) — used for the o2o head of `compute_grads_dual`.
        pub assigner_o2o: TaskAlignedAssigner,
        /// Number of detection classes.
        pub nc:           usize,
        /// Target GPU compute capability for compiled kernels.
        pub cap:          Capability,
        block_n:          i32,
    }

    impl Yolo26Loss {
        /// Builds the loss state for a model trained at `img_h × img_w` with `nc` classes.
        pub fn new(img_h: usize, img_w: usize, nc: usize, cap: Capability) -> Self {
            Self {
                grid:         AnchorGrid::yolo26(img_h, img_w),
                assigner:     TaskAlignedAssigner::default(),
                assigner_o2o: TaskAlignedAssigner { top_k: 1, ..TaskAlignedAssigner::default() },
                nc, cap, block_n: 64,
            }
        }

        /// Compute loss gradients for one batch (one2many head only).
        ///
        /// `boxes`   – raw LTRB predictions `[B, 4*A]` (host f32, channels-first)
        /// `scores`  – class logits `[B, nc*A]` (host f32, channels-first)
        ///
        /// Returns `(d_boxes [B,4*A], d_scores [B,nc*A])` as host f32.
        pub fn compute_grads(
            &self,
            device: &CudaDevice<'_>,
            boxes:      &[f32],
            scores:     &[f32],
            gt_boxes_b: &[Vec<[f32; 4]>],
            gt_cls_b:   &[Vec<usize>],
        ) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
            let target = Target::new(self.cap);
            let bn = self.block_n;
            let ciou_fwd_ptx = std::fs::read(compile_kernel(&YoloCiouLossForward::<f32>::new(bn), &target, true, false)?)?;
            let ciou_bwd_ptx = std::fs::read(compile_kernel(&YoloCiouLossBackward::<f32>::new(bn), &target, true, false)?)?;
            let cls_bwd_ptx  = std::fs::read(compile_kernel(&YoloBceClsLossBackward::<f32>::new(bn), &target, true, false)?)?;
            let prog_ciou_fwd = CudaProgram::<YoloCiouLossForward<f32>>::try_from_ptx(&ciou_fwd_ptx)?;
            let prog_ciou_bwd = CudaProgram::<YoloCiouLossBackward<f32>>::try_from_ptx(&ciou_bwd_ptx)?;
            let prog_cls_bwd  = CudaProgram::<YoloBceClsLossBackward<f32>>::try_from_ptx(&cls_bwd_ptx)?;
            self.compute_grads_for_head(
                device, &prog_ciou_fwd, &prog_ciou_bwd, &prog_cls_bwd,
                &self.assigner, boxes, scores, gt_boxes_b, gt_cls_b, 1.0,
            )
        }

        /// Compute dual-head loss gradients for consistent assignment training.
        ///
        /// Runs TAL assignment independently for both heads using their respective
        /// assigners (top_k=10 for o2m, top_k=1 for o2o), then scales the resulting
        /// gradients by `w_o2m` and `w_o2o` respectively.
        ///
        /// # Loss weight schedule (ultralytics-style)
        /// - `w_o2m = 1.0` (constant throughout training)
        /// - `w_o2o = step / total_steps` (ramps 0→1 linearly; caller controls the schedule)
        ///
        /// Returns `(d_boxes_o2m, d_scores_o2m, d_boxes_o2o, d_scores_o2o)`.
        #[allow(clippy::too_many_arguments, clippy::type_complexity)]
        pub fn compute_grads_dual(
            &self,
            device:      &CudaDevice<'_>,
            boxes_o2m:   &[f32],
            scores_o2m:  &[f32],
            boxes_o2o:   &[f32],
            scores_o2o:  &[f32],
            gt_boxes_b:  &[Vec<[f32; 4]>],
            gt_cls_b:    &[Vec<usize>],
            w_o2m:       f32,
            w_o2o:       f32,
        ) -> anyhow::Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
            let target = Target::new(self.cap);
            let bn = self.block_n;
            let ciou_fwd_ptx = std::fs::read(compile_kernel(&YoloCiouLossForward::<f32>::new(bn), &target, true, false)?)?;
            let ciou_bwd_ptx = std::fs::read(compile_kernel(&YoloCiouLossBackward::<f32>::new(bn), &target, true, false)?)?;
            let cls_bwd_ptx  = std::fs::read(compile_kernel(&YoloBceClsLossBackward::<f32>::new(bn), &target, true, false)?)?;
            let prog_ciou_fwd = CudaProgram::<YoloCiouLossForward<f32>>::try_from_ptx(&ciou_fwd_ptx)?;
            let prog_ciou_bwd = CudaProgram::<YoloCiouLossBackward<f32>>::try_from_ptx(&ciou_bwd_ptx)?;
            let prog_cls_bwd  = CudaProgram::<YoloBceClsLossBackward<f32>>::try_from_ptx(&cls_bwd_ptx)?;

            let (d_boxes_o2m, d_scores_o2m) = self.compute_grads_for_head(
                device, &prog_ciou_fwd, &prog_ciou_bwd, &prog_cls_bwd,
                &self.assigner, boxes_o2m, scores_o2m, gt_boxes_b, gt_cls_b, w_o2m,
            )?;
            let (d_boxes_o2o, d_scores_o2o) = self.compute_grads_for_head(
                device, &prog_ciou_fwd, &prog_ciou_bwd, &prog_cls_bwd,
                &self.assigner_o2o, boxes_o2o, scores_o2o, gt_boxes_b, gt_cls_b, w_o2o,
            )?;
            Ok((d_boxes_o2m, d_scores_o2m, d_boxes_o2o, d_scores_o2o))
        }

        #[allow(clippy::too_many_arguments)]
        fn compute_grads_for_head(
            &self,
            device:       &CudaDevice<'_>,
            prog_ciou_fwd: &teeny_cuda::device::program::CudaProgram<'_, YoloCiouLossForward<f32>>,
            prog_ciou_bwd: &teeny_cuda::device::program::CudaProgram<'_, YoloCiouLossBackward<f32>>,
            prog_cls_bwd:  &teeny_cuda::device::program::CudaProgram<'_, YoloBceClsLossBackward<f32>>,
            assigner:     &TaskAlignedAssigner,
            boxes:        &[f32],
            scores:       &[f32],
            gt_boxes_b:   &[Vec<[f32; 4]>],
            gt_cls_b:     &[Vec<usize>],
            loss_weight:  f32,
        ) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
            let b  = gt_boxes_b.len();
            let a  = self.grid.n_anchors;
            let nc = self.nc;

            assert_eq!(boxes.len(),  b * 4 * a);
            assert_eq!(scores.len(), b * nc * a);

            let mut d_boxes_out  = vec![0.0f32; b * 4 * a];
            let mut d_scores_out = vec![0.0f32; b * nc * a];

            for bi in 0..b {
                let boxes_i  = &boxes[bi * 4 * a .. (bi + 1) * 4 * a];
                let scores_i = &scores[bi * nc * a .. (bi + 1) * nc * a];

                // 1. Decode LTRB → XYWH (CPU).
                let xywh = self.grid.decode_ltrb_to_xywh(boxes_i);

                // 2. Assign GT targets (CPU).
                let assign = assigner.assign(
                    &xywh, scores_i,
                    &self.grid.cx, &self.grid.cy,
                    &gt_boxes_b[bi], &gt_cls_b[bi],
                );
                let n_pos: usize = assign.is_positive.iter().filter(|&&p| p).count();
                // Normalise by sum of soft targets, matching ultralytics target_scores_sum.
                let target_scores_sum: f32 = (0..a)
                    .filter(|&i| assign.is_positive[i])
                    .map(|i| assign.soft_target[i])
                    .sum::<f32>()
                    .max(1.0);
                let norm = loss_weight / target_scores_sum;

                // 3. CIoU backward for positive anchors.
                let mut d_xywh = vec![0.0f32; 4 * a];
                if n_pos > 0 {
                    let pos_idx: Vec<usize> = (0..a).filter(|&i| assign.is_positive[i]).collect();
                    let np = pos_idx.len();

                    let mut pred_pos   = vec![0.0f32; 4 * np];
                    let mut target_pos = vec![0.0f32; 4 * np];
                    for (j, &i) in pos_idx.iter().enumerate() {
                        for ch in 0..4 {
                            pred_pos[ch * np + j]   = xywh[ch * a + i];
                            target_pos[ch * np + j] = assign.target_boxes[ch * a + i];
                        }
                    }

                    // CIoU forward on GPU to get saved activations.
                    let (iou, v, alpha) = self.ciou_fwd_gpu(
                        device, prog_ciou_fwd, &pred_pos, &target_pos, np
                    )?;

                    // Box loss weight = soft_target (alignment-metric based), not raw IoU.
                    let dy_pos: Vec<f32> = pos_idx.iter()
                        .map(|&i| norm * assign.soft_target[i])
                        .collect();

                    let d_pred_pos = self.ciou_bwd_gpu(
                        device, prog_ciou_bwd,
                        &dy_pos, &pred_pos, &target_pos, &iou, &v, &alpha, np,
                    )?;

                    for (j, &i) in pos_idx.iter().enumerate() {
                        for ch in 0..4 { d_xywh[ch * a + i] = d_pred_pos[ch * np + j]; }
                    }
                }

                // 4. Decode backward: d_xywh → d_ltrb.
                let d_ltrb = self.grid.decode_backward(&d_xywh);
                d_boxes_out[bi * 4 * a .. (bi + 1) * 4 * a].copy_from_slice(&d_ltrb);

                // 5. BCE cls backward — all anchors with soft targets.
                //
                // Soft labels: positive anchor's class slot gets soft_target (∈ (0,max_iou])
                // rather than a hard 1.0.  All other slots (and all negatives) stay 0.
                let mut cls_target = vec![0.0f32; nc * a];
                for (i, &pos) in assign.is_positive.iter().enumerate() {
                    if pos {
                        let c = assign.target_cls[i];
                        if c < nc { cls_target[c * a + i] = assign.soft_target[i]; }
                    }
                }
                let dy_cls = vec![norm; a];

                let d_scores_i = self.cls_bwd_gpu(
                    device, prog_cls_bwd,
                    &dy_cls, scores_i, &cls_target, a, nc,
                )?;
                d_scores_out[bi * nc * a .. (bi + 1) * nc * a].copy_from_slice(&d_scores_i);
            }

            Ok((d_boxes_out, d_scores_out))
        }

        // ── GPU helpers ───────────────────────────────────────────────────────

        fn ciou_fwd_gpu(
            &self, device: &CudaDevice<'_>,
            prog: &teeny_cuda::device::program::CudaProgram<'_, YoloCiouLossForward<f32>>,
            pred: &[f32], target: &[f32], n: usize,
        ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
            let mut pred_buf   = device.buffer::<f32>(4 * n)?;
            let mut target_buf = device.buffer::<f32>(4 * n)?;
            let loss_buf   = device.buffer::<f32>(n)?;
            let iou_buf    = device.buffer::<f32>(n)?;
            let v_buf      = device.buffer::<f32>(n)?;
            let alpha_buf  = device.buffer::<f32>(n)?;
            pred_buf.to_device(pred)?;
            target_buf.to_device(target)?;
            let cfg = launch_config_with_grid(n.div_ceil(self.block_n as usize), prog);
            device.launch(prog, &cfg, (
                pred_buf.as_device_ptr(),
                target_buf.as_device_ptr(),
                loss_buf.as_device_ptr(),
                iou_buf.as_device_ptr(),
                v_buf.as_device_ptr(),
                alpha_buf.as_device_ptr(),
                n as i32,
            ))?;
            let (mut iou, mut v, mut alpha) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
            iou_buf.to_host(&mut iou)?;
            v_buf.to_host(&mut v)?;
            alpha_buf.to_host(&mut alpha)?;
            Ok((iou, v, alpha))
        }

        #[allow(clippy::too_many_arguments)]
        fn ciou_bwd_gpu(
            &self, device: &CudaDevice<'_>,
            prog: &teeny_cuda::device::program::CudaProgram<'_, YoloCiouLossBackward<f32>>,
            dy: &[f32], pred: &[f32], target: &[f32],
            iou: &[f32], v: &[f32], alpha: &[f32], n: usize,
        ) -> Result<Vec<f32>> {
            let mut dy_buf     = device.buffer::<f32>(n)?;
            let mut pred_buf   = device.buffer::<f32>(4 * n)?;
            let mut target_buf = device.buffer::<f32>(4 * n)?;
            let mut iou_buf    = device.buffer::<f32>(n)?;
            let mut v_buf      = device.buffer::<f32>(n)?;
            let mut alpha_buf  = device.buffer::<f32>(n)?;
            let d_pred_buf     = device.buffer::<f32>(4 * n)?;
            dy_buf.to_device(dy)?; pred_buf.to_device(pred)?; target_buf.to_device(target)?;
            iou_buf.to_device(iou)?; v_buf.to_device(v)?; alpha_buf.to_device(alpha)?;
            let cfg = launch_config_with_grid(n.div_ceil(self.block_n as usize), prog);
            device.launch(prog, &cfg, (
                dy_buf.as_device_ptr(),
                pred_buf.as_device_ptr(),
                target_buf.as_device_ptr(),
                iou_buf.as_device_ptr(),
                v_buf.as_device_ptr(),
                alpha_buf.as_device_ptr(),
                d_pred_buf.as_device_ptr(),
                n as i32,
            ))?;
            let mut out = vec![0.0f32; 4 * n];
            d_pred_buf.to_host(&mut out)?;
            Ok(out)
        }

        #[allow(clippy::too_many_arguments)]
        fn cls_bwd_gpu(
            &self, device: &CudaDevice<'_>,
            prog: &teeny_cuda::device::program::CudaProgram<'_, YoloBceClsLossBackward<f32>>,
            dy: &[f32], pred: &[f32], target: &[f32],
            n: usize, c: usize,
        ) -> Result<Vec<f32>> {
            let mut dy_buf     = device.buffer::<f32>(n)?;
            let mut pred_buf   = device.buffer::<f32>(c * n)?;
            let mut target_buf = device.buffer::<f32>(c * n)?;
            let d_pred_buf     = device.buffer::<f32>(c * n)?;
            dy_buf.to_device(dy)?; pred_buf.to_device(pred)?; target_buf.to_device(target)?;
            let cfg = launch_config_with_grid(n.div_ceil(self.block_n as usize), prog);
            device.launch(prog, &cfg, (
                dy_buf.as_device_ptr(),
                pred_buf.as_device_ptr(),
                target_buf.as_device_ptr(),
                d_pred_buf.as_device_ptr(),
                n as i32,
                c as i32,
            ))?;
            let mut out = vec![0.0f32; c * n];
            d_pred_buf.to_host(&mut out)?;
            Ok(out)
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::Yolo26Loss;
