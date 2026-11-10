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

//! Simplified TaskAlignedAssigner (CPU-side).
//!
//! For each GT box, alignment scores are:
//!   score = cls_score ^ alpha * iou ^ beta
//!
//! The top-k anchors per GT are selected as positives.  Conflicts (multiple GTs
//! assigning the same anchor) are broken by highest alignment score.

/// Result of assigning GT targets to anchors for a single image.
pub struct AssignResult {
    /// Whether each anchor is a positive. Shape: `[A]`.
    pub is_positive: Vec<bool>,
    /// GT box (cx, cy, w, h) assigned to each anchor (only valid for positives).
    /// Shape: `[4, A]` channels-first.
    pub target_boxes: Vec<f32>,
    /// GT class index assigned to each anchor (only valid for positives).
    /// Shape: `[A]`.
    pub target_cls: Vec<usize>,
    /// Soft target per anchor: `(align / max_align_for_gt) * max_iou_for_gt`.
    ///
    /// Zero for negatives.  Used as the soft class label AND box loss weight,
    /// matching the ultralytics E2ELoss normalisation.  Shape: `[A]`.
    pub soft_target: Vec<f32>,
}

/// Compute the IoU between a predicted XYWH box and a GT XYWH box.
fn iou_xywh(p: [f32; 4], g: [f32; 4]) -> f32 {
    let (px1, py1) = (p[0] - p[2] * 0.5, p[1] - p[3] * 0.5);
    let (px2, py2) = (p[0] + p[2] * 0.5, p[1] + p[3] * 0.5);
    let (gx1, gy1) = (g[0] - g[2] * 0.5, g[1] - g[3] * 0.5);
    let (gx2, gy2) = (g[0] + g[2] * 0.5, g[1] + g[3] * 0.5);

    let ix1 = px1.max(gx1);
    let iy1 = py1.max(gy1);
    let ix2 = px2.min(gx2);
    let iy2 = py2.min(gy2);
    let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
    let union = p[2] * p[3] + g[2] * g[3] - inter;
    inter / (union + 1e-7)
}

/// TaskAlignedAssigner parameters.
#[derive(Clone)]
pub struct TaskAlignedAssigner {
    /// Top-k anchors selected per GT.
    pub top_k: usize,
    /// Exponent on the predicted class confidence.
    pub alpha: f32,
    /// Exponent on the predicted IoU.
    pub beta: f32,
}

impl Default for TaskAlignedAssigner {
    fn default() -> Self {
        Self {
            top_k: 8,
            alpha: 0.5,
            beta: 6.0,
        }
    }
}

impl TaskAlignedAssigner {
    /// Assign GT boxes to anchors for a **single image**.
    ///
    /// # Arguments
    /// * `pred_xywh`   – decoded anchor predictions `[4, A]` channels-first
    /// * `pred_scores` – per-anchor class logits `[nc, A]` channels-first (sigmoid applied internally)
    /// * `anchor_cx`   – anchor centre x `[A]`
    /// * `anchor_cy`   – anchor centre y `[A]`
    /// * `gt_boxes`    – GT boxes `[[cx, cy, w, h]; M]`
    /// * `gt_cls`      – GT class indices `[M]`
    pub fn assign(
        &self,
        pred_xywh: &[f32],
        pred_scores: &[f32],
        anchor_cx: &[f32],
        anchor_cy: &[f32],
        gt_boxes: &[[f32; 4]],
        gt_cls: &[usize],
    ) -> AssignResult {
        let a = anchor_cx.len();
        let m = gt_boxes.len();
        let nc = pred_scores.len() / a;

        let mut target_boxes = vec![0.0f32; 4 * a];
        let mut target_cls = vec![0usize; a];
        let mut soft_target = vec![0.0f32; a];
        let mut assigned_score = vec![-1.0f32; a]; // -1 = unassigned
        let mut assigned_gt = vec![0usize; a];
        let mut is_positive = vec![false; a];

        if m == 0 {
            return AssignResult {
                is_positive,
                target_boxes,
                target_cls,
                soft_target,
            };
        }

        // Pre-compute sigmoid of predicted class scores.
        let sig: Vec<f32> = pred_scores
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();

        // Per-GT tracking for soft-target normalisation.
        let mut gt_max_align = vec![0.0f32; m]; // max alignment score among top-k
        let mut gt_max_iou = vec![0.0f32; m]; // max IoU among top-k

        // Flat list of (gt_idx, anchor_idx, align_score, iou) from top-k per GT.
        let mut candidates: Vec<(usize, usize, f32, f32)> = Vec::new();

        // Phase 1: per-GT top-k selection — collect candidates and track per-GT maxes.
        for (gi, &gt_box) in gt_boxes.iter().enumerate() {
            let gc = gt_cls[gi];
            let [gx, gy, gw, gh] = gt_box;

            let mut scores: Vec<(usize, f32, f32)> = (0..a)
                .filter_map(|ai| {
                    let cx = anchor_cx[ai];
                    let cy = anchor_cy[ai];
                    if cx < gx - gw * 0.5
                        || cx > gx + gw * 0.5
                        || cy < gy - gh * 0.5
                        || cy > gy + gh * 0.5
                    {
                        return None;
                    }

                    let p = [
                        pred_xywh[ai],
                        pred_xywh[a + ai],
                        pred_xywh[2 * a + ai],
                        pred_xywh[3 * a + ai],
                    ];
                    let iou = iou_xywh(p, gt_box).max(0.0);
                    let cls_score = if gc < nc { sig[gc * a + ai] } else { 0.0 };
                    let align = cls_score.powf(self.alpha) * iou.powf(self.beta);
                    Some((ai, align, iou))
                })
                .collect();

            if scores.is_empty() {
                continue;
            }

            scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scores.truncate(self.top_k);

            for &(ai, align, iou) in &scores {
                if align > gt_max_align[gi] {
                    gt_max_align[gi] = align;
                }
                if iou > gt_max_iou[gi] {
                    gt_max_iou[gi] = iou;
                }
                candidates.push((gi, ai, align, iou));
            }
        }

        // Phase 2: conflict resolution — highest alignment score wins — and box assignment.
        for (gi, ai, align, _iou) in &candidates {
            if *align > assigned_score[*ai] {
                assigned_score[*ai] = *align;
                assigned_gt[*ai] = *gi;
                is_positive[*ai] = true;
                target_cls[*ai] = gt_cls[*gi];
                let gt_box = gt_boxes[*gi];
                target_boxes[*ai] = gt_box[0];
                target_boxes[a + *ai] = gt_box[1];
                target_boxes[2 * a + *ai] = gt_box[2];
                target_boxes[3 * a + *ai] = gt_box[3];
            }
        }

        // Phase 3: soft target = (align / max_align_for_gt) * max_iou_for_gt.
        const EPS: f32 = 1e-8;
        for ai in 0..a {
            if is_positive[ai] {
                let gi = assigned_gt[ai];
                let align = assigned_score[ai];
                soft_target[ai] = (align / (gt_max_align[gi] + EPS)) * gt_max_iou[gi];
            }
        }

        AssignResult {
            is_positive,
            target_boxes,
            target_cls,
            soft_target,
        }
    }
}
