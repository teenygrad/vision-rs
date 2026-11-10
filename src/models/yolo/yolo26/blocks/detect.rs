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

//! YOLO26 Detect head.
//!
//! Implements `ultralytics.nn.modules.head.Detect(nc, ch)` with `reg_max=1`.
//!
//! Architecture per scale:
//!   `cv2[i]`: Conv(c_in,c2,3,1) → Conv(c2,c2,3,1) → Conv2d(c2,4*reg_max,1,bias=True)
//!   `cv3[i]`: DWConv(c_in,3,1) → Conv(c_in,c3,1,1) → DWConv(c3,3,1) → Conv(c3,c3,1,1)
//!           → Conv2d(c3,nc,1,bias=True)

use teeny_core::{
    dtype::Float,
    graph::{Op, SymTensor},
    name_scope::name_scope,
};

use super::conv::{conv, conv_plain, dwconv};

// ── Output types ──────────────────────────────────────────────────────────────

/// Output of the YOLO26 Detect head (training mode).
pub struct DetectOutput {
    /// Raw box predictions, concatenated across all FPN scales: (B, 4·A).
    /// A = H0·W0 + H1·W1 + H2·W2 total anchors.  reg_max = 1.
    pub boxes: SymTensor,
    /// Raw class logits, concatenated across all FPN scales: (B, nc·A).
    pub scores: SymTensor,
}

/// Combined output of both YOLO26 detect heads for dual-assignment training.
///
/// `one2many` (cv2/cv3) receives TAL assignment with top_k=10 and dominates
/// early training; `one2one` (one2one_cv2/cv3) receives TAL with top_k=1 and
/// is used exclusively at inference time.
pub struct DualDetectOutput {
    /// Output of the `cv2`/`cv3` (dense, TAL top_k=10) head.
    pub one2many: DetectOutput,
    /// Output of the `one2one_cv2`/`one2one_cv3` (TAL top_k=1) head.
    pub one2one: DetectOutput,
}

// ── Graph helper ──────────────────────────────────────────────────────────────

/// Concatenate (B, C, H_i, W_i) tensors along the flattened spatial dimension.
///
/// Treats each input as (B, C·H_i·W_i) and concatenates along that flat
/// channel/spatial axis, producing (B, sum(C·H_i·W_i), 1, 1) in the graph IR.
/// This matches the `torch.cat([t.view(B, C, -1) for t in ...], dim=-1)` pattern
/// used in the ultralytics Detect head forward pass.
fn channel_cat_flat(tensors: Vec<SymTensor>) -> SymTensor {
    let c_total: usize = tensors
        .iter()
        .map(|t| t.shape[1].unwrap_or(0) * t.shape[2].unwrap_or(1) * t.shape[3].unwrap_or(1))
        .sum();
    let first = &tensors[0];
    let shape = vec![first.shape[0], Some(c_total), Some(1), Some(1)];
    let inputs: Vec<usize> = tensors.iter().map(|t| t.node_id).collect();
    let node_id = first.graph.borrow_mut().add_node(
        Op::ChannelCat { c_total },
        inputs,
        first.dtype,
        shape.clone(),
    );
    SymTensor {
        node_id,
        graph: first.graph.clone(),
        dtype: first.dtype,
        shape,
    }
}

// ── Detect head ───────────────────────────────────────────────────────────────

/// Which weight namespace the detect head should bind to.
///
/// `OneToMany` maps to `cv2` / `cv3` (the training head used for dense
/// anchor assignment).  `OneToOne` maps to `one2one_cv2` / `one2one_cv3`
/// (the inference head used by ultralytics in eval mode — better mAP).
pub enum DetectHead {
    /// The dense training head (`cv2`/`cv3`).
    OneToMany,
    /// The inference head (`one2one_cv2`/`one2one_cv3`).
    OneToOne,
}

/// YOLO26 Detect head: 3 FPN feature maps → raw boxes + class logits.
///
/// Parameters:
///   - `nc`   — number of classes (e.g. 80 for COCO)
///   - `ch`   — channel counts for each FPN input, e.g. `[256, 512, 512]`
///   - `head` — which weight namespace to use (`OneToMany` for training,
///     `OneToOne` for inference — the latter loads `one2one_cv2/cv3`
///     weights and matches ultralytics eval-mode mAP)
///
/// Derived widths (matching ultralytics defaults, reg_max = 1):
///   - `c2 = max(16, ch[0]/4, reg_max*4)` — box branch hidden width (16 for n-size)
///   - `c3 = max(ch[0], min(nc, 100))`    — cls branch hidden width (80 for n-size)
pub fn detect<D: Float + 'static>(
    nc: usize,
    ch: &[usize],
    head: DetectHead,
) -> impl Fn(Vec<SymTensor>) -> DetectOutput + use<D> {
    let reg_max = 1usize;
    let c2 = [16usize, ch[0] / 4, reg_max * 4].into_iter().max().unwrap();
    let c3 = ch[0].max(nc.min(100));
    let (cv2_prefix, cv3_prefix) = match head {
        DetectHead::OneToMany => ("cv2", "cv3"),
        DetectHead::OneToOne => ("one2one_cv2", "one2one_cv3"),
    };

    // cv2[i]: Conv(c_in→c2,3) → Conv(c2→c2,3) → Conv2d(c2→4,1,bias)
    let cv2: Vec<Box<dyn Fn(SymTensor) -> SymTensor>> = ch
        .iter()
        .map(|&c_in| {
            let l1 = conv::<D>(c_in, c2, 3, 1);
            let l2 = conv::<D>(c2, c2, 3, 1);
            let l3 = conv_plain::<D>(c2, 4 * reg_max, 1, 1);
            Box::new(move |x: SymTensor| {
                let x = {
                    let _g = name_scope("0");
                    l1(x)
                };
                let x = {
                    let _g = name_scope("1");
                    l2(x)
                };
                {
                    let _g = name_scope("2");
                    l3(x)
                }
            }) as Box<dyn Fn(SymTensor) -> SymTensor>
        })
        .collect();

    // cv3[i]: Sequential(DWConv, Conv) → Sequential(DWConv, Conv) → nn.Conv2d
    // Ultralytics naming: cv3[i][0][0]=DWConv, cv3[i][0][1]=Conv,
    //                     cv3[i][1][0]=DWConv, cv3[i][1][1]=Conv, cv3[i][2]=plain conv
    let cv3: Vec<Box<dyn Fn(SymTensor) -> SymTensor>> = ch
        .iter()
        .map(|&c_in| {
            let dw1 = dwconv::<D>(c_in, 3, 1);
            let pw1 = conv::<D>(c_in, c3, 1, 1);
            let dw2 = dwconv::<D>(c3, 3, 1);
            let pw2 = conv::<D>(c3, c3, 1, 1);
            let out = conv_plain::<D>(c3, nc, 1, 1);
            Box::new(move |x: SymTensor| {
                let x = {
                    let _g = name_scope("0.0");
                    dw1(x)
                };
                let x = {
                    let _g = name_scope("0.1");
                    pw1(x)
                };
                let x = {
                    let _g = name_scope("1.0");
                    dw2(x)
                };
                let x = {
                    let _g = name_scope("1.1");
                    pw2(x)
                };
                {
                    let _g = name_scope("2");
                    out(x)
                }
            }) as Box<dyn Fn(SymTensor) -> SymTensor>
        })
        .collect();

    move |feats: Vec<SymTensor>| {
        let box_tensors: Vec<SymTensor> = feats
            .iter()
            .enumerate()
            .zip(cv2.iter())
            .map(|((i, x), f)| {
                let _g = name_scope(format!("{cv2_prefix}.{i}"));
                f(x.clone())
            })
            .collect();

        let cls_tensors: Vec<SymTensor> = feats
            .iter()
            .enumerate()
            .zip(cv3.iter())
            .map(|((i, x), f)| {
                let _g = name_scope(format!("{cv3_prefix}.{i}"));
                f(x.clone())
            })
            .collect();

        let boxes = channel_cat_flat(box_tensors);
        let scores = channel_cat_flat(cls_tensors);

        DetectOutput { boxes, scores }
    }
}
