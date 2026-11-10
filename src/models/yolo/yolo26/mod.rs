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

//! YOLO26 detection model.
//!
//! Architecture reference: `ultralytics/cfg/models/26/yolo26.yaml`.
//!
//! Returns raw `DetectOutput { boxes (B, 4·A), scores (B, nc·A) }` — the
//! training-mode layout.  For inference, apply `detect_decode` on `boxes`
//! using the anchor grid and strides [8, 16, 32] appropriate for the
//! runtime input resolution.

use teeny_core::{dtype::Float, graph::SymTensor, name_scope::name_scope};

use blocks::{
    c2psa::c2psa,
    c3k2::{c3k2, c3k2_psa},
    concat::concat,
    conv::conv,
    detect::{DetectHead, DetectOutput, DualDetectOutput, detect},
    sppf::sppf,
    upsample::upsample,
};

/// Building blocks (backbone/neck/head layers) used to assemble YOLO26.
pub mod blocks;

// ── Variant + Config ──────────────────────────────────────────────────────────

/// Per-variant scaling parameters derived from the yaml model definition.
#[derive(Clone, Copy, Debug)]
pub struct Yolo26Config {
    /// Depth multiplier, scaling block repeat counts.
    pub depth: f32,
    /// Width multiplier, scaling channel counts.
    pub width: f32,
    /// Maximum channel count cap.
    pub mc: usize,
}

/// YOLO26 model size variant.
#[derive(Clone, Copy, Debug)]
pub enum Yolo26Variant {
    /// Nano.
    N,
    /// Small.
    S,
    /// Medium.
    M,
    /// Large.
    L,
    /// Extra-large.
    XL,
}

impl Yolo26Variant {
    /// Returns this variant's depth/width/max-channel scaling parameters.
    pub fn config(&self) -> Yolo26Config {
        match self {
            Yolo26Variant::N => Yolo26Config {
                depth: 0.5,
                width: 0.25,
                mc: 1024,
            },
            Yolo26Variant::S => Yolo26Config {
                depth: 0.5,
                width: 0.50,
                mc: 1024,
            },
            Yolo26Variant::M => Yolo26Config {
                depth: 0.5,
                width: 1.00,
                mc: 512,
            },
            Yolo26Variant::L => Yolo26Config {
                depth: 1.0,
                width: 1.00,
                mc: 512,
            },
            Yolo26Variant::XL => Yolo26Config {
                depth: 1.0,
                width: 1.50,
                mc: 512,
            },
        }
    }
}

// ── Scaling helpers ───────────────────────────────────────────────────────────

/// Scale a yaml base channel count by width, capped at max_channels.
fn ch(base: usize, width: f32, mc: usize) -> usize {
    (base.min(mc) as f32 * width).round() as usize
}

/// Scale a yaml repeat count by depth, minimum 1.
fn rep(base: usize, depth: f32) -> usize {
    if base > 1 {
        ((base as f32 * depth).max(1.0)).round() as usize
    } else {
        base
    }
}

// ── Shared backbone + FPN neck ────────────────────────────────────────────────

/// Build the YOLO26 backbone and FPN neck layers.
///
/// Returns `(neck_fn, [c2, c3, c4])` where `neck_fn` maps an input image
/// tensor to the three FPN feature maps `(p3d, p4d, p5d)` consumed by the
/// detect head(s), and `[c2, c3, c4]` are the corresponding channel widths.
fn build_neck<D: Float + Send + Sync + 'static>(
    variant: &Yolo26Variant,
) -> (
    impl Fn(SymTensor) -> (SymTensor, SymTensor, SymTensor),
    usize,
    usize,
    usize,
) {
    let cfg = variant.config();
    let (d, w, mc) = (cfg.depth, cfg.width, cfg.mc);

    let c0 = ch(64, w, mc);
    let c1 = ch(128, w, mc);
    let c2 = ch(256, w, mc);
    let c3 = ch(512, w, mc);
    let c4 = ch(1024, w, mc);
    let n = rep(2, d);

    // Backbone layers (yaml 0–10)
    let l0 = conv::<D>(3, c0, 3, 2);
    let l1 = conv::<D>(c0, c1, 3, 2);
    let l2 = c3k2::<D>(c1, c2, n, false, true, 0.25);
    let l3 = conv::<D>(c2, c2, 3, 2);
    let l4 = c3k2::<D>(c2, c3, n, false, true, 0.25);
    let l5 = conv::<D>(c3, c3, 3, 2);
    let l6 = c3k2::<D>(c3, c3, n, true, true, 0.5);
    let l7 = conv::<D>(c3, c4, 3, 2);
    let l8 = c3k2::<D>(c4, c4, n, true, true, 0.5);
    let l9 = sppf::<D>(c4, c4, true);
    let l10 = c2psa::<D>(c4, c4, n, 0.5);

    // FPN neck layers (yaml 11–22)
    let up = upsample(2, 2);
    let cat = concat();
    let l13 = c3k2::<D>(c4 + c3, c3, n, true, true, 0.5);
    let l16 = c3k2::<D>(c3 + c3, c2, n, true, true, 0.5);
    let l17 = conv::<D>(c2, c2, 3, 2);
    let l19 = c3k2::<D>(c2 + c3, c3, n, true, true, 0.5);
    let l20 = conv::<D>(c3, c3, 3, 2);
    let l22 = c3k2_psa::<D>(c3 + c4, c4, 1, true, 0.5);

    let neck_fn = move |x: SymTensor| -> (SymTensor, SymTensor, SymTensor) {
        let x = {
            let _g = name_scope("model.0");
            l0(x)
        };
        let x = {
            let _g = name_scope("model.1");
            l1(x)
        };
        let x = {
            let _g = name_scope("model.2");
            l2(x)
        };
        let x = {
            let _g = name_scope("model.3");
            l3(x)
        };
        let p3 = {
            let _g = name_scope("model.4");
            l4(x)
        };
        let x = {
            let _g = name_scope("model.5");
            l5(p3.clone())
        };
        let p4 = {
            let _g = name_scope("model.6");
            l6(x)
        };
        let x = {
            let _g = name_scope("model.7");
            l7(p4.clone())
        };
        let x = {
            let _g = name_scope("model.8");
            l8(x)
        };
        let x = {
            let _g = name_scope("model.9");
            l9(x)
        };
        let p5 = {
            let _g = name_scope("model.10");
            l10(x)
        };

        let x = up(p5.clone());
        let x = cat(vec![x, p4]);
        let nk4 = {
            let _g = name_scope("model.13");
            l13(x)
        };

        let x = up(nk4.clone());
        let x = cat(vec![x, p3]);
        let p3d = {
            let _g = name_scope("model.16");
            l16(x)
        };

        let x = {
            let _g = name_scope("model.17");
            l17(p3d.clone())
        };
        let x = cat(vec![x, nk4]);
        let p4d = {
            let _g = name_scope("model.19");
            l19(x)
        };

        let x = {
            let _g = name_scope("model.20");
            l20(p4d.clone())
        };
        let x = cat(vec![x, p5]);
        let p5d = {
            let _g = name_scope("model.22");
            l22(x)
        };

        (p3d, p4d, p5d)
    };

    (neck_fn, c2, c3, c4)
}

// ── Public model constructors ─────────────────────────────────────────────────

/// YOLO26 single-head forward closure (inference or one-head training).
///
/// # Arguments
/// * `nc`      — number of detection classes (e.g. 80 for COCO)
/// * `variant` — model size variant (N / S / M / L / XL)
/// * `head`    — `OneToMany` (cv2/cv3, dense) or `OneToOne` (one2one_cv2/cv3, inference)
///
/// For dual-assignment training use [`yolo26_dual`] instead.
pub fn yolo26<D: Float + Send + Sync + 'static>(
    nc: usize,
    variant: &Yolo26Variant,
    head: DetectHead,
) -> impl Fn(SymTensor) -> DetectOutput {
    let (neck, c2, c3, c4) = build_neck::<D>(variant);
    let head_fn = detect::<D>(nc, &[c2, c3, c4], head);

    move |x: SymTensor| {
        let (p3d, p4d, p5d) = neck(x);
        {
            let _g = name_scope("model.23");
            head_fn(vec![p3d, p4d, p5d])
        }
    }
}

/// YOLO26 dual-head forward closure for training with consistent dual assignment.
///
/// Traces **both** the one2many head (cv2/cv3, TAL top_k=10) and the one2one
/// head (one2one_cv2/cv3, TAL top_k=1) in a single graph, sharing the backbone
/// and FPN neck.  The returned [`DualDetectOutput`] exposes both sets of
/// predictions so the caller can compute weighted losses for each.
///
/// Loss weighting schedule (ultralytics-style): early in training weight the
/// one2many head more heavily (it provides dense, stable gradients); gradually
/// shift weight toward the one2one head so it matches inference behaviour by the
/// end of training.  Use `Yolo26Loss::compute_grads_dual` (in
/// `models::yolo::loss::yolo26`, behind the `cuda` feature) to apply this.
pub fn yolo26_dual<D: Float + Send + Sync + 'static>(
    nc: usize,
    variant: &Yolo26Variant,
) -> impl Fn(SymTensor) -> DualDetectOutput {
    let (neck, c2, c3, c4) = build_neck::<D>(variant);
    let head_o2m = detect::<D>(nc, &[c2, c3, c4], DetectHead::OneToMany);
    let head_o2o = detect::<D>(nc, &[c2, c3, c4], DetectHead::OneToOne);

    move |x: SymTensor| {
        let (p3d, p4d, p5d) = neck(x);
        // one2many is traced before one2one so its terminal nodes get lower DAG
        // indices — the training loop relies on this for stable node identification.
        let one2many = {
            let _g = name_scope("model.23");
            head_o2m(vec![p3d.clone(), p4d.clone(), p5d.clone()])
        };
        let one2one = {
            let _g = name_scope("model.23");
            head_o2o(vec![p3d, p4d, p5d])
        };
        DualDetectOutput { one2many, one2one }
    }
}
