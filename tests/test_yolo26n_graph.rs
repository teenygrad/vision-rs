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

// Graph-level tests for the YOLO26 nano variant (depth=0.5, width=0.25, mc=1024).
//
// YOLO26n channel widths:
//   c0=16, c1=32, c2=64, c3=128, c4=256   (base × 0.25, capped at 1024)
//
// Repeat count: n = max(round(2 × 0.5), 1) = 1
//
// These tests verify the output tensor shapes and graph structure (op counts)
// produced by the symbolic tracer for a [2, 3, 64, 64] input.  No CUDA is
// needed — the graph IR is pure Rust.
//
// YOLO26n architecture (layer numbers match ultralytics yaml):
//
//   Backbone:
//     L0:  Conv(3→16, 3×3, s=2)          → 32×32
//     L1:  Conv(16→32, 3×3, s=2)         → 16×16
//     L2:  C3k2(32→64, n=1, c3k=F, e=0.25) → 16×16
//     L3:  Conv(64→64, 3×3, s=2)         → 8×8
//     L4:  C3k2(64→128, n=1, c3k=F, e=0.25) → 8×8  [P3]
//     L5:  Conv(128→128, 3×3, s=2)       → 4×4
//     L6:  C3k2(128→128, n=1, c3k=T, e=0.5) → 4×4  [P4]
//     L7:  Conv(128→256, 3×3, s=2)       → 2×2
//     L8:  C3k2(256→256, n=1, c3k=T, e=0.5) → 2×2
//     L9:  SPPF(256→256)                 → 2×2
//     L10: C2PSA(256→256, n=1, e=0.5)    → 2×2  [P5]
//
//   Head (neck + detect):
//     Up×2 → Cat([P5↑,P4]=384ch) → L13(384→128) → nk4
//     Up×2 → Cat([nk4↑,P3]=256ch) → L16(256→64) → p3d
//     L17(64→64,s=2) → Cat([L17,nk4]=192ch) → L19(192→128) → p4d
//     L20(128→128,s=2) → Cat([L20,P5]=384ch) → L22(384→256) → p5d
//     Detect([p3d,p4d,p5d], nc=80)
//
// Detect head (reg_max=1, nc=80):
//   c2 = max(16, 64/4=16, 4) = 16
//   c3 = max(64, min(80,100)) = 80
//
//   cv2[i]: Conv(ch[i]→16,3)→Conv(16→16,3)→Conv2d(16→4,1)
//   cv3[i]: DWConv(ch[i],3)→Conv(ch[i]→80,1)→DWConv(80,3)→Conv(80→80,1)→Conv2d(80→80,1)
//
//   flat concat: boxes=[B,4·A,1,1]  scores=[B,80·A,1,1]
//     A = H_p3·W_p3 + H_p4·W_p4 + H_p5·W_p5
//       = 8·8 + 4·4 + 2·2 = 64 + 16 + 4 = 84 (with 64×64 input)
//
//   boxes  = [B, 4·84, 1, 1] = [B, 336, 1, 1]
//   scores = [B, 80·84, 1, 1] = [B, 6720, 1, 1]

use teeny_core::graph::{DtypeRepr, Op, SymTensor};
use vision_rs::models::yolo::yolo26::{Yolo26Variant, blocks::detect::DetectHead, yolo26};

fn sym_input() -> SymTensor {
    let (t, _graph) = SymTensor::input(DtypeRepr::F32, vec![Some(2), Some(3), Some(64), Some(64)]);
    t
}

fn count_op(y: &SymTensor, pred: impl Fn(&Op) -> bool) -> usize {
    y.graph
        .borrow()
        .nodes
        .iter()
        .filter(|n| pred(&n.op))
        .count()
}

// ── Output shape tests ────────────────────────────────────────────────────────

#[test]
fn test_yolo26n_boxes_shape() {
    let x = sym_input();
    let out = yolo26::<f32>(80, &Yolo26Variant::N, DetectHead::OneToMany)(x);
    // 4 × (8²+4²+2²) = 4 × 84 = 336
    assert_eq!(out.boxes.shape, vec![Some(2), Some(336), Some(1), Some(1)]);
}

#[test]
fn test_yolo26n_scores_shape() {
    let x = sym_input();
    let out = yolo26::<f32>(80, &Yolo26Variant::N, DetectHead::OneToMany)(x);
    // 80 × (8²+4²+2²) = 80 × 84 = 6720
    assert_eq!(
        out.scores.shape,
        vec![Some(2), Some(6720), Some(1), Some(1)]
    );
}

// ── Backbone structural checks ────────────────────────────────────────────────

#[test]
fn test_yolo26n_has_attention() {
    // C2PSA implements attention as custom ops (PsaPackQkv, FlashAttn2, PsaMergeAttn,
    // PsaExtractV) rather than Op::Attention. Verify at least one custom op is present.
    let x = sym_input();
    let out = yolo26::<f32>(80, &Yolo26Variant::N, DetectHead::OneToMany)(x);
    let n_custom = count_op(&out.boxes, |op| matches!(op, Op::Custom { .. }));
    assert!(
        n_custom >= 1,
        "expected custom ops from C2PSA PSA attention, got 0"
    );
}

#[test]
fn test_yolo26n_sppf_maxpool_count() {
    // SPPF applies MaxPool2d exactly 3 times.
    let x = sym_input();
    let out = yolo26::<f32>(80, &Yolo26Variant::N, DetectHead::OneToMany)(x);
    assert_eq!(
        count_op(&out.boxes, |op| matches!(op, Op::MaxPool2d { .. })),
        3,
        "expected 3 MaxPool2d nodes from SPPF"
    );
}

#[test]
fn test_yolo26n_upsample_count() {
    // The head has 2 UpsampleNearest2d nodes (top-down neck path).
    let x = sym_input();
    let out = yolo26::<f32>(80, &Yolo26Variant::N, DetectHead::OneToMany)(x);
    assert_eq!(
        count_op(&out.boxes, |op| matches!(op, Op::UpsampleNearest2d { .. })),
        2,
        "expected 2 UpsampleNearest2d nodes in the FPN neck"
    );
}

#[test]
fn test_yolo26n_has_conv2d() {
    // Many Conv2d nodes (backbone + head + detect branches).
    let x = sym_input();
    let out = yolo26::<f32>(80, &Yolo26Variant::N, DetectHead::OneToMany)(x);
    let n_conv = count_op(&out.boxes, |op| matches!(op, Op::Conv2d { .. }));
    assert!(
        n_conv >= 30,
        "expected at least 30 Conv2d nodes, got {n_conv}"
    );
}

#[test]
fn test_yolo26n_has_batchnorm() {
    // BatchNorm2d follows (nearly) every Conv2d.
    let x = sym_input();
    let out = yolo26::<f32>(80, &Yolo26Variant::N, DetectHead::OneToMany)(x);
    let n_bn = count_op(&out.boxes, |op| matches!(op, Op::BatchNorm2d { .. }));
    assert!(
        n_bn >= 25,
        "expected at least 25 BatchNorm2d nodes, got {n_bn}"
    );
}

#[test]
fn test_yolo26n_channel_cat_count() {
    // ChannelCat nodes come from: C3k2 inner merges, neck concats, detect head.
    let x = sym_input();
    let out = yolo26::<f32>(80, &Yolo26Variant::N, DetectHead::OneToMany)(x);
    let n_cat = count_op(&out.scores, |op| matches!(op, Op::ChannelCat { .. }));
    // neck: 2 cats, detect: 2 cats (boxes + scores flat concat), plus C3k2 internal cats
    assert!(
        n_cat >= 10,
        "expected at least 10 ChannelCat nodes, got {n_cat}"
    );
}

// ── Batch dimension preserved ─────────────────────────────────────────────────

#[test]
fn test_yolo26n_batch_dim_preserved() {
    let x = sym_input();
    let out = yolo26::<f32>(80, &Yolo26Variant::N, DetectHead::OneToMany)(x);
    assert_eq!(out.boxes.shape[0], Some(2));
    assert_eq!(out.scores.shape[0], Some(2));
}

// ── Dynamic batch dimension (None in input) ───────────────────────────────────

#[test]
fn test_yolo26n_dynamic_batch() {
    let (x, _graph) = SymTensor::input(DtypeRepr::F32, vec![None, Some(3), Some(64), Some(64)]);
    let out = yolo26::<f32>(80, &Yolo26Variant::N, DetectHead::OneToMany)(x);
    // Batch dim should propagate as None (dynamic)
    assert_eq!(out.boxes.shape[0], None);
    assert_eq!(out.scores.shape[0], None);
}
