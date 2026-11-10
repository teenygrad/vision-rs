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

// C3k2 block tests for YOLO11n-n (depth=0.5, width=0.25).
//
// C3k2 inherits from C2f and overrides its bottleneck module list:
//
//   c3k=false  →  1×1 bottleneck  (two 1×1 Conv-BN-SiLU)
//   c3k=true   →  C3k bottleneck  (CSP block with two 3×3 Conv-BN-SiLU)
//
// Forward (C2f base):
//   cv1(c_in → 2·c)  →  ChannelChunk → [y0, y1]   (each B×c×H×W)
//   y1 → bottleneck₀ → … → bottleneckₙ
//   ChannelCat([y0, y1, f₀…fₙ])  →  cv2((2+n)·c → c_out)
//
// All convs are stride=1; H and W are preserved end-to-end.
// shortcut adds an Op::Add inside each bottleneck; it does not affect shape.
//
// YOLO11n-n channel progression (base × 0.25, repeats × 0.5 → n=1):
//   Layer 2:  C3k2(32→64,   n=1, c3k=False, e=0.25)  backbone P2→P3
//   Layer 6:  C3k2(128→128, n=1, c3k=True,  e=0.5)   backbone P4
//   Head 13:  C3k2(384→128, n=1, c3k=False, e=0.5)   after Concat(P5↑, P4)
//   Head 22:  C3k2(384→256, n=1, c3k=True,  e=0.5)   after Concat(head, P5)
//
// Tests cover:
//   1. Output shape (the primary contract)
//   2. Graph Op counts for Add / ChannelChunk / ChannelCat to verify that
//      the shortcut wiring and split/cat topology are structurally correct.
//      This is important because shape-only tests pass for any c_in==c_out
//      case, including an identity stub.

use teeny_core::graph::{DtypeRepr, Op, SymTensor};
use vision_rs::models::yolo::yolo26::blocks::c3k2::c3k2;

// ── helpers ───────────────────────────────────────────────────────────────────

fn sym_input(c: usize, h: usize, w: usize) -> SymTensor {
    let (t, _graph) = SymTensor::input(DtypeRepr::F32, vec![Some(2), Some(c), Some(h), Some(w)]);
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

// ── backbone: shallow layer (c3k=False, c_in≠c_out, e=0.25) ─────────────────
//
// YOLO11n-n layer 2: C3k2(32, 64, n=1, c3k=False, e=0.25)
// c = round(64 * 0.25) = 16
// Graph: 2 ChannelChunk (split), 1 ChannelCat (outer), 1 Add if shortcut

#[test]
fn test_c3k2_backbone_shallow_shortcut_true() {
    let x = sym_input(32, 16, 16);
    let y = c3k2::<f32>(32, 64, 1, false, true, 0.25)(x);

    assert_eq!(y.shape, vec![Some(2), Some(64), Some(16), Some(16)]);
    assert_eq!(count_op(&y, |op| matches!(op, Op::Add)), 1);
    assert_eq!(count_op(&y, |op| matches!(op, Op::ChannelChunk { .. })), 2);
    assert_eq!(count_op(&y, |op| matches!(op, Op::ChannelCat { .. })), 1);
}

#[test]
fn test_c3k2_backbone_shallow_shortcut_false() {
    let x = sym_input(32, 16, 16);
    let y = c3k2::<f32>(32, 64, 1, false, false, 0.25)(x);

    assert_eq!(y.shape, vec![Some(2), Some(64), Some(16), Some(16)]);
    assert_eq!(count_op(&y, |op| matches!(op, Op::Add)), 0);
}

// ── backbone: deep layer (c3k=True, c_in==c_out, e=0.5) ─────────────────────
//
// YOLO11n-n layer 6: C3k2(128, 128, n=1, c3k=True, e=0.5)
// c_in == c_out → shape alone cannot distinguish a correct implementation
// from an identity stub; graph structure is the discriminating check.
//
// c = 64; c3k_inner(64, shortcut) has two inner bottleneck_3x3 blocks.
// shortcut=true  → 2 Add nodes (one per inner bottleneck_3x3)
// shortcut=false → 0 Add nodes
// Graph: 2 ChannelChunk (outer split), 2 ChannelCat (1 inner C3k + 1 outer)

#[test]
fn test_c3k2_backbone_deep_shortcut_true() {
    let x = sym_input(128, 16, 16);
    let y = c3k2::<f32>(128, 128, 1, true, true, 0.5)(x);

    assert_eq!(y.shape, vec![Some(2), Some(128), Some(16), Some(16)]);
    assert_eq!(count_op(&y, |op| matches!(op, Op::Add)), 2);
    assert_eq!(count_op(&y, |op| matches!(op, Op::ChannelChunk { .. })), 2);
    assert_eq!(count_op(&y, |op| matches!(op, Op::ChannelCat { .. })), 2);
}

#[test]
fn test_c3k2_backbone_deep_shortcut_false() {
    let x = sym_input(128, 16, 16);
    let y = c3k2::<f32>(128, 128, 1, true, false, 0.5)(x);

    assert_eq!(y.shape, vec![Some(2), Some(128), Some(16), Some(16)]);
    assert_eq!(count_op(&y, |op| matches!(op, Op::Add)), 0);
    assert_eq!(count_op(&y, |op| matches!(op, Op::ChannelCat { .. })), 2);
}

// ── head: c3k=False after Concat (c_in >> c_out) ─────────────────────────────
//
// YOLO11n-n head layer 13: C3k2(384, 128, n=1, c3k=False, e=0.5)
// c_in=384 is Concat(P5↑=256, P4=128); c_out=128.

#[test]
fn test_c3k2_head_c3k_false() {
    let x = sym_input(384, 16, 16);
    let y = c3k2::<f32>(384, 128, 1, false, false, 0.5)(x);

    assert_eq!(y.shape, vec![Some(2), Some(128), Some(16), Some(16)]);
    assert_eq!(count_op(&y, |op| matches!(op, Op::Add)), 0);
}

// ── head: c3k=True with c_in ≠ c_out ─────────────────────────────────────────
//
// YOLO11n-n head layer 22: C3k2(384, 256, n=1, c3k=True, e=0.5)
// c_in=384, c_out=256; c3k=True path with c_in≠c_out is otherwise untested.

#[test]
fn test_c3k2_head_c3k_true() {
    let x = sym_input(384, 16, 16);
    let y = c3k2::<f32>(384, 256, 1, true, true, 0.5)(x);

    assert_eq!(y.shape, vec![Some(2), Some(256), Some(16), Some(16)]);
    assert_eq!(count_op(&y, |op| matches!(op, Op::Add)), 2);
    assert_eq!(count_op(&y, |op| matches!(op, Op::ChannelCat { .. })), 2);
}
