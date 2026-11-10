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

// Graph-level tests for the SPPF block.
//
// SPPF architecture:
//   cv1: Conv(c_in, c//2, 1, 1)  →  y
//   p1 = MaxPool2d(k=5, stride=1, pad=2)(y)
//   p2 = MaxPool2d(k=5, stride=1, pad=2)(p1)
//   p3 = MaxPool2d(k=5, stride=1, pad=2)(p2)
//   cv2: Conv(4*(c//2), c_out, 1, 1)(concat([y, p1, p2, p3]))
//
// Tests verify:
//   1. Output shape (same H, W; c_out channels)
//   2. Op types and counts in the traced graph

use teeny_core::graph::{DtypeRepr, Op, SymTensor};
use vision_rs::models::yolo::yolo26::blocks::sppf::sppf;

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

// ── Shape tests ───────────────────────────────────────────────────────────────

#[test]
fn test_sppf_output_shape() {
    // YOLO26n backbone: SPPF(256, 256)
    let x = sym_input(256, 20, 20);
    let y = sppf::<f32>(256, 256, false)(x);
    assert_eq!(y.shape, vec![Some(2), Some(256), Some(20), Some(20)]);
}

#[test]
fn test_sppf_spatial_preserved() {
    // H and W must not change — SPPF uses same-padding MaxPool
    let x = sym_input(64, 8, 8);
    let y = sppf::<f32>(64, 128, false)(x);
    assert_eq!(y.shape[2], Some(8));
    assert_eq!(y.shape[3], Some(8));
}

#[test]
fn test_sppf_different_cin_cout() {
    let x = sym_input(512, 10, 10);
    let y = sppf::<f32>(512, 256, false)(x);
    assert_eq!(y.shape, vec![Some(2), Some(256), Some(10), Some(10)]);
}

#[test]
fn test_sppf_batch_preserved() {
    let x = sym_input(128, 16, 16);
    let y = sppf::<f32>(128, 128, false)(x);
    assert_eq!(y.shape[0], Some(2));
}

// ── Op-count tests ────────────────────────────────────────────────────────────

#[test]
fn test_sppf_maxpool_count() {
    let x = sym_input(64, 8, 8);
    let y = sppf::<f32>(64, 64, false)(x);
    // 3 MaxPool2d nodes (p1, p2, p3)
    assert_eq!(count_op(&y, |op| matches!(op, Op::MaxPool2d { .. })), 3);
}

#[test]
fn test_sppf_concat_count() {
    let x = sym_input(64, 8, 8);
    let y = sppf::<f32>(64, 64, false)(x);
    // 1 ChannelCat node
    assert_eq!(count_op(&y, |op| matches!(op, Op::ChannelCat { .. })), 1);
}

#[test]
fn test_sppf_conv_count() {
    let x = sym_input(64, 8, 8);
    let y = sppf::<f32>(64, 64, false)(x);
    // 2 Conv2d nodes: cv1 and cv2
    assert_eq!(count_op(&y, |op| matches!(op, Op::Conv2d { .. })), 2);
}

#[test]
fn test_sppf_maxpool_padding_same() {
    // With k=5, stride=1, pad=2, output spatial dims equal input
    let x = sym_input(64, 13, 13);
    let y = sppf::<f32>(64, 64, false)(x);
    assert_eq!(y.shape, vec![Some(2), Some(64), Some(13), Some(13)]);
}
