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

// Graph-level tests for the Concat and Upsample blocks.
//
// Concat  — channels cat of two NCHW tensors along dim 1.
// Upsample — nearest-neighbor 2× spatial upsampling.
//
// Tests verify:
//   1. Output shape
//   2. Op type/count in the traced graph

use std::cell::RefCell;
use std::rc::Rc;
use teeny_core::graph::{DtypeRepr, Graph, Op, SymTensor};
use vision_rs::models::yolo::yolo26::blocks::{concat::concat, upsample::upsample};

fn sym_input(c: usize, h: usize, w: usize) -> SymTensor {
    let (t, _) = SymTensor::input(DtypeRepr::F32, vec![Some(2), Some(c), Some(h), Some(w)]);
    t
}

/// Create two inputs that share the same graph (simulates two branches of a network).
fn sym_two_inputs(c0: usize, c1: usize, h: usize, w: usize) -> (SymTensor, SymTensor) {
    let (x0, graph) = SymTensor::input(DtypeRepr::F32, vec![Some(2), Some(c0), Some(h), Some(w)]);
    let shape1 = vec![Some(2), Some(c1), Some(h), Some(w)];
    let node_id = graph
        .borrow_mut()
        .add_node(Op::Input, vec![], DtypeRepr::F32, shape1.clone());
    let x1 = SymTensor {
        node_id,
        graph: x0.graph.clone(),
        dtype: x0.dtype,
        shape: shape1,
    };
    (x0, x1)
}

/// Create three inputs that share the same graph.
fn sym_three_inputs(
    c0: usize,
    c1: usize,
    c2: usize,
    h: usize,
    w: usize,
) -> (SymTensor, SymTensor, SymTensor) {
    let (x0, graph) = SymTensor::input(DtypeRepr::F32, vec![Some(2), Some(c0), Some(h), Some(w)]);
    let s1 = vec![Some(2), Some(c1), Some(h), Some(w)];
    let s2 = vec![Some(2), Some(c2), Some(h), Some(w)];
    let nid1 = graph
        .borrow_mut()
        .add_node(Op::Input, vec![], DtypeRepr::F32, s1.clone());
    let nid2 = graph
        .borrow_mut()
        .add_node(Op::Input, vec![], DtypeRepr::F32, s2.clone());
    let x1 = SymTensor {
        node_id: nid1,
        graph: x0.graph.clone(),
        dtype: x0.dtype,
        shape: s1,
    };
    let x2 = SymTensor {
        node_id: nid2,
        graph: x0.graph.clone(),
        dtype: x0.dtype,
        shape: s2,
    };
    (x0, x1, x2)
}

/// Create two inputs with different spatial sizes on the same graph (for upsample+concat tests).
fn sym_two_inputs_diff_spatial(
    c: usize,
    h0: usize,
    w0: usize,
    h1: usize,
    w1: usize,
) -> (SymTensor, SymTensor) {
    let (x0, graph) = SymTensor::input(DtypeRepr::F32, vec![Some(2), Some(c), Some(h0), Some(w0)]);
    let shape1 = vec![Some(2), Some(c), Some(h1), Some(w1)];
    let node_id = graph
        .borrow_mut()
        .add_node(Op::Input, vec![], DtypeRepr::F32, shape1.clone());
    let x1 = SymTensor {
        node_id,
        graph: x0.graph.clone(),
        dtype: x0.dtype,
        shape: shape1,
    };
    (x0, x1)
}

fn count_op(y: &SymTensor, pred: impl Fn(&Op) -> bool) -> usize {
    y.graph
        .borrow()
        .nodes
        .iter()
        .filter(|n| pred(&n.op))
        .count()
}

fn _unused_graph() -> Rc<RefCell<Graph>> {
    Rc::new(RefCell::new(Graph::new()))
}

// ── Concat ────────────────────────────────────────────────────────────────────

#[test]
fn test_concat_two_equal_chunks_shape() {
    let (x0, x1) = sym_two_inputs(32, 32, 8, 8);
    let y = concat()(vec![x0, x1]);
    assert_eq!(y.shape, vec![Some(2), Some(64), Some(8), Some(8)]);
}

#[test]
fn test_concat_two_equal_chunks_op_count() {
    let (x0, x1) = sym_two_inputs(32, 32, 8, 8);
    let y = concat()(vec![x0, x1]);
    assert_eq!(count_op(&y, |op| matches!(op, Op::ChannelCat { .. })), 1);
    // Two Input nodes + one ChannelCat node = 3 total
    assert_eq!(y.graph.borrow().nodes.len(), 3);
}

#[test]
fn test_concat_unequal_chunks_shape() {
    // Typical YOLO neck: P3 (128 ch) and upsampled P4 (256 ch) concatenated
    let (p3, p4) = sym_two_inputs(128, 256, 40, 40);
    let y = concat()(vec![p3, p4]);
    assert_eq!(y.shape, vec![Some(2), Some(384), Some(40), Some(40)]);
}

#[test]
fn test_concat_three_tensors_shape() {
    let (a, b, c) = sym_three_inputs(16, 32, 48, 4, 4);
    let y = concat()(vec![a, b, c]);
    assert_eq!(y.shape, vec![Some(2), Some(96), Some(4), Some(4)]);
    assert_eq!(count_op(&y, |op| matches!(op, Op::ChannelCat { .. })), 1);
}

// ── Upsample ──────────────────────────────────────────────────────────────────

#[test]
fn test_upsample_2x_shape() {
    let x = sym_input(64, 20, 20);
    let y = upsample(2, 2)(x);
    assert_eq!(y.shape, vec![Some(2), Some(64), Some(40), Some(40)]);
}

#[test]
fn test_upsample_2x_op_count() {
    let x = sym_input(64, 20, 20);
    let y = upsample(2, 2)(x);
    assert_eq!(
        count_op(&y, |op| matches!(op, Op::UpsampleNearest2d { .. })),
        1
    );
    // Input + UpsampleNearest2d = 2 nodes
    assert_eq!(y.graph.borrow().nodes.len(), 2);
}

#[test]
fn test_upsample_anisotropic_shape() {
    // scale_h=2, scale_w=4 (unusual but should work)
    let x = sym_input(32, 10, 10);
    let y = upsample(2, 4)(x);
    assert_eq!(y.shape, vec![Some(2), Some(32), Some(20), Some(40)]);
}

#[test]
fn test_upsample_preserves_channels() {
    let x = sym_input(256, 5, 5);
    let y = upsample(2, 2)(x);
    assert_eq!(y.shape[1], Some(256));
}

// ── Combined: upsample then concat (YOLO FPN neck pattern) ───────────────────

#[test]
fn test_upsample_then_concat_shape() {
    // P5 feature (256 ch, 20×20) upsampled to 40×40 then concat with P4 (256 ch, 40×40)
    let (p5, p4) = sym_two_inputs_diff_spatial(256, 20, 20, 40, 40);
    let p5_up = upsample(2, 2)(p5);
    let y = concat()(vec![p5_up, p4]);
    assert_eq!(y.shape, vec![Some(2), Some(512), Some(40), Some(40)]);
    assert_eq!(
        count_op(&y, |op| matches!(op, Op::UpsampleNearest2d { .. })),
        1
    );
    assert_eq!(count_op(&y, |op| matches!(op, Op::ChannelCat { .. })), 1);
}
