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

// C2PSA graph-structure tests.
//
// C2PSA(c1, c2, n, e) architecture:
//   cv1:   Conv2d + BN + SiLU  (c_in → 2*c)
//   split: ChannelChunk × 2    (→ a [B,c,H,W] and b [B,c,H,W])
//   for each PSABlock (× n):
//     Attention sub-graph (all via Op::Custom):
//       psa_pack_qkv, flash_attention2_forward × 2,
//       psa_merge_attn_nchw, psa_extract_v_nchw
//     Add (pe merge inside attention)
//     Add (attn residual)
//     ffn0: Conv2d + BN + SiLU (c → 2c)
//     ffn1: Conv2d + BN        (2c → c)
//     Add (ffn residual)
//   ChannelCat (a, b)
//   cv2:   Conv2d + BN + SiLU  (2c → c_out)
//
// Shape contract: output is [B, c_out, H, W] for any spatial size.
// Graph contract: op counts verify correct wiring.

use teeny_core::graph::{DtypeRepr, Op, SymTensor};
use vision_rs::models::yolo::yolo26::blocks::c2psa::c2psa;

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

fn count_custom(y: &SymTensor, name: &str) -> usize {
    y.graph
        .borrow()
        .nodes
        .iter()
        .filter(|n| matches!(&n.op, Op::Custom { data } if data.name() == name))
        .count()
}

// ── n=1 (standard YOLO26 C2PSA) ──────────────────────────────────────────────
//
// C2PSA(256, 256, n=1, e=0.5): c=128, num_heads=2, key_dim=32.
// Custom ops per block: 5 (pack_qkv, fa2_lo, fa2_hi, merge_attn, extract_v)
// Add nodes per block:  3 (pe merge, attn residual, ffn residual)
// ChannelChunk:         2 (split cv1 output into a and b)
// ChannelCat:           1 (cat a and b before cv2)

#[test]
fn test_c2psa_n1_output_shape() {
    let x = sym_input(256, 16, 16);
    let y = c2psa::<f32>(256, 256, 1, 0.5)(x);
    assert_eq!(y.shape, vec![Some(2), Some(256), Some(16), Some(16)]);
}

#[test]
fn test_c2psa_n1_graph_structure() {
    let x = sym_input(256, 16, 16);
    let y = c2psa::<f32>(256, 256, 1, 0.5)(x);

    // PSA kernel custom ops (5 per block × 1 block = 5 total)
    assert_eq!(count_custom(&y, "psa_pack_qkv"), 1);
    assert_eq!(count_custom(&y, "flash_attention2_forward"), 2);
    assert_eq!(count_custom(&y, "psa_merge_attn_nchw"), 1);
    assert_eq!(count_custom(&y, "psa_extract_v_nchw"), 1);

    // Add nodes: pe_merge + attn_residual + ffn_residual = 3
    assert_eq!(count_op(&y, |op| matches!(op, Op::Add)), 3);

    assert_eq!(count_op(&y, |op| matches!(op, Op::ChannelChunk { .. })), 2);
    assert_eq!(count_op(&y, |op| matches!(op, Op::ChannelCat { .. })), 1);
}

// ── n=2 ───────────────────────────────────────────────────────────────────────

#[test]
fn test_c2psa_n2_graph_structure() {
    let x = sym_input(256, 16, 16);
    let y = c2psa::<f32>(256, 256, 2, 0.5)(x);

    assert_eq!(y.shape, vec![Some(2), Some(256), Some(16), Some(16)]);

    // 5 custom ops × 2 blocks
    assert_eq!(count_custom(&y, "psa_pack_qkv"), 2);
    assert_eq!(count_custom(&y, "flash_attention2_forward"), 4);
    assert_eq!(count_custom(&y, "psa_merge_attn_nchw"), 2);
    assert_eq!(count_custom(&y, "psa_extract_v_nchw"), 2);

    // 3 Add nodes × 2 blocks
    assert_eq!(count_op(&y, |op| matches!(op, Op::Add)), 6);
    assert_eq!(count_op(&y, |op| matches!(op, Op::ChannelChunk { .. })), 2);
    assert_eq!(count_op(&y, |op| matches!(op, Op::ChannelCat { .. })), 1);
}

// ── PSA custom-op nodes carry correct output shapes ───────────────────────────

#[test]
fn test_c2psa_attention_shapes() {
    // c=128, num_heads=2, key_dim=32, B=2, H=W=16
    // packed shape: [4, BH, N, KEY_DIM] = [4, 4, 256, 32]
    // fa2 shape:    [BH, N, KEY_DIM]    = [4, 256, 32]
    // merge shape:  [B, c, H, W]        = [2, 128, 16, 16]
    let x = sym_input(256, 16, 16);
    let y = c2psa::<f32>(256, 256, 1, 0.5)(x);
    let g = y.graph.borrow();

    let pack_node = g
        .nodes
        .iter()
        .find(|n| matches!(&n.op, Op::Custom { data } if data.name() == "psa_pack_qkv"))
        .expect("psa_pack_qkv node not found");
    // Shape: [B=2, sections=4, num_heads=2, N=16*16=256, KEY_DIM=32]
    assert_eq!(
        pack_node.shape,
        vec![Some(2), Some(4), Some(2), Some(256), Some(32)]
    );

    let merge_node = g
        .nodes
        .iter()
        .find(|n| matches!(&n.op, Op::Custom { data } if data.name() == "psa_merge_attn_nchw"))
        .expect("psa_merge_attn_nchw node not found");
    assert_eq!(
        merge_node.shape,
        vec![Some(2), Some(128), Some(16), Some(16)]
    );
}
