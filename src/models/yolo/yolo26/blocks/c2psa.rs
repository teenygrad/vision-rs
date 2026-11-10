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

use teeny_core::{
    dtype::Float,
    graph::{CustomData, Op, SymTensor},
    name_scope::name_scope,
};

use super::conv::{conv, conv_bn};
use crate::models::yolo::kernels::attention::psa::{
    FlashAttn2PsaOp, PsaExtractVOp, PsaMergeAttnOp, PsaPackQkvOp,
};

// ── Graph helpers ─────────────────────────────────────────────────────────────

fn channel_chunk(x: SymTensor, c_total: usize, chunk_c: usize, chunk_offset: usize) -> SymTensor {
    let op = Op::ChannelChunk {
        c_total,
        chunk_c,
        chunk_offset,
    };
    let shape = vec![x.shape[0], Some(chunk_c), x.shape[2], x.shape[3]];
    let node_id = x
        .graph
        .borrow_mut()
        .add_node(op, vec![x.node_id], x.dtype, shape.clone());
    SymTensor {
        node_id,
        graph: x.graph.clone(),
        dtype: x.dtype,
        shape,
    }
}

fn channel_cat(tensors: Vec<SymTensor>, c_total: usize) -> SymTensor {
    let first = &tensors[0];
    let shape = vec![
        first.shape[0],
        Some(c_total),
        first.shape[2],
        first.shape[3],
    ];
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

fn elem_add(a: SymTensor, b: SymTensor) -> SymTensor {
    let shape = a.shape.clone();
    let node_id =
        a.graph
            .borrow_mut()
            .add_node(Op::Add, vec![a.node_id, b.node_id], a.dtype, shape.clone());
    SymTensor {
        node_id,
        graph: a.graph.clone(),
        dtype: a.dtype,
        shape,
    }
}

// ── PSA Attention ─────────────────────────────────────────────────────────────

/// Multi-head self-attention with Flash Attention 2 + position encoding.
///
/// Matches `ultralytics.nn.modules.block.Attention.forward(x)`:
///   qkv conv+BN → pack_qkv → FA2×2 → merge_attn
///   ↕ + extract_v → pe_dw_conv+BN → (merge + pe) → proj_conv+BN
///
/// Input/output: `[B, c, H, W]`  (residual add is applied by the caller).
fn psa_attention<D: Float + Send + Sync + 'static>(
    c: usize,
    num_heads: usize,
    key_dim: usize,
) -> impl Fn(SymTensor) -> SymTensor {
    let qkv_h = num_heads * 4 * key_dim;
    let qkv_conv = conv_bn::<D>(c, qkv_h, 1, 1, 1);
    let pe_dw = conv_bn::<D>(c, c, 3, 1, c); // depthwise position encoding
    let proj = conv_bn::<D>(c, c, 1, 1, 1);

    move |x: SymTensor| {
        let h = x.shape[2].unwrap_or(1);
        let w = x.shape[3].unwrap_or(1);

        // [B, c, H, W] → [B, qkv_h, H, W]
        let qkv = {
            let _g = name_scope("qkv");
            qkv_conv(x)
        };

        // [B, qkv_h, H, W] → [4, BH, N, KEY_DIM]
        let packed = qkv.record_custom(
            CustomData::new(PsaPackQkvOp::<D>::new(key_dim as i32, num_heads)),
            &[],
            None,
        );

        // [4, BH, N, KEY_DIM] → [BH, N, KEY_DIM]  (V_lo, V_hi separately)
        let lo = packed.record_custom(
            CustomData::new(FlashAttn2PsaOp::<D>::new_lo(key_dim as i32)),
            &[],
            None,
        );
        let hi = packed.record_custom(
            CustomData::new(FlashAttn2PsaOp::<D>::new_hi(key_dim as i32)),
            &[],
            None,
        );

        // (lo, hi) [BH, N, KEY_DIM] → [B, c, H, W]
        let merged = lo.record_custom(
            CustomData::new(PsaMergeAttnOp::<D>::new(key_dim as i32, num_heads, h, w)),
            &[&hi],
            None,
        );

        // [B, qkv_h, H, W] → [B, c, H, W]  (V channels in NCHW for PE)
        let v_nchw = qkv.record_custom(
            CustomData::new(PsaExtractVOp::<D>::new(key_dim as i32, num_heads)),
            &[],
            None,
        );

        // PE depthwise conv, then add to merged attention
        let pe = {
            let _g = name_scope("pe");
            pe_dw(v_nchw)
        };
        let attn_pe = elem_add(merged, pe);

        // Final projection
        {
            let _g = name_scope("proj");
            proj(attn_pe)
        }
    }
}

// ── PSABlock ──────────────────────────────────────────────────────────────────

/// Single PSABlock iteration — attention residual + FFN residual.
///
/// Matches `ultralytics.nn.modules.block.PSABlock(c, attn_ratio=0.5, num_heads, shortcut=True)`.
pub(super) fn psa_block<D: Float + Send + Sync + 'static>(
    c: usize,
    num_heads: usize,
    key_dim: usize,
) -> impl Fn(SymTensor) -> SymTensor {
    let attn = psa_attention::<D>(c, num_heads, key_dim);
    let ffn0 = conv::<D>(c, 2 * c, 1, 1);
    let ffn1 = conv_bn::<D>(2 * c, c, 1, 1, 1);
    move |b: SymTensor| {
        let b = elem_add(b.clone(), {
            let _g = name_scope("attn");
            attn(b)
        });
        let ffn_out = {
            let tmp = {
                let _g = name_scope("ffn.0");
                ffn0(b.clone())
            };
            let _g = name_scope("ffn.1");
            ffn1(tmp)
        };
        elem_add(b, ffn_out)
    }
}

// ── C2PSA ─────────────────────────────────────────────────────────────────────

/// C2PSA: cross-stage partial network with PSA attention blocks.
///
/// Matches `ultralytics.nn.modules.block.C2PSA(c1, c2, n, e)`.
///
/// Forward:
///   h        = cv1(x)           // [B, 2*c, H, W]
///   [a, b]   = split(h, c)      // each [B, c, H, W]
///   b        = PSABlock(b) × n
///   output   = cv2(cat(a, b))   // [B, c2, H, W]
pub fn c2psa<D: Float + Send + Sync + 'static>(
    c_in: usize,
    c_out: usize,
    n: usize,
    e: f32,
) -> impl Fn(SymTensor) -> SymTensor {
    assert_eq!(c_in, c_out, "C2PSA requires c_in == c_out");
    let c = (c_out as f32 * e) as usize;
    let num_heads = c / 64;
    let key_dim = 32;

    let cv1 = conv::<D>(c_in, 2 * c, 1, 1);
    let cv2 = conv::<D>(2 * c, c_out, 1, 1);
    let blocks: Vec<Box<dyn Fn(SymTensor) -> SymTensor>> = (0..n)
        .map(|_| -> Box<dyn Fn(SymTensor) -> SymTensor> {
            Box::new(psa_block::<D>(c, num_heads, key_dim))
        })
        .collect();

    move |x: SymTensor| {
        let h = {
            let _g = name_scope("cv1");
            cv1(x)
        };
        let a = channel_chunk(h.clone(), 2 * c, c, 0);
        let mut b = channel_chunk(h, 2 * c, c, c);
        for (i, blk) in blocks.iter().enumerate() {
            let _g = name_scope(format!("m.{i}"));
            b = blk(b);
        }
        {
            let _g = name_scope("cv2");
            cv2(channel_cat(vec![a, b], 2 * c))
        }
    }
}
