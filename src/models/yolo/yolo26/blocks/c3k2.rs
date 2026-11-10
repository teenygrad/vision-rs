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
    graph::{Op, SymTensor},
    name_scope::name_scope,
};

use super::bottleneck::{bottleneck_3x3, bottleneck_std};
use super::c2psa::psa_block;
use super::conv::conv;

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

// ── C3k inner block ───────────────────────────────────────────────────────────

/// Matches ultralytics C3k(c, c, n=2, shortcut, e=0.5).
fn c3k_inner<D: Float + 'static>(c: usize, shortcut: bool) -> impl Fn(SymTensor) -> SymTensor {
    let c_h = ((c as f32) * 0.5) as usize;
    let cv1 = conv::<D>(c, c_h, 1, 1);
    let cv2 = conv::<D>(c, c_h, 1, 1);
    let cv3 = conv::<D>(2 * c_h, c, 1, 1);
    let m0 = bottleneck_3x3::<D>(c_h, shortcut);
    let m1 = bottleneck_3x3::<D>(c_h, shortcut);
    move |x: SymTensor| {
        let after_cv1 = {
            let _g = name_scope("cv1");
            cv1(x.clone())
        };
        let after_m0 = {
            let _g = name_scope("m.0");
            m0(after_cv1)
        };
        let after_m1 = {
            let _g = name_scope("m.1");
            m1(after_m0)
        };
        let after_cv2 = {
            let _g = name_scope("cv2");
            cv2(x)
        };
        let _g = name_scope("cv3");
        cv3(channel_cat(vec![after_m1, after_cv2], 2 * c_h))
    }
}

/// Sequential([Bottleneck, PSABlock]) inner for model.22.
///
/// Matches the special C3k2 variant used at P5/32 in YOLO26.
/// Named as "0" (Bottleneck) and "1" (PSABlock) — ultralytics Sequential indexing.
fn bottleneck_psa_seq<D: Float + Send + Sync + 'static>(
    c: usize,
    shortcut: bool,
    num_heads: usize,
    key_dim: usize,
) -> impl Fn(SymTensor) -> SymTensor {
    let bneck = bottleneck_std::<D>(c, shortcut);
    let psa = psa_block::<D>(c, num_heads, key_dim);
    move |x: SymTensor| {
        let x = {
            let _g = name_scope("0");
            bneck(x)
        };
        {
            let _g = name_scope("1");
            psa(x)
        }
    }
}

// ── C3k2 block ────────────────────────────────────────────────────────────────

/// C3k2: the primary feature-extraction block in YOLO11/26.
///
/// Matches `ultralytics.nn.modules.block.C3k2` (which inherits from C2f).
/// Forward:
///   h = cv1(x)                          // [N, 2*c, H, W]
///   [y0, y1] = h.chunk(2, dim=1)        // each [N, c, H, W]
///   parts = [y0, y1]
///   last = y1
///   for each bottleneck b:
///       last = b(last); parts.append(last)
///   return cv2(cat(parts))              // [(2+n)*c → c_out]
pub fn c3k2<D: Float + 'static>(
    c_in: usize,
    c_out: usize,
    n: usize,
    c3k: bool,
    shortcut: bool,
    e: f32,
) -> impl Fn(SymTensor) -> SymTensor {
    let c = (c_out as f32 * e) as usize;
    let cv1 = conv::<D>(c_in, 2 * c, 1, 1);
    let cv2 = conv::<D>((2 + n) * c, c_out, 1, 1);
    let bottlenecks: Vec<Box<dyn Fn(SymTensor) -> SymTensor>> = (0..n)
        .map(|_| -> Box<dyn Fn(SymTensor) -> SymTensor> {
            if c3k {
                Box::new(c3k_inner::<D>(c, shortcut))
            } else {
                Box::new(bottleneck_std::<D>(c, shortcut))
            }
        })
        .collect();
    move |x: SymTensor| {
        let h = {
            let _g = name_scope("cv1");
            cv1(x)
        };
        let y0 = channel_chunk(h.clone(), 2 * c, c, 0);
        let y1 = channel_chunk(h, 2 * c, c, c);
        let mut last = y1.clone();
        let mut parts = vec![y0, y1];
        for (i, b) in bottlenecks.iter().enumerate() {
            let _g = name_scope(format!("m.{i}"));
            last = b(last);
            parts.push(last.clone());
        }
        {
            let _g = name_scope("cv2");
            cv2(channel_cat(parts, (2 + n) * c))
        }
    }
}

/// C3k2 variant whose inner blocks are `Sequential([Bottleneck, PSABlock])`.
///
/// Used for model.22 (P5/32 detect-path block) in YOLO26. The inner channel
/// width `c = c_out * e`; `num_heads = c / 64`, `key_dim = 32` follow the
/// same convention as C2PSA.
pub fn c3k2_psa<D: Float + Send + Sync + 'static>(
    c_in: usize,
    c_out: usize,
    n: usize,
    shortcut: bool,
    e: f32,
) -> impl Fn(SymTensor) -> SymTensor {
    let c = (c_out as f32 * e) as usize;
    let num_heads = c / 64;
    let key_dim = 32;
    let cv1 = conv::<D>(c_in, 2 * c, 1, 1);
    let cv2 = conv::<D>((2 + n) * c, c_out, 1, 1);
    let bottlenecks: Vec<Box<dyn Fn(SymTensor) -> SymTensor>> = (0..n)
        .map(|_| -> Box<dyn Fn(SymTensor) -> SymTensor> {
            Box::new(bottleneck_psa_seq::<D>(c, shortcut, num_heads, key_dim))
        })
        .collect();
    move |x: SymTensor| {
        let h = {
            let _g = name_scope("cv1");
            cv1(x)
        };
        let y0 = channel_chunk(h.clone(), 2 * c, c, 0);
        let y1 = channel_chunk(h, 2 * c, c, c);
        let mut last = y1.clone();
        let mut parts = vec![y0, y1];
        for (i, b) in bottlenecks.iter().enumerate() {
            let _g = name_scope(format!("m.{i}"));
            last = b(last);
            parts.push(last.clone());
        }
        {
            let _g = name_scope("cv2");
            cv2(channel_cat(parts, (2 + n) * c))
        }
    }
}
