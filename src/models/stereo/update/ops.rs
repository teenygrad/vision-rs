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

//! Small graph-building helpers shared by the refinement-head blocks.
//!
//! Layer names follow PyTorch's `state_dict` keys: wrap a call in
//! `name_scope("convz")` and its weight slot is recorded as `convz.weight`, so
//! checkpoint tensors load by name.

use teeny_core::{
    dtype::{Float, FloatBytes},
    graph::{CustomData, DtypeRepr, Op, Shape, SymTensor},
    name_scope::name_scope,
    nn::{
        Layer,
        activation::{relu::Relu, sigmoid::Sigmoid, tanh::Tanh},
        conv2d::Conv2d,
    },
};

use crate::models::stereo::kernels::gated_blend::GatedBlendOp;

/// Launch block for the element-wise custom ops built here.
const BLOCK: i32 = 256;

fn wrap(like: &SymTensor, node_id: usize, shape: Shape) -> SymTensor {
    SymTensor {
        node_id,
        graph: like.graph.clone(),
        dtype: like.dtype,
        shape,
    }
}

/// Adds another `Input` placeholder to `like`'s graph.
///
/// `RuntimeModel::run` feeds inputs to `Input` nodes in topological order; for
/// placeholders with no parents that is the order they were created in.
pub fn input_like(like: &SymTensor, shape: Shape) -> SymTensor {
    let node_id =
        like.graph
            .borrow_mut()
            .add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
    wrap(like, node_id, shape)
}

fn binary(op: Op, a: &SymTensor, b: &SymTensor) -> SymTensor {
    assert_eq!(
        a.shape, b.shape,
        "{op:?}: operand shapes differ (no broadcasting)"
    );
    let shape = a.shape.clone();
    let node_id =
        a.graph
            .borrow_mut()
            .add_node(op, vec![a.node_id, b.node_id], a.dtype, shape.clone());
    wrap(a, node_id, shape)
}

/// Element-wise `a + b` (same shape).
pub fn add(a: &SymTensor, b: &SymTensor) -> SymTensor {
    binary(Op::Add, a, b)
}

/// Element-wise `a * b` (same shape).
pub fn mul(a: &SymTensor, b: &SymTensor) -> SymTensor {
    binary(Op::Mul, a, b)
}

/// `torch.cat(tensors, dim=1)` for `[B, C, H, W]` tensors.
pub fn cat(tensors: &[&SymTensor]) -> SymTensor {
    let first = tensors[0];
    let c_total: usize = tensors
        .iter()
        .map(|t| t.shape[1].expect("cat: channel count must be static"))
        .sum();
    let shape = vec![
        first.shape[0],
        Some(c_total),
        first.shape[2],
        first.shape[3],
    ];
    let inputs = tensors.iter().map(|t| t.node_id).collect();
    let node_id = first.graph.borrow_mut().add_node(
        Op::ChannelCat { c_total },
        inputs,
        first.dtype,
        shape.clone(),
    );
    wrap(first, node_id, shape)
}

/// `a * g + b * (1 - g)`; `g` has `a`'s channel count or one channel.
pub fn gated_blend<D: FloatBytes + Send + Sync + 'static>(
    a: &SymTensor,
    b: &SymTensor,
    g: &SymTensor,
) -> SymTensor {
    assert_eq!(a.shape, b.shape, "gated_blend: a and b shapes differ");
    a.record_custom(
        CustomData::new(GatedBlendOp::<D>::new(BLOCK)),
        &[b, g],
        None,
    )
}

/// `nn.Conv2d(c_in, c_out, k, padding=k // 2)` with bias, recorded under `name`.
pub fn conv<D: Float>(
    name: &'static str,
    c_in: usize,
    c_out: usize,
    k: usize,
) -> impl Fn(&SymTensor) -> SymTensor {
    let p = k / 2;
    let layer =
        Conv2d::<D, SymTensor, SymTensor, 4>::new(c_in, c_out, (k, k), (1, 1), (p, p), true);
    move |x: &SymTensor| {
        let _g = name_scope(name);
        layer.call(x.clone())
    }
}

/// ReLU.
pub fn relu<D: Float>(x: SymTensor) -> SymTensor {
    Relu::<D, SymTensor, 4>::new().call(x)
}

/// Logistic sigmoid.
pub fn sigmoid<D: Float>(x: SymTensor) -> SymTensor {
    Sigmoid::<D, SymTensor, 4>::new().call(x)
}

/// Hyperbolic tangent.
pub fn tanh<D: Float>(x: SymTensor) -> SymTensor {
    Tanh::<D, SymTensor, 4>::new().call(x)
}
