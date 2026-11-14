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
//! checkpoint tensors load by name. The one exception is a bare `nn.Parameter`
//! recorded as a layer (`EdgeNextConvEncoder.gamma`); [`torch_key`] maps those slots
//! back to their `state_dict` key.

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

use crate::models::stereo::kernels::{
    avg_pool2d_padded::AvgPool2dPaddedOp, gated_blend::GatedBlendOp, gelu_erf::GeluErfOp,
    interpolate_bilinear::InterpolateBilinearOp, scale::ScaleOp,
};

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

/// Depthwise `nn.Conv2d(c, c, k, padding=k // 2, groups=c)` with bias, under `name`.
pub fn dwconv<D: Float>(
    name: &'static str,
    c: usize,
    k: usize,
) -> impl Fn(&SymTensor) -> SymTensor {
    let p = k / 2;
    let layer =
        Conv2d::<D, SymTensor, SymTensor, 4>::new_grouped(c, c, (k, k), (1, 1), (p, p), true, c);
    move |x: &SymTensor| {
        let _g = name_scope(name);
        layer.call(x.clone())
    }
}

/// Per-channel scale `x * gamma[c]` for a bare `nn.Parameter` of shape `[C]`.
///
/// Recorded as a bias-free depthwise 1x1 conv, which computes exactly one multiply per
/// element. Its slot is `<name>.weight`; [`torch_key`] maps it back to `<name>`.
pub fn channel_scale<D: Float>(name: &'static str, c: usize) -> impl Fn(&SymTensor) -> SymTensor {
    let layer =
        Conv2d::<D, SymTensor, SymTensor, 4>::new_grouped(c, c, (1, 1), (1, 1), (0, 0), false, c);
    move |x: &SymTensor| {
        let _g = name_scope(name);
        layer.call(x.clone())
    }
}

/// Names of bare `nn.Parameter`s recorded through [`channel_scale`].
const BARE_PARAMS: &[&str] = &["gamma"];

/// The PyTorch `state_dict` key for a graph weight slot.
///
/// Identical except for [`channel_scale`] slots: `….gamma.weight` is the parameter
/// `….gamma`. The data needs no transform: `[C]` and `[C, 1, 1, 1]` are the same
/// values in the same order. Likewise a `nn.Linear` weight `[out, in]` loads unchanged
/// into the 1x1 conv that replaces it, `[out, in, 1, 1]`.
pub fn torch_key(slot: &str) -> String {
    for p in BARE_PARAMS {
        if let Some(stem) = slot.strip_suffix(".weight")
            && (stem == *p || stem.ends_with(&format!(".{p}")))
        {
            return stem.to_string();
        }
    }
    slot.to_string()
}

/// Exact GELU (`nn.GELU()`, erf form). teeny's `Op::Gelu` is the tanh approximation.
pub fn gelu<D: FloatBytes + Send + Sync + 'static>(x: &SymTensor) -> SymTensor {
    x.record_custom(CustomData::new(GeluErfOp::<D>::new(BLOCK)), &[], None)
}

/// `y = factor * x` for a constant `factor`.
pub fn scale<D: FloatBytes + Send + Sync + 'static>(x: &SymTensor, factor: f32) -> SymTensor {
    x.record_custom(CustomData::new(ScaleOp::<D>::new(factor, BLOCK)), &[], None)
}

/// FoundationStereo's `pool2x`: `F.avg_pool2d(x, 3, stride=2, padding=1)`.
pub fn pool2x<D: FloatBytes + Send + Sync + 'static>(x: &SymTensor) -> SymTensor {
    x.record_custom(
        CustomData::new(AvgPool2dPaddedOp::<D>::new(3, 2, 1, 16)),
        &[],
        None,
    )
}

/// FoundationStereo's `interp(x, dest)`:
/// `F.interpolate(x, dest.shape[2:], mode='bilinear', align_corners=True)`.
pub fn interp<D: FloatBytes + Send + Sync + 'static>(x: &SymTensor, dest: &SymTensor) -> SymTensor {
    x.record_custom(
        CustomData::new(InterpolateBilinearOp::<D>::new(true, 16)),
        &[dest],
        None,
    )
}

/// `nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False)`, recorded under `name`.
pub fn conv_no_bias<D: Float>(
    name: &'static str,
    c_in: usize,
    c_out: usize,
    k: usize,
) -> impl Fn(&SymTensor) -> SymTensor {
    let p = k / 2;
    let layer =
        Conv2d::<D, SymTensor, SymTensor, 4>::new(c_in, c_out, (k, k), (1, 1), (p, p), false);
    move |x: &SymTensor| {
        let _g = name_scope(name);
        layer.call(x.clone())
    }
}
