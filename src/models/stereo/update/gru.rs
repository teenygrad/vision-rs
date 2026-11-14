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

//! FoundationStereo's convolutional GRU cells (`core/update.py`).

use teeny_core::{
    dtype::{Float, FloatBytes},
    graph::SymTensor,
    name_scope::name_scope,
};

use super::ops::{cat, conv, gated_blend, mul, relu, sigmoid, tanh};

/// `RaftConvGRU(hidden_dim, input_dim, kernel_size)`:
///
/// ```python
/// z = sigmoid(convz(hx))
/// r = sigmoid(convr(hx))
/// q = tanh(convq(cat([r * h, x])))
/// h = (1 - z) * h + z * q
/// ```
///
/// `hx` is passed in rather than formed here: `SelectiveConvGRU` builds it once (with
/// an extra conv) and shares it between its two GRUs. Weights are recorded as
/// `convz.*`, `convr.*`, `convq.*` under the caller's scope.
pub fn raft_conv_gru<D: Float + FloatBytes + Send + Sync + 'static>(
    hidden_dim: usize,
    input_dim: usize,
    kernel_size: usize,
) -> impl Fn(&SymTensor, &SymTensor, &SymTensor) -> SymTensor {
    let c = hidden_dim + input_dim;
    let convz = conv::<D>("convz", c, hidden_dim, kernel_size);
    let convr = conv::<D>("convr", c, hidden_dim, kernel_size);
    let convq = conv::<D>("convq", c, hidden_dim, kernel_size);
    move |h: &SymTensor, x: &SymTensor, hx: &SymTensor| {
        let z = sigmoid::<D>(convz(hx));
        let r = sigmoid::<D>(convr(hx));
        let q = tanh::<D>(convq(&cat(&[&mul(&r, h), x])));
        // (1 - z) * h + z * q
        gated_blend::<D>(&q, h, &z)
    }
}

/// `SelectiveConvGRU(hidden_dim, input_dim, small_kernel_size=1, large_kernel_size=3)`:
///
/// ```python
/// x = cat(x_list)
/// x = relu(conv0(x))
/// hx = relu(conv1(cat([x, h])))
/// h = small_gru(h, x, hx) * att + large_gru(h, x, hx) * (1 - att)
/// ```
///
/// `att` is the one-channel spatial attention map `[B, 1, H, W]`; `input_dim` is the
/// total channel count of `x_list`. Weights are recorded as `conv0.0.*`, `conv1.0.*`,
/// `small_gru.*` and `large_gru.*` under the caller's scope, matching the PyTorch
/// module (`conv0`/`conv1` are `nn.Sequential(Conv2d, ReLU)`).
pub fn selective_conv_gru<D: Float + FloatBytes + Send + Sync + 'static>(
    hidden_dim: usize,
    input_dim: usize,
) -> impl Fn(&SymTensor, &SymTensor, &[&SymTensor]) -> SymTensor {
    let conv0 = conv::<D>("0", input_dim, input_dim, 3);
    let conv1 = conv::<D>("0", input_dim + hidden_dim, input_dim + hidden_dim, 3);
    let small_gru = raft_conv_gru::<D>(hidden_dim, input_dim, 1);
    let large_gru = raft_conv_gru::<D>(hidden_dim, input_dim, 3);
    move |att: &SymTensor, h: &SymTensor, x_list: &[&SymTensor]| {
        let x = cat(x_list);
        let x = {
            let _g = name_scope("conv0");
            relu::<D>(conv0(&x))
        };
        let hx = {
            let _g = name_scope("conv1");
            relu::<D>(conv1(&cat(&[&x, h])))
        };
        let small = {
            let _g = name_scope("small_gru");
            small_gru(h, &x, &hx)
        };
        let large = {
            let _g = name_scope("large_gru");
            large_gru(h, &x, &hx)
        };
        // small * att + large * (1 - att)
        gated_blend::<D>(&small, &large, att)
    }
}
