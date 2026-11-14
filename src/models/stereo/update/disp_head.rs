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

//! FoundationStereo's `DispHead` and the `EdgeNextConvEncoder` it uses.

use teeny_core::{
    dtype::{Float, FloatBytes},
    graph::SymTensor,
    name_scope::name_scope,
};

use super::ops::{add, channel_scale, conv, dwconv, gelu, relu};

/// `EdgeNextConvEncoder(dim, expan_ratio, kernel_size, norm=None)`
/// (`core/submodule.py:561`), as `DispHead` builds it:
///
/// ```python
/// x = dwconv(input)                 # depthwise k x k
/// x = x.permute(0, 2, 3, 1)         # NCHW -> NHWC
/// x = pwconv2(gelu(pwconv1(x)))     # nn.Linear over channels
/// x = gamma * x
/// return input + x.permute(0, 3, 1, 2)
/// ```
///
/// teeny-kernels cannot lower `Transpose`, so the two `nn.Linear` layers over the
/// channel axis are recorded as the 1x1 convs they are equivalent to (same weights,
/// `[out, in]` read as `[out, in, 1, 1]`), with no permutes. `gamma` is a per-channel
/// scale ([`channel_scale`]). GELU is the exact erf form, as `nn.GELU()` defaults to.
///
/// Only `norm=None` is implemented, which is what `DispHead` uses. Weights are
/// recorded as `dwconv.*`, `pwconv1.*`, `pwconv2.*` and `gamma` under the caller's
/// scope (see [`super::ops::torch_key`]).
pub fn edge_next_conv_encoder<D: Float + FloatBytes + Send + Sync + 'static>(
    dim: usize,
    expan_ratio: usize,
    kernel_size: usize,
) -> impl Fn(&SymTensor) -> SymTensor {
    let dw = dwconv::<D>("dwconv", dim, kernel_size);
    let pw1 = conv::<D>("pwconv1", dim, expan_ratio * dim, 1);
    let pw2 = conv::<D>("pwconv2", expan_ratio * dim, dim, 1);
    let gamma = channel_scale::<D>("gamma", dim);
    move |input: &SymTensor| {
        let x = dw(input);
        let x = pw2(&gelu::<D>(&pw1(&x)));
        add(input, &gamma(&x))
    }
}

/// `DispHead(input_dim, hidden_dim, output_dim=1)`:
///
/// ```python
/// nn.Sequential(
///     nn.Conv2d(input_dim, input_dim, 3, padding=1),   # conv.0
///     nn.ReLU(),                                       # conv.1
///     EdgeNextConvEncoder(input_dim, 4, 7, norm=None), # conv.2
///     EdgeNextConvEncoder(input_dim, 4, 7, norm=None), # conv.3
///     nn.Conv2d(input_dim, output_dim, 3, padding=1),  # conv.4
/// )
/// ```
///
/// (`hidden_dim` is accepted and unused by the PyTorch module, so it is not taken
/// here.) Weights are recorded under `conv.0`, `conv.2`, `conv.3` and `conv.4`.
pub fn disp_head<D: Float + FloatBytes + Send + Sync + 'static>(
    input_dim: usize,
    output_dim: usize,
) -> impl Fn(&SymTensor) -> SymTensor {
    let c0 = conv::<D>("0", input_dim, input_dim, 3);
    let e2 = edge_next_conv_encoder::<D>(input_dim, 4, 7);
    let e3 = edge_next_conv_encoder::<D>(input_dim, 4, 7);
    let c4 = conv::<D>("4", input_dim, output_dim, 3);
    move |x: &SymTensor| {
        let _g = name_scope("conv");
        let x = relu::<D>(c0(x));
        let x = {
            let _g = name_scope("2");
            e2(&x)
        };
        let x = {
            let _g = name_scope("3");
            e3(&x)
        };
        c4(&x)
    }
}
