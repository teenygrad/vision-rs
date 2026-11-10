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
    graph::SymTensor,
    name_scope::name_scope,
    nn::{Layer, activation::sigmoid::Silu, batchnorm::BatchNorm2d, conv2d::Conv2d},
};

/// Conv2d → BatchNorm2d → SiLU — the basic building block of YOLO26.
///
/// Matches `ultralytics.nn.modules.conv.Conv(c1, c2, k, s)`:
///   - bias=False (BN subsumes the bias term)
///   - autopad: p = k / 2 (same-padding for odd kernels)
///   - BN eps=0.001 matching the pretrained YOLO26n weights (trained with eps=1e-3)
pub fn conv<D: Float>(
    c_in: usize,
    c_out: usize,
    k: usize,
    s: usize,
) -> impl Fn(SymTensor) -> SymTensor {
    let p = k / 2;
    let conv2d =
        Conv2d::<D, SymTensor, SymTensor, 4>::new(c_in, c_out, (k, k), (s, s), (p, p), false);
    let bn = BatchNorm2d::<D, SymTensor, SymTensor, 4>::new(c_out).with_eps(0.001);
    let act = Silu::<D, SymTensor, 4>::new();
    move |x: SymTensor| {
        let x = {
            let _g = name_scope("conv");
            conv2d.call(x)
        };
        let x = {
            let _g = name_scope("bn");
            bn.call(x)
        };
        act.call(x)
    }
}

/// Conv2d(groups) → BatchNorm2d — no activation.
///
/// Used for layers in PSABlock that skip the SiLU activation (qkv, pe, proj, ffn1).
/// `g=1` for standard 1×1 convs; `g=c_in=c_out` for depthwise.
/// BN eps=0.001 matching the pretrained YOLO26n weights.
pub fn conv_bn<D: Float>(
    c_in: usize,
    c_out: usize,
    k: usize,
    s: usize,
    g: usize,
) -> impl Fn(SymTensor) -> SymTensor {
    let p = k / 2;
    let conv2d = Conv2d::<D, SymTensor, SymTensor, 4>::new_grouped(
        c_in,
        c_out,
        (k, k),
        (s, s),
        (p, p),
        false,
        g,
    );
    let bn = BatchNorm2d::<D, SymTensor, SymTensor, 4>::new(c_out).with_eps(0.001);
    move |x: SymTensor| {
        let x = {
            let _g = name_scope("conv");
            conv2d.call(x)
        };
        {
            let _g = name_scope("bn");
            bn.call(x)
        }
    }
}

/// Depth-wise Conv2d (groups = c) → BatchNorm2d → SiLU.
///
/// Matches `ultralytics DWConv(c, c, k)` with act=True.
/// BN eps=0.001 matching the pretrained YOLO26n weights.
pub fn dwconv<D: Float>(c: usize, k: usize, s: usize) -> impl Fn(SymTensor) -> SymTensor {
    let p = k / 2;
    let conv2d =
        Conv2d::<D, SymTensor, SymTensor, 4>::new_grouped(c, c, (k, k), (s, s), (p, p), false, c);
    let bn = BatchNorm2d::<D, SymTensor, SymTensor, 4>::new(c).with_eps(0.001);
    let act = Silu::<D, SymTensor, 4>::new();
    move |x: SymTensor| {
        let x = {
            let _g = name_scope("conv");
            conv2d.call(x)
        };
        let x = {
            let _g = name_scope("bn");
            bn.call(x)
        };
        act.call(x)
    }
}

/// Plain Conv2d with bias — no BatchNorm, no activation.
///
/// Matches `nn.Conv2d(c_in, c_out, k, bias=True)` from the ultralytics Detect head.
pub fn conv_plain<D: Float>(
    c_in: usize,
    c_out: usize,
    k: usize,
    s: usize,
) -> impl Fn(SymTensor) -> SymTensor {
    let p = k / 2;
    let layer =
        Conv2d::<D, SymTensor, SymTensor, 4>::new(c_in, c_out, (k, k), (s, s), (p, p), true);
    move |x| layer.call(x)
}
