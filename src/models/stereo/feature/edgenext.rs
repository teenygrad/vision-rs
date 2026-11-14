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

//! The EdgeNeXt-small backbone (timm `edgenext_small`) that `Feature` runs as
//! `self.stem` and `self.stages`.
//!
//! Every timm `nn.Linear` becomes a 1x1 conv on NCHW data (the `[out, in]` weight
//! loads unchanged as `[out, in, 1, 1]`), and every LayerNorm, whether timm's
//! `LayerNorm2d` or a channels-last `nn.LayerNorm` between two permutes, becomes
//! [`ChannelLayerNormOp`]. The permutes themselves disappear.

use teeny_core::{
    dtype::{Float, FloatBytes},
    graph::{CustomData, SymTensor},
    name_scope::name_scope,
    nn::{Layer, conv2d::Conv2d},
};

use crate::models::stereo::kernels::{
    channel_layer_norm::ChannelLayerNormOp, channel_slice::ChannelSliceOp,
    pos_fourier::PosFourierOp, xca_apply::XcaApplyOp, xca_gram::XcaGramOp,
};
use crate::models::stereo::update::ops::{add, cat, channel_scale, conv, dwconv, gelu};

/// EdgeNeXt's LayerNorm epsilon.
pub const EPS: f32 = 1e-6;
/// `edgenext_small` stage widths.
pub const DIMS: [usize; 4] = [48, 96, 160, 304];
/// `edgenext_small` depthwise kernel size per stage.
pub const KERNELS: [usize; 4] = [3, 5, 7, 9];
/// `edgenext_small` channel splits per stage's `SplitTransposeBlock` (stage 0 has none).
pub const SPLITS: [usize; 4] = [0, 2, 3, 4];
/// `CrossCovarianceAttn` heads.
pub const HEADS: usize = 8;

const BLOCK: i32 = 256;

/// LayerNorm over channels, `weight` and `bias` slots under `name`.
pub fn layer_norm<D: FloatBytes + Send + Sync + 'static>(
    name: &str,
    x: &SymTensor,
    c: usize,
) -> SymTensor {
    let _g = name_scope(name);
    x.record_custom(
        CustomData::new(ChannelLayerNormOp::<D>::new(c, EPS, BLOCK)),
        &[],
        None,
    )
}

/// `nn.Conv2d(c_in, c_out, k, stride=k)` with bias: a non-overlapping patch conv.
fn patch_conv<D: Float>(
    name: &str,
    x: &SymTensor,
    c_in: usize,
    c_out: usize,
    k: usize,
) -> SymTensor {
    let _g = name_scope(name);
    Conv2d::<D, SymTensor, SymTensor, 4>::new(c_in, c_out, (k, k), (k, k), (0, 0), true)
        .call(x.clone())
}

/// `stem`: `Conv2d(3, DIMS[0], 4, stride=4)`, then `LayerNorm2d`.
pub fn stem<D: Float + FloatBytes + Send + Sync + 'static>(x: &SymTensor) -> SymTensor {
    let _g = name_scope("stem");
    let x = patch_conv::<D>("0", x, 3, DIMS[0], 4);
    layer_norm::<D>("1", &x, DIMS[0])
}

/// A stage's `downsample`: `LayerNorm2d(c_in)`, then `Conv2d(c_in, c_out, 2, stride=2)`.
pub fn downsample<D: Float + FloatBytes + Send + Sync + 'static>(
    x: &SymTensor,
    c_in: usize,
    c_out: usize,
) -> SymTensor {
    let _g = name_scope("downsample");
    let x = layer_norm::<D>("0", x, c_in);
    patch_conv::<D>("1", &x, c_in, c_out, 2)
}

/// timm's EdgeNeXt `ConvBlock` (`dim == dim_out`, so the shortcut is the input):
///
/// ```python
/// x = conv_dw(x)                  # depthwise k x k
/// x = mlp(norm(x))                # LayerNorm, fc1 (4x), exact GELU, fc2
/// x = shortcut + gamma * x
/// ```
pub fn conv_block<D: Float + FloatBytes + Send + Sync + 'static>(
    x: &SymTensor,
    c: usize,
    k: usize,
) -> SymTensor {
    let y = dwconv::<D>("conv_dw", c, k)(x);
    mlp_residual::<D>(x, &y, c)
}

/// The inverted bottleneck both block types end with:
/// `x + gamma * fc2(gelu(fc1(norm(y))))`.
fn mlp_residual<D: Float + FloatBytes + Send + Sync + 'static>(
    x: &SymTensor,
    y: &SymTensor,
    c: usize,
) -> SymTensor {
    let y = layer_norm::<D>("norm", y, c);
    let y = {
        let _g = name_scope("mlp");
        let h = gelu::<D>(&conv::<D>("fc1", c, 4 * c, 1)(&y));
        conv::<D>("fc2", 4 * c, c, 1)(&h)
    };
    add(x, &channel_scale::<D>("gamma", c)(&y))
}

fn channel_slice<D: FloatBytes + Send + Sync + 'static>(
    x: &SymTensor,
    start: usize,
    len: usize,
) -> SymTensor {
    x.record_custom(
        CustomData::new(ChannelSliceOp::<D>::new(start, len, BLOCK)),
        &[],
        None,
    )
}

/// timm's `CrossCovarianceAttn` on NCHW data, `qkv` and `proj` as 1x1 convs:
/// attention across channels, per head, with a learned temperature.
pub fn xca<D: Float + FloatBytes + Send + Sync + 'static>(x: &SymTensor, c: usize) -> SymTensor {
    let _g = name_scope("xca");
    let qkv = conv::<D>("qkv", c, 3 * c, 1)(x);
    let attn = qkv.record_custom(
        CustomData::new(XcaGramOp::<D>::new(c, HEADS, BLOCK)),
        &[],
        None,
    );
    let out = attn.record_custom(
        CustomData::new(XcaApplyOp::<D>::new(c, HEADS, BLOCK)),
        &[&qkv],
        None,
    );
    conv::<D>("proj", c, c, 1)(&out)
}

/// timm's `SplitTransposeBlock` with `splits` channel groups (`num_scales`):
///
/// ```python
/// spx = x.chunk(splits, dim=1)
/// sp = convs[0](spx[0]); spo = [sp]
/// for i in 1..splits-1: sp = convs[i](sp + spx[i]); spo.append(sp)
/// x = cat(spo + [spx[-1]])                       # depthwise 3x3 cascade
/// x = x + pos_embd(x)                             # if use_pos
/// x = x + gamma_xca * xca(norm_xca(x))
/// x = shortcut + gamma * mlp(norm(x))
/// ```
///
/// Chunks are `ceil(c / splits)` wide, the last taking the remainder, as
/// `torch.chunk` gives them.
pub fn split_transpose_block<D: Float + FloatBytes + Send + Sync + 'static>(
    x: &SymTensor,
    c: usize,
    splits: usize,
    use_pos: bool,
) -> SymTensor {
    let width = c.div_ceil(splits);
    let chunk = |i: usize| channel_slice::<D>(x, i * width, width.min(c - i * width));

    let mut parts = Vec::new();
    {
        let _g = name_scope("convs");
        let mut sp = chunk(0);
        for i in 0..splits - 1 {
            if i > 0 {
                sp = add(&sp, &chunk(i));
            }
            let _i = name_scope(i.to_string());
            sp = Conv2d::<D, SymTensor, SymTensor, 4>::new_grouped(
                width,
                width,
                (3, 3),
                (1, 1),
                (1, 1),
                true,
                width,
            )
            .call(sp);
            parts.push(sp.clone());
        }
    }
    parts.push(chunk(splits - 1));
    let mut y = cat(&parts.iter().collect::<Vec<_>>());

    if use_pos {
        let _g = name_scope("pos_embd");
        let feats = y.record_custom(CustomData::new(PosFourierOp::<D>::new(BLOCK)), &[], None);
        let pos = conv::<D>("token_projection", 64, c, 1)(&feats);
        y = add(&y, &pos);
    }
    let a = xca::<D>(&layer_norm::<D>("norm_xca", &y, c), c);
    let y2 = add(&y, &channel_scale::<D>("gamma_xca", c)(&a));
    mlp_residual::<D>(x, &y2, c)
}

/// The whole backbone as `Feature.forward` runs it, `self.stem` then `self.stages`,
/// returning `[x4, x8, x16, x32]` (48, 96, 160 and 304 channels at 1/4 to 1/32 of the
/// input). Weights under `stem.*` and `stages.*`, relative to the caller's scope.
pub fn backbone<D: Float + FloatBytes + Send + Sync + 'static>(x: &SymTensor) -> [SymTensor; 4] {
    let mut x = stem::<D>(x);
    let _s = name_scope("stages");
    let depths = [3, 3, 9, 3];
    let mut outs = Vec::new();
    for i in 0..4 {
        let _i = name_scope(i.to_string());
        if i > 0 {
            x = downsample::<D>(&x, DIMS[i - 1], DIMS[i]);
        }
        let _b = name_scope("blocks");
        for b in 0..depths[i] {
            let _n = name_scope(b.to_string());
            x = if SPLITS[i] > 0 && b == depths[i] - 1 {
                split_transpose_block::<D>(&x, DIMS[i], SPLITS[i], i == 1)
            } else {
                conv_block::<D>(&x, DIMS[i], KERNELS[i])
            };
        }
        outs.push(x.clone());
    }
    match outs.try_into() {
        Ok(o) => o,
        Err(_) => unreachable!("four stages"),
    }
}
