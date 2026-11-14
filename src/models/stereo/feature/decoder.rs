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

//! The decoder half of FoundationStereo's `Feature` (`core/extractor.py`): three
//! `Conv2x_IN` upsampling stages from the EdgeNeXt maps, the concat with the ViT
//! features at 1/4, and `conv4`.
//!
//! ```python
//! x16 = self.deconv32_16(x32, x16)
//! x8 = self.deconv16_8(x16, x8)
//! x4 = self.deconv8_4(x8, x4)
//! x4 = self.conv4(torch.cat([x4, vit_feat], dim=1))
//! return [x4, x8, x16, x32]
//! ```
//!
//! `Conv2x_IN` interpolates its upsampled map when it does not match the skip
//! connection's size, which only happens for inputs that are not multiples of 32.
//! FoundationStereo pads its input to a multiple of 32 (`InputPadder(divis_by=32)`),
//! so the builders assert the shapes match instead. InstanceNorm here is PyTorch's
//! default (`affine=False`, eps 1e-5), so it has no weights.

use teeny_core::{
    dtype::{Float, FloatBytes},
    graph::{CustomData, SymTensor},
    name_scope::name_scope,
};

use crate::models::stereo::kernels::instance_norm3d::InstanceNorm3dOp;
use crate::models::stereo::update::ops::{add, cat, conv, conv_no_bias, conv_t, leaky, relu};

/// Channel widths of the decoder.
#[derive(Clone, Copy, Debug)]
pub struct DecoderDims {
    /// Backbone stage widths (`Feature.chans`).
    pub chans: [usize; 4],
    /// ViT feature channels (`DepthAnythingFeature` `features // 2`).
    pub vit: usize,
}

impl DecoderDims {
    /// The released checkpoints: `edgenext_small` widths and the `vitl` branch.
    pub fn foundation_stereo() -> Self {
        Self {
            chans: [48, 96, 160, 304],
            vit: 128,
        }
    }

    /// `Feature.d_out`: channels of the returned `[x4, x8, x16, x32]`.
    pub fn d_out(&self) -> [usize; 4] {
        let c = self.chans;
        [c[0] * 2 + self.vit, c[1] * 2, c[2] * 2, c[3]]
    }
}

/// `nn.InstanceNorm2d(c)`: per-(batch, channel) normalisation, no affine.
pub fn instance_norm<D: FloatBytes + Send + Sync + 'static>(x: &SymTensor) -> SymTensor {
    x.record_custom(CustomData::new(InstanceNorm3dOp::<D>::new(256)), &[], None)
}

/// `ResnetBasicBlock(c, c, norm_layer=nn.InstanceNorm2d)` (bias-free convs):
/// `relu(x + IN(conv2(relu(IN(conv1(x))))))`.
pub fn resnet_basic_block<D: Float + FloatBytes + Send + Sync + 'static>(
    x: &SymTensor,
    c: usize,
) -> SymTensor {
    let y = relu::<D>(instance_norm::<D>(&conv_no_bias::<D>("conv1", c, c, 3)(x)));
    let y = instance_norm::<D>(&conv_no_bias::<D>("conv2", c, c, 3)(&y));
    relu::<D>(add(x, &y))
}

/// `ResidualBlock(c, c, norm_fn='instance')` (`core/extractor.py`, convs with bias):
/// `relu(x + relu(IN(conv2(relu(IN(conv1(x)))))))`.
pub fn residual_block<D: Float + FloatBytes + Send + Sync + 'static>(
    x: &SymTensor,
    c: usize,
) -> SymTensor {
    let y = relu::<D>(instance_norm::<D>(&conv::<D>("conv1", c, c, 3)(x)));
    let y = relu::<D>(instance_norm::<D>(&conv::<D>("conv2", c, c, 3)(&y)));
    relu::<D>(add(x, &y))
}

/// `Conv2x_IN(c_in, c_out, deconv=True, concat=True)` on `(x, rem)`:
/// `conv1` is a bias-free k4 s2 p1 transposed conv, InstanceNorm and LeakyReLU; the
/// result is concatenated with `rem` and passed through a `ResnetBasicBlock` of
/// `2 * c_out` channels.
pub fn conv2x_in<D: Float + FloatBytes + Send + Sync + 'static>(
    x: &SymTensor,
    rem: &SymTensor,
    c_in: usize,
    c_out: usize,
) -> SymTensor {
    let y = {
        let _g = name_scope("conv1");
        let _c = name_scope("conv");
        conv_t::<D>(x, c_in, c_out, false)
    };
    assert_eq!(
        y.shape, rem.shape,
        "Conv2x_IN: upsampled map and skip differ; pad the input to a multiple of 32"
    );
    let y = cat(&[&leaky::<D>(&instance_norm::<D>(&y)), rem]);
    let _g = name_scope("conv2");
    resnet_basic_block::<D>(&y, 2 * c_out)
}

/// The decoder, from the backbone's `[x4, x8, x16, x32]` and `vit_feat`
/// `[B, vit, H/4, W/4]` to `Feature`'s outputs `[x4, x8, x16, x32]` (widths
/// [`DecoderDims::d_out`]). Weights under `deconv32_16`, `deconv16_8`, `deconv8_4`
/// and `conv4`, relative to the caller's scope.
pub fn decoder<D: Float + FloatBytes + Send + Sync + 'static>(
    dims: DecoderDims,
    feats: [&SymTensor; 4],
    vit_feat: &SymTensor,
) -> [SymTensor; 4] {
    let c = dims.chans;
    let [x4, x8, x16, x32] = feats;
    let x16 = {
        let _g = name_scope("deconv32_16");
        conv2x_in::<D>(x32, x16, c[3], c[2])
    };
    let x8 = {
        let _g = name_scope("deconv16_8");
        conv2x_in::<D>(&x16, x8, c[2] * 2, c[1])
    };
    let x4 = {
        let _g = name_scope("deconv8_4");
        conv2x_in::<D>(&x8, x4, c[1] * 2, c[0])
    };
    let w = dims.d_out()[0];
    let x4 = {
        let _g = name_scope("conv4");
        let y = cat(&[&x4, vit_feat]);
        let y = {
            let _g = name_scope("0");
            leaky::<D>(&instance_norm::<D>(&conv_no_bias::<D>("conv", w, w, 3)(&y)))
        };
        let y = {
            let _g = name_scope("1");
            residual_block::<D>(&y, w)
        };
        let _g = name_scope("2");
        residual_block::<D>(&y, w)
    };
    [x4, x8, x16, x32.clone()]
}
