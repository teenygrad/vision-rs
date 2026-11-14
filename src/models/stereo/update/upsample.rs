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

//! FoundationStereo's final disparity upsampling, `FoundationStereo.upsample_disp`
//! (`core/foundation_stereo.py`), from the 1/4-resolution refined disparity to full
//! resolution.

use teeny_core::{
    dtype::{Float, FloatBytes},
    graph::{CustomData, SymTensor},
    name_scope::name_scope,
};

use super::ops::{cat, conv_no_bias, conv_t, leaky};
use crate::models::stereo::kernels::{
    channel_softmax::ChannelSoftmaxOp, context_upsample::ContextUpsampleOp,
};

/// `upsample_disp(disp, mask_feat_4, stem_2x)`:
///
/// ```python
/// xspx = spx_2_gru(mask_feat_4, stem_2x)      # Conv2x(c, c, deconv=True, bn=False), to 1/2 res
/// spx_pred = spx_gru(xspx)                    # ConvTranspose2d(2c, 9, 4, 2, 1), to full res
/// spx_pred = F.softmax(spx_pred, 1)
/// up_disp = context_upsample(disp * 4., spx_pred).unsqueeze(1)
/// ```
///
/// with `Conv2x` as
///
/// ```python
/// x = leaky_relu(ConvTranspose2d(c, c, 4, 2, 1, bias=False)(mask_feat_4))   # conv1
/// x = cat([x, stem_2x])
/// x = leaky_relu(Conv2d(2c, 2c, 3, padding=1, bias=False)(x))                # conv2
/// ```
///
/// `c` is 32 in the checkpoints. `disp` is `[B, 1, h, w]` at 1/4 resolution,
/// `mask_feat_4` is `[B, c, h, w]` (the update block's mask), `stem_2x` is
/// `[B, c, 2h, 2w]`; the result is `[B, 1, 4h, 4w]`. `Conv2x` resizes its upsampled
/// branch when it does not match `stem_2x`; that only happens for image sizes that
/// are not a multiple of 32, which FoundationStereo pads away, so here it is an error.
///
/// Weights are recorded as `spx_2_gru.conv1.conv.weight`,
/// `spx_2_gru.conv2.conv.weight` and `spx_gru.0.{weight,bias}`, matching the model's
/// `state_dict` under the caller's scope.
pub fn upsample_disp<D: Float + FloatBytes + Send + Sync + 'static>(
    c: usize,
) -> impl Fn(&SymTensor, &SymTensor, &SymTensor) -> SymTensor {
    let conv2 = conv_no_bias::<D>("conv", 2 * c, 2 * c, 3);
    move |disp: &SymTensor, mask_feat_4: &SymTensor, stem_2x: &SymTensor| {
        let xspx = {
            let _g = name_scope("spx_2_gru");
            let x = {
                let _g = name_scope("conv1");
                let _g2 = name_scope("conv");
                conv_t::<D>(mask_feat_4, c, c, false)
            };
            let x = leaky::<D>(&x);
            assert_eq!(
                (x.shape[2], x.shape[3]),
                (stem_2x.shape[2], stem_2x.shape[3]),
                "upsample_disp: stem_2x must be exactly twice mask_feat_4's size \
                 (pad the image to a multiple of 32)"
            );
            let x = cat(&[&x, stem_2x]);
            let _g = name_scope("conv2");
            leaky::<D>(&conv2(&x))
        };
        let spx = {
            let _g = name_scope("spx_gru");
            let _g2 = name_scope("0");
            conv_t::<D>(&xspx, 2 * c, 9, true)
        };
        let spx = spx.record_custom(CustomData::new(ChannelSoftmaxOp::<D>::new(256)), &[], None);
        disp.record_custom(
            CustomData::new(ContextUpsampleOp::<D>::new(4.0, 32)),
            &[&spx],
            None,
        )
    }
}
