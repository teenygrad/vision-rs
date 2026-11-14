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

//! FoundationStereo's `BasicSelectiveMultiUpdateBlock` (`core/update.py`): one
//! refinement iteration across the three GRU levels.

use teeny_core::{
    dtype::{Float, FloatBytes},
    graph::SymTensor,
    name_scope::name_scope,
};

use super::disp_head::disp_head;
use super::gru::selective_conv_gru;
use super::motion_encoder::{MotionEncoderDims, basic_motion_encoder};
use super::ops::{cat, conv, interp, pool2x, relu, scale};

/// Channel widths of [`basic_selective_multi_update_block`].
#[derive(Clone, Copy, Debug)]
pub struct UpdateBlockDims {
    /// GRU hidden width at every level (`hidden_dims[0]`).
    pub hidden: usize,
    /// The motion encoder's widths. Its output plus `inp[0]` must total `2 * hidden`.
    pub motion: MotionEncoderDims,
    /// Middle width of the mask head.
    pub mask_mid: usize,
    /// Output width of the mask head (`mask_feat_4`).
    pub mask_out: usize,
}

impl UpdateBlockDims {
    /// The shipped checkpoints: hidden 128, mask head 128 -> 64 -> 32.
    pub fn foundation_stereo() -> Self {
        Self {
            hidden: 128,
            motion: MotionEncoderDims::foundation_stereo(),
            mask_mid: 64,
            mask_out: 32,
        }
    }

    /// Channels of the 1/4-resolution context `inp[0]`.
    pub fn inp0(&self) -> usize {
        assert!(
            self.motion.out < 2 * self.hidden,
            "motion encoder output must leave room for inp[0]"
        );
        2 * self.hidden - self.motion.out
    }
}

/// One refinement step's inputs. Levels are 1/4, 1/8 and 1/16 resolution.
pub struct UpdateInputs<'a> {
    /// GRU hidden states `net[0..3]`, each `[B, hidden, H_i, W_i]`.
    pub net: [&'a SymTensor; 3],
    /// Context features `inp[0..3]`: `inp[0]` has [`UpdateBlockDims::inp0`] channels,
    /// `inp[1]` and `inp[2]` have `hidden`.
    pub inp: [&'a SymTensor; 3],
    /// Geometry-encoding lookup `[B, cor_planes, H_0, W_0]`.
    pub corr: &'a SymTensor,
    /// Current disparity `[B, 1, H_0, W_0]`.
    pub disp: &'a SymTensor,
    /// Spatial attention per level, `[B, 1, H_i, W_i]`.
    pub att: [&'a SymTensor; 3],
}

/// One step's outputs.
pub struct UpdateOutputs {
    /// Updated hidden states, same shapes as the inputs.
    pub net: [SymTensor; 3],
    /// `mask_feat_4`, `[B, mask_out, H_0, W_0]`, already scaled by 0.25.
    pub mask: SymTensor,
    /// Disparity update `[B, 1, H_0, W_0]`.
    pub delta_disp: SymTensor,
}

/// `BasicSelectiveMultiUpdateBlock` with `n_gru_layers = 3`, the shipped
/// configuration:
///
/// ```python
/// net[2] = gru16(att[2], net[2], inp[2], pool2x(net[1]))
/// net[1] = gru08(att[1], net[1], inp[1], pool2x(net[0]), interp(net[2], net[1]))
/// motion_features = cat([inp[0], encoder(disp, corr)])
/// net[0] = gru04(att[0], net[0], motion_features, interp(net[1], net[0]))
/// delta_disp = disp_head(net[0])
/// mask = .25 * mask(net[0])           # Conv-ReLU-Conv-ReLU
/// ```
///
/// Each level reads the level below's *already updated* state, as in PyTorch. Weights
/// are recorded under `encoder`, `gru16`, `gru08`, `gru04`, `disp_head` and `mask`,
/// matching the module's `state_dict` (with the caller's scope prepended).
pub fn basic_selective_multi_update_block<D: Float + FloatBytes + Send + Sync + 'static>(
    dims: UpdateBlockDims,
) -> impl Fn(&UpdateInputs) -> UpdateOutputs {
    let h = dims.hidden;
    let encoder = basic_motion_encoder::<D>(dims.motion);
    let gru16 = selective_conv_gru::<D>(h, 2 * h);
    let gru08 = selective_conv_gru::<D>(h, 3 * h);
    let gru04 = selective_conv_gru::<D>(h, 3 * h);
    let head = disp_head::<D>(h, 1);
    let mask0 = conv::<D>("0", h, dims.mask_mid, 3);
    let mask2 = conv::<D>("2", dims.mask_mid, dims.mask_out, 3);
    let inp0 = dims.inp0();
    move |i: &UpdateInputs| {
        assert_eq!(
            i.inp[0].shape[1],
            Some(inp0),
            "inp[0] must have 2 * hidden - motion.out channels"
        );
        let net2 = {
            let _g = name_scope("gru16");
            gru16(i.att[2], i.net[2], &[i.inp[2], &pool2x::<D>(i.net[1])])
        };
        let net1 = {
            let pooled = pool2x::<D>(i.net[0]);
            let up = interp::<D>(&net2, i.net[1]);
            let _g = name_scope("gru08");
            gru08(i.att[1], i.net[1], &[i.inp[1], &pooled, &up])
        };
        let motion = {
            let _g = name_scope("encoder");
            encoder(i.disp, i.corr)
        };
        let motion = cat(&[i.inp[0], &motion]);
        let net0 = {
            let up = interp::<D>(&net1, i.net[0]);
            let _g = name_scope("gru04");
            gru04(i.att[0], i.net[0], &[&motion, &up])
        };
        let delta_disp = {
            let _g = name_scope("disp_head");
            head(&net0)
        };
        let mask = {
            let _g = name_scope("mask");
            let m = relu::<D>(mask2(&relu::<D>(mask0(&net0))));
            scale::<D>(&m, 0.25)
        };
        UpdateOutputs {
            net: [net0, net1, net2],
            mask,
            delta_disp,
        }
    }
}
