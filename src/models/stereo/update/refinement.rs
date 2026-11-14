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

//! One step of FoundationStereo's refinement loop (`FoundationStereo.forward`,
//! `core/foundation_stereo.py`):
//!
//! ```python
//! for itr in range(iters):
//!     geo_feat = geo_fn(disp, coords)
//!     net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, att)
//!     disp = disp + delta_disp
//! ```
//!
//! The step is a graph of its own, run once per iteration with its `net` and `disp`
//! outputs fed back as the next step's inputs, so the update block's weights are
//! loaded once however many iterations run. The geometry pyramid
//! ([`crate::models::stereo::geometry::geo_pyramid`]) is built once per image pair,
//! outside the loop.

use teeny_core::{
    dtype::{Float, FloatBytes},
    graph::SymTensor,
};

use super::ops::add;
use super::update_block::{UpdateBlockDims, UpdateInputs, basic_selective_multi_update_block};
use crate::models::stereo::geometry::{GeoPyramid, geo_lookup};

/// One refinement step's inputs.
pub struct StepInputs<'a> {
    /// GRU hidden states at 1/4, 1/8 and 1/16 resolution.
    pub net: [&'a SymTensor; 3],
    /// Context features per level (fixed across iterations).
    pub inp: [&'a SymTensor; 3],
    /// Spatial attention per level (fixed across iterations).
    pub att: [&'a SymTensor; 3],
    /// The geometry pyramid (fixed across iterations).
    pub pyramid: &'a GeoPyramid,
    /// Current disparity `[B, 1, H, W]` at 1/4 resolution.
    pub disp: &'a SymTensor,
}

/// One refinement step's outputs.
pub struct StepOutputs {
    /// Updated hidden states.
    pub net: [SymTensor; 3],
    /// `mask_feat_4`, for the final upsampling.
    pub mask: SymTensor,
    /// `disp + delta_disp`.
    pub disp: SymTensor,
}

/// One refinement iteration: geometry lookup at the current disparity, the update
/// block, and the disparity update. Weights are the update block's, recorded under the
/// caller's scope (`update_block` in the full model).
pub fn refinement_step<D: Float + FloatBytes + Send + Sync + 'static>(
    dims: UpdateBlockDims,
    radius: usize,
) -> impl Fn(&StepInputs) -> StepOutputs {
    let update = basic_selective_multi_update_block::<D>(dims);
    move |s: &StepInputs| {
        let corr = geo_lookup::<D>(s.pyramid, s.disp, radius);
        let out = update(&UpdateInputs {
            net: s.net,
            inp: s.inp,
            corr: &corr,
            disp: s.disp,
            att: s.att,
        });
        StepOutputs {
            disp: add(s.disp, &out.delta_disp),
            net: out.net,
            mask: out.mask,
        }
    }
}
