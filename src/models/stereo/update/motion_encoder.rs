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

//! FoundationStereo's `BasicMotionEncoder` (`core/update.py`).

use teeny_core::{
    dtype::{Float, FloatBytes},
    graph::SymTensor,
};

use super::ops::{cat, conv, relu};

/// Channel widths of [`basic_motion_encoder`].
///
/// [`MotionEncoderDims::foundation_stereo`] gives the model's; tests use smaller ones
/// with the same structure.
#[derive(Clone, Copy, Debug)]
pub struct MotionEncoderDims {
    /// Channels of the geometry-encoding lookup:
    /// `corr_levels * (2 * corr_radius + 1) * (ngroup + 1)`.
    pub cor_planes: usize,
    /// Width of the correlation branch (`convc1`/`convc2`).
    pub cor: usize,
    /// Width of the disparity branch (`convd1`/`convd2`).
    pub disp: usize,
    /// Output channels, including the disparity channel appended at the end.
    pub out: usize,
}

impl MotionEncoderDims {
    /// The shipped checkpoints: `corr_levels = 2`, `corr_radius = 4`, `ngroup = 8`.
    pub fn foundation_stereo() -> Self {
        Self {
            cor_planes: 2 * (2 * 4 + 1) * (8 + 1),
            cor: 256,
            disp: 64,
            out: 128,
        }
    }
}

/// `BasicMotionEncoder`:
///
/// ```python
/// cor = relu(convc2(relu(convc1(corr))))
/// disp_ = relu(convd2(relu(convd1(disp))))
/// out = relu(conv(cat([cor, disp_])))
/// return cat([out, disp])
/// ```
///
/// `disp` is `[B, 1, H, W]` and `corr` is `[B, cor_planes, H, W]`; the result has
/// `dims.out` channels, the last of which is `disp` itself. Weights are recorded as
/// `convc1.*`, `convc2.*`, `convd1.*`, `convd2.*`, `conv.*` under the caller's scope.
pub fn basic_motion_encoder<D: Float + FloatBytes + Send + Sync + 'static>(
    dims: MotionEncoderDims,
) -> impl Fn(&SymTensor, &SymTensor) -> SymTensor {
    let convc1 = conv::<D>("convc1", dims.cor_planes, dims.cor, 1);
    let convc2 = conv::<D>("convc2", dims.cor, dims.cor, 3);
    let convd1 = conv::<D>("convd1", 1, dims.disp, 7);
    let convd2 = conv::<D>("convd2", dims.disp, dims.disp, 3);
    let conv_out = conv::<D>("conv", dims.cor + dims.disp, dims.out - 1, 3);
    move |disp: &SymTensor, corr: &SymTensor| {
        let cor = relu::<D>(convc2(&relu::<D>(convc1(corr))));
        let disp_ = relu::<D>(convd2(&relu::<D>(convd1(disp))));
        let out = relu::<D>(conv_out(&cat(&[&cor, &disp_])));
        cat(&[&out, disp])
    }
}
