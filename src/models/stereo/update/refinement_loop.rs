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

//! The driver for FoundationStereo's refinement loop: the step graph's input layout,
//! a builder that records [`refinement_step`] over it, and the host loop that runs a
//! compiled step `iters` times, feeding each step's hidden states and disparity into
//! the next.
//!
//! The driver does not compile or launch anything itself: [`run_refinement`] takes the
//! compiled step as a closure, so it works with any runtime (`RuntimeModel`, a CUDA
//! graph, a board build) that runs a graph on host buffers.
//!
//! Step graph inputs, in creation (and therefore feed) order:
//!
//! | index | input | shape |
//! |---|---|---|
//! | 0..3 | `net0..net2` | `[B, hidden, H / 2^l, W / 2^l]` |
//! | 3..6 | `inp0..inp2` | as `net` |
//! | 6..9 | `att0..att2` | `[B, 1, H / 2^l, W / 2^l]` |
//! | then `levels` | `geo_l` | `[B, G, D / 2^l, H, W]` |
//! | then `levels` | `corr_l` | `[B, H, W, W2 / 2^l]` |
//! | last | `disp` | `[B, 1, H, W]` |
//!
//! Outputs, in order: `net0`, `net1`, `net2`, `mask`, `disp`.

use teeny_core::{
    dtype::{Float, FloatBytes},
    graph::SymTensor,
};

use super::refinement::{StepInputs, refinement_step};
use super::update_block::UpdateBlockDims;
use crate::models::stereo::geometry::GeoPyramid;

/// Step graph output names, in the order [`build_step`] returns them.
pub const STEP_OUTPUTS: [&str; 5] = ["net0", "net1", "net2", "mask", "disp"];

/// Sizes that fix the step graph's input shapes.
#[derive(Clone, Copy, Debug)]
pub struct StepShape {
    /// Batch size.
    pub batch: usize,
    /// Height and width at 1/4 resolution (the disparity's).
    pub h: usize,
    /// See `h`.
    pub w: usize,
    /// Cost-volume groups (`volume_dim`, 28 in FoundationStereo).
    pub groups: usize,
    /// Disparities in the level-0 cost volume.
    pub disps: usize,
    /// Right-image columns in the level-0 correlation (`W` for a rectified pair).
    pub w2: usize,
    /// Pyramid levels.
    pub levels: usize,
}

/// The step graph's inputs as `(name, shape)`, in feed order (see the module docs).
pub fn step_inputs(dims: &UpdateBlockDims, s: &StepShape) -> Vec<(String, Vec<usize>)> {
    let (b, hid) = (s.batch, dims.hidden);
    let lvl = |l: usize| (s.h >> l, s.w >> l);
    let mut v = Vec::new();
    for (name, ch) in [("net", hid), ("inp", hid), ("att", 1)] {
        for l in 0..3 {
            let (h, w) = lvl(l);
            v.push((format!("{name}{l}"), vec![b, ch, h, w]));
        }
    }
    for l in 0..s.levels {
        v.push((format!("geo{l}"), vec![b, s.groups, s.disps >> l, s.h, s.w]));
    }
    for l in 0..s.levels {
        v.push((format!("corr{l}"), vec![b, s.h, s.w, s.w2 >> l]));
    }
    v.push(("disp".into(), vec![b, 1, s.h, s.w]));
    v
}

/// Records one refinement step over `inputs` (created in [`step_inputs`] order, for a
/// pyramid of `levels`), returning the outputs named in [`STEP_OUTPUTS`].
pub fn build_step<D: Float + FloatBytes + Send + Sync + 'static>(
    dims: UpdateBlockDims,
    radius: usize,
    levels: usize,
    inputs: &[SymTensor],
) -> Vec<(&'static str, SymTensor)> {
    assert_eq!(inputs.len(), 10 + 2 * levels, "step graph inputs");
    let pyramid = GeoPyramid {
        geo: inputs[9..9 + levels].to_vec(),
        corr: inputs[9 + levels..9 + 2 * levels].to_vec(),
    };
    let out = refinement_step::<D>(dims, radius)(&StepInputs {
        net: [&inputs[0], &inputs[1], &inputs[2]],
        inp: [&inputs[3], &inputs[4], &inputs[5]],
        att: [&inputs[6], &inputs[7], &inputs[8]],
        pyramid: &pyramid,
        disp: &inputs[9 + 2 * levels],
    });
    let [n0, n1, n2] = out.net;
    STEP_OUTPUTS
        .into_iter()
        .zip([n0, n1, n2, out.mask, out.disp])
        .collect()
}

/// Inputs that stay fixed across iterations, as host buffers.
pub struct LoopFixed<'a> {
    /// Context features per level.
    pub inp: [&'a [f32]; 3],
    /// Spatial attention per level.
    pub att: [&'a [f32]; 3],
    /// Cost-volume pyramid levels.
    pub geo: &'a [&'a [f32]],
    /// Correlation pyramid levels.
    pub corr: &'a [&'a [f32]],
}

/// The loop's carried state.
#[derive(Clone, Debug, Default)]
pub struct LoopState {
    /// GRU hidden states per level.
    pub net: [Vec<f32>; 3],
    /// Current disparity at 1/4 resolution.
    pub disp: Vec<f32>,
    /// The last step's `mask_feat_4` (empty before the first step).
    pub mask: Vec<f32>,
}

/// Runs `iters` refinement steps from `state`. `run_step` runs the compiled step graph
/// on its inputs in [`step_inputs`] order and returns the [`STEP_OUTPUTS`];
/// `on_iter(it, &state)` sees the state after each iteration (for per-iteration
/// disparities, as FoundationStereo's training returns them).
pub fn run_refinement<E>(
    iters: usize,
    fixed: &LoopFixed,
    mut state: LoopState,
    mut run_step: impl FnMut(&[&[f32]]) -> Result<Vec<Vec<f32>>, E>,
    mut on_iter: impl FnMut(usize, &LoopState),
) -> Result<LoopState, E> {
    assert_eq!(fixed.geo.len(), fixed.corr.len(), "pyramid levels");
    for it in 0..iters {
        let out = {
            let mut feed: Vec<&[f32]> = state.net.iter().map(Vec::as_slice).collect();
            feed.extend(fixed.inp);
            feed.extend(fixed.att);
            feed.extend(fixed.geo);
            feed.extend(fixed.corr);
            feed.push(&state.disp);
            run_step(&feed)?
        };
        assert_eq!(out.len(), STEP_OUTPUTS.len(), "step graph outputs");
        let [n0, n1, n2, mask, disp]: [Vec<f32>; 5] = out.try_into().unwrap();
        state = LoopState {
            net: [n0, n1, n2],
            disp,
            mask,
        };
        on_iter(it, &state);
    }
    Ok(state)
}
