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

//! 1-D bilinear sampler — the correlation-volume lookup used by stereo matching.
//!
//! This is the stereo special case of `grid_sample`. FoundationStereo's
//! `bilinear_sampler` (core/utils/utils.py) asserts
//!
//! ```text
//! assert torch.unique(ygrid).numel() == 1 and H == 1  # This is a stereo problem
//! ```
//!
//! so the sampled signal is always height-1 with a constant y grid. A full 2-D
//! `GridSample` is therefore unnecessary: the operation reduces to a linear
//! interpolation between two adjacent columns.
//!
//! **Layout**: `img` is `[N, C, W]` row-major, where `N` folds every batch/pixel
//! index (`B*H*W` in the caller). `coords` is `[N, K]` holding sample positions in
//! **pixel** units along `W`. Output is `[N, C, K]`.
//!
//! Positions outside `[0, W-1]` contribute zero, matching `grid_sample`'s
//! `padding_mode="zeros"` with `align_corners=True`.
//!
//! Grid: `N * C` — one CTA per `(n, channel)` pair. `K` is small in practice
//! (`2*radius+1`, radius 4 → 9), so a single `BLOCK_K` tile covers it.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// 1-D bilinear gather along the last axis of `img`.
///
/// Inputs:
///   `img_ptr`    — `[N, C, W]` row-major
///   `coords_ptr` — `[N, K]` sample positions in pixel units
///
/// Output:
///   `out_ptr`    — `[N, C, K]`
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn bilinear_sampler_1d_forward<T: Triton, D: Float, const BLOCK_K: i32>(
    img_ptr: InPtr<T::Pointer<D>>,
    coords_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    C: i32,
    W: i32,
    K: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Tensor<D>: Comparison<T::Tensor<D>, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let n = pid / C;
    let c = pid % C;

    let k_offs = T::arange(0, BLOCK_K);
    let kmask = k_offs.lt(K);

    let fzero = T::zeros::<D>(&[BLOCK_K]);
    let one = T::full(&[BLOCK_K], D::from_f64(1.0));
    let wmax = T::full(&[BLOCK_K], D::from_f64((W - 1) as f64));

    // Sample positions, in pixel units along W.
    let x = T::load(
        coords_ptr.add_offsets(k_offs + (n * K)),
        Some(kmask),
        Some(fzero),
        &[],
        None,
        None,
        None,
        false,
    );

    // Left/right taps and their interpolation weights.
    let x0f = T::floor(x);
    let x1f = x0f + one;
    let w1 = x - x0f;
    let w0 = one - w1;

    // grid_sample(padding_mode="zeros"): a tap outside [0, W-1] contributes nothing.
    let in0 = x0f.ge(fzero);
    let in0b = x0f.le(wmax);
    let in1 = x1f.ge(fzero);
    let in1b = x1f.le(wmax);

    // Clamp in float so every load is in bounds, then zero the invalid taps below.
    let x0c = T::minimum(T::maximum(x0f, fzero), wmax);
    let x1c = T::minimum(T::maximum(x1f, fzero), wmax);
    let i0 = T::cast::<D, i32>(x0c, None, false);
    let i1 = T::cast::<D, i32>(x1c, None, false);

    let base = (n * C + c) * W;
    let v0 = T::load(
        img_ptr.add_offsets(i0 + base),
        Some(kmask),
        Some(fzero),
        &[],
        None,
        None,
        None,
        false,
    );
    let v1 = T::load(
        img_ptr.add_offsets(i1 + base),
        Some(kmask),
        Some(fzero),
        &[],
        None,
        None,
        None,
        false,
    );

    let v0 = T::where_(in0, v0, fzero);
    let v0 = T::where_(in0b, v0, fzero);
    let v1 = T::where_(in1, v1, fzero);
    let v1 = T::where_(in1b, v1, fzero);

    let acc = v0 * w0 + v1 * w1;

    T::store(
        out_ptr.add_offsets(k_offs + ((n * C + c) * K)),
        acc,
        Some(kmask),
        &[],
        None,
        None,
    );
}
