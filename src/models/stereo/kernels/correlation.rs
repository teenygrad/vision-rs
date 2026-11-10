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

//! All-pairs cosine correlation volume for stereo matching.
//!
//! The einsum FoundationStereo builds its initial correlation volume with
//! (`core/geometry.py:74`):
//!
//! ```python
//! corr = torch.einsum('aijk,aijh->ajkh',
//!                     F.normalize(fmap1.float(), dim=1),
//!                     F.normalize(fmap2.float(), dim=1))
//! ```
//!
//! With `a=batch`, `i=channel`, `j=row`, `k=left column`, `h=right column`, that is
//!
//! ```text
//! out[a, j, k, h] = sum_i  n1[a, i, j, k] * n2[a, i, j, h]
//! ```
//!
//! where `n1`, `n2` are `fmap1`, `fmap2` L2-normalised along the channel axis. Since
//! both operands are unit vectors this is a cosine similarity, so the result is in
//! [-1, 1] regardless of feature scale.
//!
//! **Inputs must already be L2-normalised along the channel axis** — use
//! [`super::l2_normalize`] first. This mirrors FoundationStereo, which normalises whole
//! feature maps before the einsum, and keeps the normalised maps reusable.
//!
//! **Layout**: `fmap1` is `[B, C, H, W1]` and `fmap2` is `[B, C, H, W2]`, both
//! row-major. Output is `[B, H, W1, W2]`, which is exactly the layout the 1-D bilinear
//! sampler then consumes as `[N = B*H*W1, 1, W2]`.
//!
//! Grid: `B * H * W1` — one CTA per left pixel, sweeping every right column, so the
//! left column is loaded once and reused across all `W2` dot products.
//!
//! `BLOCK_C` must be >= `C`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// All-pairs correlation of two channel-normalised feature maps.
///
/// Inputs:
///   `fmap1_ptr` — `[B, C, H, W1]` row-major, channel-normalised
///   `fmap2_ptr` — `[B, C, H, W2]` row-major, channel-normalised
///
/// Output:
///   `out_ptr`   — `[B, H, W1, W2]`
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn correlation_forward<T: Triton, D: Float, const BLOCK_C: i32>(
    fmap1_ptr: InPtr<T::Pointer<D>>,
    fmap2_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    C: i32,
    H: i32,
    W1: i32,
    W2: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let k = pid % W1;
    let rest = pid / W1;
    let y = rest % H;
    let b = rest / H;

    let c_offs = T::arange(0, BLOCK_C);
    let cmask = c_offs.lt(C);
    let zeros = T::zeros::<D>(&[BLOCK_C]);
    let s1 = H * W1;
    let s2 = H * W2;

    let f1 = T::load(
        fmap1_ptr.add_offsets(c_offs * s1 + (b * C * s1 + y * W1 + k)),
        Some(cmask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let out_base = ((b * H + y) * W1 + k) * W2;

    for h in 0..W2 {
        let f2 = T::load(
            fmap2_ptr.add_offsets(c_offs * s2 + (b * C * s2 + y * W2 + h)),
            Some(cmask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        // Reduce to a 1-element tensor and store that. Storing a broadcast
        // (keepdim) reduction under a lane-0 mask instead triggers an ICE in
        // teenyc 0.3.0: "MLIR cleanup passes failed: CodegenFailed".
        let dot = T::sum(f1 * f2, Some(0), false) / T::full(&[1], D::from_f64(1.0));
        T::store(
            out_ptr.add_offsets(T::arange(0, 1) + (out_base + h)),
            dot,
            None,
            &[],
            None,
            None,
        );
    }
}
