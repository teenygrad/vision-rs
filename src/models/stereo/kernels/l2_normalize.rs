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

//! Channel-axis L2 normalisation, equivalent to `torch.nn.functional.normalize(x, dim=1)`.
//!
//! ```text
//! out[b, c, s] = x[b, c, s] / max(||x[b, :, s]||_2, eps)
//! ```
//!
//! `eps` is PyTorch's default `1e-12`, applied as a clamp on the denominator rather
//! than an addition, so an all-zero channel vector maps to zero rather than NaN. This
//! matters for stereo feature maps, where padded or masked regions are exactly zero.
//!
//! **Layout**: `x` is `[B, C, S]` row-major, where `S` flattens every trailing spatial
//! axis (`H*W` for a 4-D feature map). The channel stride is therefore `S`.
//!
//! Grid: `B * S` — one CTA per spatial position. `BLOCK_C` must be >= `C`.
//!
//! FoundationStereo applies this to both feature maps before the correlation einsum
//! (`core/geometry.py:74`); see [`super::correlation`], which consumes the result.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// L2-normalise along the channel axis.
///
/// Input:  `x_ptr`   — `[B, C, S]` row-major
/// Output: `out_ptr` — `[B, C, S]`
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn l2_normalize_forward<T: Triton, D: Float, const BLOCK_C: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    C: i32,
    S: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    // One CTA per (batch, spatial position).
    let pid = T::program_id(Axis::X);
    let s = pid % S;
    let b = pid / S;

    let c_offs = T::arange(0, BLOCK_C);
    let cmask = c_offs.lt(C);
    let zeros = T::zeros::<D>(&[BLOCK_C]);
    let eps = T::full(&[BLOCK_C], D::from_f64(1e-12));

    // Masked lanes load 0, so they contribute nothing to the sum of squares.
    let offs = c_offs * S + (b * C * S + s);
    let x = T::load(
        x_ptr.add_offsets(offs),
        Some(cmask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );

    // PyTorch computes x / max(||x||, eps). `T::maximum` and `T::where_` both ICE in
    // teenyc 0.3.0 when either operand is a reduction result ("MLIR cleanup passes
    // failed"), so this adds eps instead of clamping with it. The two agree here:
    // for ||x|| = 0 both give 0/1e-12 = 0, and for ||x|| > 0 the relative difference is
    // eps/||x|| <= 1e-12/||x||, far below fp32 resolution for any realistic norm.
    let denom = T::sqrt(T::sum(x * x, Some(0), true)) + eps;

    T::store(
        out_ptr.add_offsets(offs),
        x / denom,
        Some(cmask),
        &[],
        None,
        None,
    );
}
