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

//! 3x3 neighbourhood unfold, the gather half of convex upsampling.
//!
//! Equivalent to `F.unfold(x, kernel_size=3, dilation=1, padding=1)` reshaped back to
//! spatial form, which is how FoundationStereo expands a low-resolution disparity map
//! before weighting it (`core/submodule.py:459`):
//!
//! ```python
//! disp_unfold = F.unfold(disp_low.reshape(b, c, h, w), 3, 1, 1).reshape(b, -1, h, w)
//! ```
//!
//! Stride is 1 and padding 1, so the output keeps the input's spatial size and each
//! output position carries its own 3x3 neighbourhood. Out-of-bounds taps read zero,
//! matching `F.unfold`'s zero padding.
//!
//! **Layout**: input `[B, C, H, W]`, output `[B, C*9, H, W]`. The tap index is
//! row-major within the window, so output channel `c*9 + ky*3 + kx` holds
//! `x[b, c, y + ky - 1, x + kx - 1]`. That matches `F.unfold`'s channel ordering.
//!
//! Grid: `B * C * H * ceil(W / BLOCK_W)` — one CTA per row tile. The nine taps are a
//! compile-time unrolled loop, so there is no runtime control flow and no reduction.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// Gather each position's 3x3 neighbourhood into 9 output channels.
///
/// Input:  `x_ptr`   — `[B, C, H, W]`
/// Output: `out_ptr` — `[B, C*9, H, W]`
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn unfold3x3_forward<T: Triton, D: Float, const BLOCK_W: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    C: i32,
    H: i32,
    W: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Tensor<D>: Comparison<T::Tensor<D>, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    // Grid: one CTA per (batch, channel, tap, spatial tile). Lanes cover flattened
    // spatial positions, so every quantity is lane-varying and the body has the same
    // shape as the 1-D sampler, whose constructs are known to compile.
    //
    // All index maths is done in float and cast once at the end. Clamping in integers
    // needs integer maximum/minimum, which ICEs in teenyc 0.3.0 ("MLIR cleanup passes
    // failed"); the float overloads are fine. Values stay well under 2^24, so the
    // float round trip is exact.
    let sp = H * W;
    let tiles = T::cdiv(sp, BLOCK_W);
    let pid = T::program_id(Axis::X);
    let tile = pid % tiles;
    let r1 = pid / tiles;
    let tap = r1 % 9;
    let r2 = r1 / 9;
    let c = r2 % C;
    let b = r2 / C;

    let ky = tap / 3;
    let kx = tap % 3;

    let p = T::arange(0, BLOCK_W) + tile * BLOCK_W;
    let pmask = p.lt(sp);
    let zeros = T::zeros::<D>(&[BLOCK_W]);

    let wf = T::full(&[BLOCK_W], D::from_f64(W as f64));
    let wmax = T::full(&[BLOCK_W], D::from_f64((W - 1) as f64));
    let hmax = T::full(&[BLOCK_W], D::from_f64((H - 1) as f64));

    // Decompose the flat position into (row, column) in float.
    let pf = T::cast::<i32, D>(p, None, false);
    let yf = T::floor(pf / wf);
    let xf = pf - yf * wf;

    // Offset to this tap.
    let yyf = yf + T::full(&[BLOCK_W], D::from_f64((ky - 1) as f64));
    let xxf = xf + T::full(&[BLOCK_W], D::from_f64((kx - 1) as f64));

    let y_lo = yyf.ge(zeros);
    let y_hi = yyf.le(hmax);
    let x_lo = xxf.ge(zeros);
    let x_hi = xxf.le(wmax);

    let yc = T::minimum(T::maximum(yyf, zeros), hmax);
    let xc = T::minimum(T::maximum(xxf, zeros), wmax);
    let src = T::cast::<D, i32>(yc * wf + xc, None, false);

    let in_plane = (b * C + c) * sp;
    let v = T::load(
        x_ptr.add_offsets(src + in_plane),
        Some(pmask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );

    // F.unfold pads with zeros.
    let v = T::where_(y_lo, v, zeros);
    let v = T::where_(y_hi, v, zeros);
    let v = T::where_(x_lo, v, zeros);
    let v = T::where_(x_hi, v, zeros);

    let out_plane = ((b * C + c) * 9 + tap) * sp;
    T::store(
        out_ptr.add_offsets(p + out_plane),
        v,
        Some(pmask),
        &[],
        None,
        None,
    );
}
