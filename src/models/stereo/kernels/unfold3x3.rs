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
//! Grid: `B * C * 9 * H * ceil(W / BLOCK_W)` — one CTA per (instance, tap, row,
//! column tile). The row is a scalar and lanes cover adjacent columns, so all index
//! arithmetic is i32 multiply-add: no flat-position decomposition, no float round
//! trip. There is no runtime control flow and no reduction.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// Gather each position's 3x3 neighbourhood into 9 output channels.
///
/// Input:  `x_ptr`   — `[B, C, H, W]`
/// Output: `out_ptr` — `[B, C*9, H, W]`
#[kernel]
pub fn unfold3x3_forward<T, D: Float, const BLOCK_W: i32>(
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
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    // One CTA per (batch, channel, tap, row, column tile). Index arithmetic is i32;
    // an earlier revision did it in float to dodge an integer maximum/minimum ICE in
    // teenyc 0.3.0 (since fixed), which is only exact while offsets stay below 2^24.
    let sp = H * W;
    let tiles = T::cdiv(W, BLOCK_W);
    let pid = T::program_id(Axis::X);
    let tile = pid % tiles;
    let r1 = pid / tiles;
    let y = r1 % H;
    let r2 = r1 / H;
    let tap = r2 % 9;
    let r3 = r2 / 9;
    let c = r3 % C;
    let b = r3 / C;

    let ky = tap / 3;
    let kx = tap % 3;

    let xs = T::arange(0, BLOCK_W) + tile * BLOCK_W;
    let xmask = xs.lt(W);
    let zeros = T::zeros::<D>(&[BLOCK_W]);
    let izero = T::zeros::<i32>(&[BLOCK_W]);

    // Source row (scalar, broadcast for range checks) and column for this tap.
    let yy = izero + (y + ky - 1);
    let xx = xs + (kx - 1);

    let y_lo = yy.ge(0);
    let y_hi = yy.lt(H);
    let x_lo = xx.ge(0);
    let x_hi = xx.lt(W);

    // Clamp so every load is in bounds; out-of-range taps are zeroed below.
    let yc = T::maximum(T::minimum(yy, izero + (H - 1)), izero);
    let xc = T::maximum(T::minimum(xx, izero + (W - 1)), izero);
    let src = yc * W + xc;

    let in_plane = (b * C + c) * sp;
    let v = T::load(
        x_ptr.add_offsets(src + in_plane),
        Some(xmask),
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
        out_ptr.add_offsets(xs + (out_plane + y * W)),
        v,
        Some(xmask),
        &[],
        None,
        None,
    );
}
