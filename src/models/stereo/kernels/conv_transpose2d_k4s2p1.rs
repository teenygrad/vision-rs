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

//! `ConvTranspose2d(kernel_size=4, stride=2, padding=1)` — an exact 2x upsample.
//!
//! This is the only transposed convolution shape FoundationStereo uses
//! (`core/foundation_stereo.py:166`):
//!
//! ```python
//! nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1)
//! ```
//!
//! Output size is `(IH-1)*2 - 2 + 4 = 2*IH`, so it doubles both axes exactly.
//!
//! **Written as a gather, not a scatter.** The scatter form that the name suggests
//! needs atomics; the gather form does not. For an output row `oy`, the contributing
//! kernel rows satisfy `oy = iy*2 - 1 + ky`, so `ky` must match the parity of `oy + 1`.
//! With `ky in 0..4` that leaves exactly two choices, `ky = p + 2j` for `j in {0,1}`
//! and `p = (oy + 1) % 2`, each giving `iy = (oy + 1 - ky) / 2`. The same holds per
//! column, so every output draws from exactly a 2x2 neighbourhood of the input,
//! reduced over `Cin`. Taps whose `iy`/`ix` fall outside the input contribute zero.
//!
//! **Layout**: input `[B, Cin, IH, IW]`, weight `[Cin, Cout, 4, 4]` (PyTorch's
//! transposed-conv layout, in-channels first), bias `[Cout]`, output
//! `[B, Cout, 2*IH, 2*IW]`.
//!
//! Grid: `B * Cout * OH * OW` — one CTA per output element, lanes over `Cin`.
//! `BLOCK_CI` must be >= `Cin`. The four taps accumulate elementwise into a register
//! tile and are reduced once at the end, so the body contains a single reduction.
//!
//! As with the other stereo kernels, index arithmetic is done in float and cast once:
//! integer `maximum`/`minimum` ICE in teenyc 0.3.0 while the float overloads are fine.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// Transposed convolution with `kernel_size=4, stride=2, padding=1`.
///
/// Inputs:  `x_ptr` `[B, Cin, IH, IW]`, `w_ptr` `[Cin, Cout, 4, 4]`, `b_ptr` `[Cout]`
/// Output:  `out_ptr` `[B, Cout, 2*IH, 2*IW]`
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn conv_transpose2d_k4s2p1_forward<T: Triton, D: Float, const BLOCK_CI: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    w_ptr: InPtr<T::Pointer<D>>,
    b_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    CIN: i32,
    COUT: i32,
    IH: i32,
    IW: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Tensor<D>: Comparison<T::Tensor<D>, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let oh = IH * 2;
    let ow = IW * 2;

    // One CTA per (batch, out channel, out row, out column).
    let pid = T::program_id(Axis::X);
    let ox = pid % ow;
    let r1 = pid / ow;
    let oy = r1 % oh;
    let r2 = r1 / oh;
    let co = r2 % COUT;
    let b = r2 / COUT;

    let ci = T::arange(0, BLOCK_CI);
    let cimask = ci.lt(CIN);
    let zeros = T::zeros::<D>(&[BLOCK_CI]);
    let ihmax = T::full(&[BLOCK_CI], D::from_f64((IH - 1) as f64));
    let iwmax = T::full(&[BLOCK_CI], D::from_f64((IW - 1) as f64));
    let iwf = T::full(&[BLOCK_CI], D::from_f64(IW as f64));
    let plane = T::full(&[BLOCK_CI], D::from_f64((IH * IW) as f64));
    let wstride = T::full(&[BLOCK_CI], D::from_f64((COUT * 16) as f64));
    let cif = T::cast::<i32, D>(ci, None, false);

    // Only kernel taps matching the parity of oy+1 / ox+1 contribute.
    let py = (oy + 1) % 2;
    let px = (ox + 1) % 2;

    // Four taps, written straight-line and summed into one expression so the body
    // holds a single reduction.
    //
    // The equivalent nested-loop form with a `let mut acc` did not compile here
    // ("MLIR cleanup passes failed" in teenyc 0.3.0), but a reduced version of that
    // pattern compiles in isolation, so the real trigger is some interaction and is
    // not yet minimised. Treat this shape as the one known to work rather than as
    // evidence that accumulators are broken.
    let xbase = b * CIN * IH * IW;

    // ---- tap (j=0, i=0) ----
    let ky00 = py;
    let kx00 = px;
    let iy00 = (oy + 1 - ky00) / 2;
    let ix00 = (ox + 1 - kx00) / 2;
    let iyf00 = T::full(&[BLOCK_CI], D::from_f64(iy00 as f64));
    let ixf00 = T::full(&[BLOCK_CI], D::from_f64(ix00 as f64));
    let xo00 = T::cast::<D, i32>(
        cif * plane
            + (T::minimum(T::maximum(iyf00, zeros), ihmax) * iwf
                + T::minimum(T::maximum(ixf00, zeros), iwmax)),
        None,
        false,
    );
    let wo00 = T::cast::<D, i32>(
        cif * wstride + T::full(&[BLOCK_CI], D::from_f64((co * 16 + ky00 * 4 + kx00) as f64)),
        None,
        false,
    );
    let xv00 = T::load(
        x_ptr.add_offsets(xo00 + xbase),
        Some(cimask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let wv00 = T::load(
        w_ptr.add_offsets(wo00),
        Some(cimask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    // Taps whose source falls outside the input contribute nothing.
    let xv00 = T::where_(iyf00.ge(zeros), xv00, zeros);
    let xv00 = T::where_(iyf00.le(ihmax), xv00, zeros);
    let xv00 = T::where_(ixf00.ge(zeros), xv00, zeros);
    let xv00 = T::where_(ixf00.le(iwmax), xv00, zeros);
    let t00 = xv00 * wv00;

    // ---- tap (j=0, i=1) ----
    let ky01 = py;
    let kx01 = px + 2;
    let iy01 = (oy + 1 - ky01) / 2;
    let ix01 = (ox + 1 - kx01) / 2;
    let iyf01 = T::full(&[BLOCK_CI], D::from_f64(iy01 as f64));
    let ixf01 = T::full(&[BLOCK_CI], D::from_f64(ix01 as f64));
    let xo01 = T::cast::<D, i32>(
        cif * plane
            + (T::minimum(T::maximum(iyf01, zeros), ihmax) * iwf
                + T::minimum(T::maximum(ixf01, zeros), iwmax)),
        None,
        false,
    );
    let wo01 = T::cast::<D, i32>(
        cif * wstride + T::full(&[BLOCK_CI], D::from_f64((co * 16 + ky01 * 4 + kx01) as f64)),
        None,
        false,
    );
    let xv01 = T::load(
        x_ptr.add_offsets(xo01 + xbase),
        Some(cimask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let wv01 = T::load(
        w_ptr.add_offsets(wo01),
        Some(cimask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    // Taps whose source falls outside the input contribute nothing.
    let xv01 = T::where_(iyf01.ge(zeros), xv01, zeros);
    let xv01 = T::where_(iyf01.le(ihmax), xv01, zeros);
    let xv01 = T::where_(ixf01.ge(zeros), xv01, zeros);
    let xv01 = T::where_(ixf01.le(iwmax), xv01, zeros);
    let t01 = xv01 * wv01;

    // ---- tap (j=1, i=0) ----
    let ky10 = py + 2;
    let kx10 = px;
    let iy10 = (oy + 1 - ky10) / 2;
    let ix10 = (ox + 1 - kx10) / 2;
    let iyf10 = T::full(&[BLOCK_CI], D::from_f64(iy10 as f64));
    let ixf10 = T::full(&[BLOCK_CI], D::from_f64(ix10 as f64));
    let xo10 = T::cast::<D, i32>(
        cif * plane
            + (T::minimum(T::maximum(iyf10, zeros), ihmax) * iwf
                + T::minimum(T::maximum(ixf10, zeros), iwmax)),
        None,
        false,
    );
    let wo10 = T::cast::<D, i32>(
        cif * wstride + T::full(&[BLOCK_CI], D::from_f64((co * 16 + ky10 * 4 + kx10) as f64)),
        None,
        false,
    );
    let xv10 = T::load(
        x_ptr.add_offsets(xo10 + xbase),
        Some(cimask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let wv10 = T::load(
        w_ptr.add_offsets(wo10),
        Some(cimask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    // Taps whose source falls outside the input contribute nothing.
    let xv10 = T::where_(iyf10.ge(zeros), xv10, zeros);
    let xv10 = T::where_(iyf10.le(ihmax), xv10, zeros);
    let xv10 = T::where_(ixf10.ge(zeros), xv10, zeros);
    let xv10 = T::where_(ixf10.le(iwmax), xv10, zeros);
    let t10 = xv10 * wv10;

    // ---- tap (j=1, i=1) ----
    let ky11 = py + 2;
    let kx11 = px + 2;
    let iy11 = (oy + 1 - ky11) / 2;
    let ix11 = (ox + 1 - kx11) / 2;
    let iyf11 = T::full(&[BLOCK_CI], D::from_f64(iy11 as f64));
    let ixf11 = T::full(&[BLOCK_CI], D::from_f64(ix11 as f64));
    let xo11 = T::cast::<D, i32>(
        cif * plane
            + (T::minimum(T::maximum(iyf11, zeros), ihmax) * iwf
                + T::minimum(T::maximum(ixf11, zeros), iwmax)),
        None,
        false,
    );
    let wo11 = T::cast::<D, i32>(
        cif * wstride + T::full(&[BLOCK_CI], D::from_f64((co * 16 + ky11 * 4 + kx11) as f64)),
        None,
        false,
    );
    let xv11 = T::load(
        x_ptr.add_offsets(xo11 + xbase),
        Some(cimask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let wv11 = T::load(
        w_ptr.add_offsets(wo11),
        Some(cimask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    // Taps whose source falls outside the input contribute nothing.
    let xv11 = T::where_(iyf11.ge(zeros), xv11, zeros);
    let xv11 = T::where_(iyf11.le(ihmax), xv11, zeros);
    let xv11 = T::where_(ixf11.ge(zeros), xv11, zeros);
    let xv11 = T::where_(ixf11.le(iwmax), xv11, zeros);
    let t11 = xv11 * wv11;

    // Single reduction over Cin, then bias. Storing a broadcast reduction under a mask
    // ICEs, so reduce to one element and store that.
    let prod = t00 + t01 + t10 + t11;
    let dot = T::sum(prod, Some(0), false) / T::full(&[1], D::from_f64(1.0));
    let bias = T::load(
        b_ptr.add_offsets(T::arange(0, 1) + co),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );

    T::store(
        out_ptr.add_offsets(T::arange(0, 1) + pid),
        dot + bias,
        None,
        &[],
        None,
        None,
    );
}
