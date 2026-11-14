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

//! 2-D bilinear resize, equivalent to
//! `F.interpolate(x, size=(OH, OW), mode='bilinear')`.
//!
//! FoundationStereo uses this at four sites with **both** `align_corners` settings
//! (`core/extractor.py:358` with `True`, `core/foundation_stereo.py:290` with `False`,
//! among others). Rather than branch on a flag, the caller passes the source-coordinate
//! mapping directly as `src = dst * scale + offset`:
//!
//! | `align_corners` | `scale` | `offset` |
//! |---|---|---|
//! | `true`  | `(IN - 1) / (OUT - 1)` | `0` |
//! | `false` | `IN / OUT`             | `0.5 * scale - 0.5` |
//!
//! Both were checked against PyTorch to within fp32 rounding. Keeping the mapping in
//! the caller also sidesteps a runtime branch, which teenyc 0.3.0 does not compile.
//!
//! Sampling is **border-clamped**, matching `F.interpolate`: source coordinates are
//! clamped to `[0, IN-1]` and the right/bottom taps to the last row/column, so edge
//! pixels replicate rather than reading zeros. This differs from the zero padding used
//! by [`super::bilinear_sampler_1d`], which implements `grid_sample`.
//!
//! **Layout**: input `[B, C, IH, IW]`, output `[B, C, OH, OW]`.
//!
//! Grid: `B * C * OH * ceil(OW / BLOCK_P)` — one CTA per (instance, output row,
//! column tile), lanes covering adjacent output columns. Source coordinates and
//! weights are fractional and stay in float; the floored row and column are cast to
//! i32 on their own (small, so exact) and clamping and addressing are done in i32, so
//! no flat offset ever passes through float.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// Bilinear resize with a caller-supplied coordinate mapping.
///
/// Input:  `x_ptr`   — `[B, C, IH, IW]`
/// Output: `out_ptr` — `[B, C, OH, OW]`
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn interpolate_bilinear_forward<T: Triton, D: Float, const BLOCK_P: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    C: i32,
    IH: i32,
    IW: i32,
    OH: i32,
    OW: i32,
    scale_y: f32,
    offset_y: f32,
    scale_x: f32,
    offset_x: f32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let sp_out = OH * OW;
    let tiles = T::cdiv(OW, BLOCK_P);
    let pid = T::program_id(Axis::X);
    let tile = pid % tiles;
    let r1 = pid / tiles;
    let oy = r1 % OH;
    let r2 = r1 / OH;
    let c = r2 % C;
    let b = r2 / C;

    let ox = T::arange(0, BLOCK_P) + tile * BLOCK_P;
    let pmask = ox.lt(OW);
    let zeros = T::zeros::<D>(&[BLOCK_P]);
    let ones = T::full(&[BLOCK_P], D::from_f64(1.0));
    let izero = T::zeros::<i32>(&[BLOCK_P]);

    let ihmax = T::full(&[BLOCK_P], D::from_f64((IH - 1) as f64));
    let iwmax = T::full(&[BLOCK_P], D::from_f64((IW - 1) as f64));

    // Map into the source image and clamp to the border.
    let oyf = T::cast::<i32, D>(izero + oy, None, false);
    let oxf = T::cast::<i32, D>(ox, None, false);
    let sy = oyf * T::full(&[BLOCK_P], D::from_f64(scale_y as f64))
        + T::full(&[BLOCK_P], D::from_f64(offset_y as f64));
    let sx = oxf * T::full(&[BLOCK_P], D::from_f64(scale_x as f64))
        + T::full(&[BLOCK_P], D::from_f64(offset_x as f64));
    let sy = T::minimum(T::maximum(sy, zeros), ihmax);
    let sx = T::minimum(T::maximum(sx, zeros), iwmax);

    // Interpolation weights stay in float; the corner taps move to i32. Coordinates
    // are below IH/IW, so the cast is exact, and addresses never pass through float.
    let y0f = T::floor(sy);
    let x0f = T::floor(sx);
    let wy = sy - y0f;
    let wx = sx - x0f;
    let y0 = T::cast::<D, i32>(y0f, None, false);
    let x0 = T::cast::<D, i32>(x0f, None, false);
    let y1 = T::minimum(y0 + 1, izero + (IH - 1));
    let x1 = T::minimum(x0 + 1, izero + (IW - 1));

    let in_plane = (b * C + c) * IH * IW;
    let i00 = y0 * IW + x0;
    let i01 = y0 * IW + x1;
    let i10 = y1 * IW + x0;
    let i11 = y1 * IW + x1;

    let v00 = T::load(
        x_ptr.add_offsets(i00 + in_plane),
        Some(pmask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let v01 = T::load(
        x_ptr.add_offsets(i01 + in_plane),
        Some(pmask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let v10 = T::load(
        x_ptr.add_offsets(i10 + in_plane),
        Some(pmask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let v11 = T::load(
        x_ptr.add_offsets(i11 + in_plane),
        Some(pmask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );

    let top = v00 * (ones - wx) + v01 * wx;
    let bot = v10 * (ones - wx) + v11 * wx;
    let v = top * (ones - wy) + bot * wy;

    T::store(
        out_ptr.add_offsets(ox + ((b * C + c) * sp_out + oy * OW)),
        v,
        Some(pmask),
        &[],
        None,
        None,
    );
}
