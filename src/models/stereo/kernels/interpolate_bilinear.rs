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

// ---------------------------------------------------------------------------
// Graph-level op
// ---------------------------------------------------------------------------

use core::marker::PhantomData;
use std::any::Any;
use std::sync::Arc;
use teeny_core::device::program::ArgVisitor;
use teeny_core::dtype::FloatBytes;
use teeny_core::graph::{CustomOp, Shape};
use teeny_core::model::{RawPtr, RuntimeOp};

/// Source-coordinate mapping `src = dst * scale + offset` for one axis, as
/// `F.interpolate(mode='bilinear')` defines it.
pub fn source_mapping(input: usize, output: usize, align_corners: bool) -> (f32, f32) {
    if align_corners {
        let scale = if output > 1 {
            (input as f32 - 1.0) / (output as f32 - 1.0)
        } else {
            0.0
        };
        (scale, 0.0)
    } else {
        let scale = input as f32 / output as f32;
        (scale, 0.5 * scale - 0.5)
    }
}

/// Graph-level `F.interpolate(x, dest.shape[2:], mode='bilinear', align_corners=…)`.
///
/// Inputs, in order: `x` `[B, C, IH, IW]`, then `dest`, which is read only for its
/// spatial size `[.., .., OH, OW]`. This is FoundationStereo's `interp(x, dest)`
/// (`core/update.py`). Output `[B, C, OH, OW]`.
pub struct InterpolateBilinearOp<D: FloatBytes + Send + Sync + 'static> {
    /// `align_corners` as passed to `F.interpolate`.
    pub align_corners: bool,
    /// Kernel launch block size along the output column axis.
    pub block_p: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> InterpolateBilinearOp<D> {
    /// Creates the op.
    pub fn new(align_corners: bool, block_p: i32) -> Self {
        Self {
            align_corners,
            block_p,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for InterpolateBilinearOp<D> {
    fn name(&self) -> &str {
        "stereo.interpolate_bilinear"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        let (x, dest) = (input_shapes[0], input_shapes[1]);
        vec![x[0], x[1], dest[2], dest[3]]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = InterpolateBilinearForward::<D>::new(self.block_p);
        Some((
            "interpolate_bilinear_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(InterpolateBilinearRuntimeOp::<D>::new(
                self.align_corners,
                self.block_p,
            )),
        ))
    }
}

/// Runtime dispatch for [`InterpolateBilinearOp`].
pub struct InterpolateBilinearRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    align_corners: bool,
    block_p: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> InterpolateBilinearRuntimeOp<D> {
    /// Creates the runtime op.
    pub fn new(align_corners: bool, block_p: i32) -> Self {
        Self {
            align_corners,
            block_p,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for InterpolateBilinearRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        2
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        Vec::new()
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        _params: &[RawPtr],
        output: RawPtr,
        output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        let x = inputs[0].1;
        let (ih, iw, oh, ow) = (x[2], x[3], output_shape[2], output_shape[3]);
        let (sy, oy) = source_mapping(ih, oh, self.align_corners);
        let (sx, ox) = source_mapping(iw, ow, self.align_corners);
        // `dest` (inputs[1]) is only consulted for its shape.
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(output_shape[1] as i32);
        visitor.visit_i32(ih as i32);
        visitor.visit_i32(iw as i32);
        visitor.visit_i32(oh as i32);
        visitor.visit_i32(ow as i32);
        visitor.visit_f32(sy);
        visitor.visit_f32(oy);
        visitor.visit_f32(sx);
        visitor.visit_f32(ox);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let tiles = output_shape[3].div_ceil(self.block_p as usize);
        [
            (output_shape[0] * output_shape[1] * output_shape[2] * tiles) as u32,
            1,
            1,
        ]
    }
}
