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

//! FoundationStereo's convex upsampling of the 1/4-resolution disparity, fused:
//! `context_upsample(disp_low * 4, up_weights)` (`core/submodule.py:452`):
//!
//! ```python
//! disp_unfold = F.unfold(disp_low, 3, 1, 1).reshape(b, -1, h, w)              # 3x3 neighbours
//! disp_unfold = F.interpolate(disp_unfold, (h*4, w*4), mode='nearest')       # to full res
//! disp = (disp_unfold * up_weights).sum(1)
//! ```
//!
//! i.e. each full-resolution pixel `(y, x)` is a weighted sum of the 3x3 neighbourhood
//! of its 1/4-resolution parent `(y / 4, x / 4)`, zero-padded, with the 9 weights at
//! that pixel:
//!
//! ```text
//! out[b, 0, y, x] = sum_k  4 * disp[b, 0, y/4 + k/3 - 1, x/4 + k%3 - 1] * w[b, k, y, x]
//! ```
//!
//! The factor 4 (the disparity is in 1/4-resolution pixels) is applied per tap before
//! the product, as `disp * 4.` precedes the unfold in PyTorch; it is a power of two, so
//! exact either way. Fusing avoids materialising the `[B, 9, 4h, 4w]` unfolded map.
//! The scale is a parameter so the same kernel serves other upsampling factors.
//!
//! **Layout**: `disp` `[B, 1, h, w]`, `w` `[B, 9, F*h, F*w]`, output
//! `[B, 1, F*h, F*w]`. Grid: `B * F*h * ceil(F*w / BLOCK_W)`, lanes on columns of one
//! output row; the 9 taps are one flat loop.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// Convex upsampling of `disp` by the integer factor `F` with 3x3 weights.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn context_upsample_forward<T: Triton, D: Float, const BLOCK_W: i32>(
    disp_ptr: InPtr<T::Pointer<D>>,
    w_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    H: i32,
    W: i32,
    F: i32,
    value_scale: f32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let OH = H * F;
    let OW = W * F;
    let tiles = (OW + BLOCK_W - 1) / BLOCK_W;
    let pid = T::program_id(Axis::X);
    let tile = pid % tiles;
    let r1 = pid / tiles;
    let oy = r1 % OH;
    let b = r1 / OH;

    let ox = T::arange(0, BLOCK_W) + tile * BLOCK_W;
    let m = ox.lt(OW);
    let zeros = T::zeros::<D>(&[BLOCK_W]);
    let izero = T::zeros::<i32>(&[BLOCK_W]);
    let scale = T::full(&[BLOCK_W], D::from_f64(value_scale as f64));

    // Parent pixel: nearest-neighbour upsampling maps (oy, ox) to (oy / F, ox / F).
    // The column is per lane; floor in float is exact at these sizes and avoids
    // tensor integer division.
    let yl = oy / F;
    let inv_f = T::full(&[BLOCK_W], D::from_f64(1.0 / F as f64));
    let xl = T::cast::<D, i32>(
        T::floor(T::cast::<i32, D>(ox, None, false) * inv_f),
        None,
        false,
    );

    let mut acc = T::zeros::<D>(&[BLOCK_W]);
    for k in 0..9 {
        let yy = izero + (yl + k / 3 - 1);
        let xx = xl + (k % 3 - 1);
        let yc = T::maximum(T::minimum(yy, izero + (H - 1)), izero);
        let xc = T::maximum(T::minimum(xx, izero + (W - 1)), izero);
        let v = T::load(
            disp_ptr.add_offsets(yc * W + xc + b * H * W),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        // F.unfold pads with zeros.
        let v = T::where_(yy.ge(0), v, zeros);
        let v = T::where_(yy.lt(H), v, zeros);
        let v = T::where_(xx.ge(0), v, zeros);
        let v = T::where_(xx.lt(W), v, zeros);
        let wk = T::load(
            w_ptr.add_offsets(ox + ((b * 9 + k) * OH + oy) * OW),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        acc = acc + (v * scale) * wk;
    }

    T::store(
        out_ptr.add_offsets(ox + (b * OH + oy) * OW),
        acc,
        Some(m),
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

/// Graph-level `context_upsample(disp * value_scale, weights)`.
///
/// Inputs, in order: `disp` `[B, 1, h, w]`, `weights` `[B, 9, F*h, F*w]`. Output
/// `[B, 1, F*h, F*w]`; `F` is taken from the shapes.
pub struct ContextUpsampleOp<D: FloatBytes + Send + Sync + 'static> {
    /// Factor applied to `disp` before upsampling (4 in FoundationStereo).
    pub value_scale: f32,
    /// Kernel launch block size along output columns.
    pub block_w: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> ContextUpsampleOp<D> {
    /// Creates the op.
    pub fn new(value_scale: f32, block_w: i32) -> Self {
        Self {
            value_scale,
            block_w,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for ContextUpsampleOp<D> {
    fn name(&self) -> &str {
        "stereo.context_upsample"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        let w = input_shapes[1];
        vec![w[0], Some(1), w[2], w[3]]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = ContextUpsampleForward::<D>::new(self.block_w);
        Some((
            "context_upsample_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(ContextUpsampleRuntimeOp::<D> {
                value_scale: self.value_scale,
                block_w: self.block_w,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`ContextUpsampleOp`].
pub struct ContextUpsampleRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    value_scale: f32,
    block_w: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for ContextUpsampleRuntimeOp<D> {
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
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        let (d, w) = (inputs[0].1, inputs[1].1);
        assert_eq!(w[1], 9, "context_upsample: weights must have 9 channels");
        let f = w[2] / d[2];
        assert!(
            f * d[2] == w[2] && f * d[3] == w[3],
            "context_upsample: weights {w:?} not an integer multiple of disp {d:?}"
        );
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(d[2] as i32);
        visitor.visit_i32(d[3] as i32);
        visitor.visit_i32(f as i32);
        visitor.visit_f32(self.value_scale);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let tiles = output_shape[3].div_ceil(self.block_w as usize);
        [(output_shape[0] * output_shape[2] * tiles) as u32, 1, 1]
    }
}
