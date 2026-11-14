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

//! Square average pooling with zero padding, equivalent to
//! `torch.nn.functional.avg_pool2d(x, K, stride=S, padding=P)` with its defaults
//! (`ceil_mode=False`, `count_include_pad=True`):
//!
//! ```text
//! OH = (IH + 2P - K) / S + 1          (floor), likewise OW
//! out[b, c, oy, ox] = sum_{ky, kx < K} x[b, c, oy*S - P + ky, ox*S - P + kx] / K^2
//! ```
//!
//! Taps that fall in the padding read zero but still count towards the `K^2` divisor,
//! which is what `count_include_pad=True` means, so border outputs are pulled towards
//! zero. teeny-kernels' `Avgpool2dForward` has no padding parameter, which is why this
//! kernel exists.
//!
//! FoundationStereo uses it in the GRU refinement (`core/update.py`): `pool2x` is
//! `K=3, S=2, P=1` and `pool4x` is `K=5, S=4, P=1`. Its other average pools
//! (`[1,2]` stride `[1,2]` in `core/geometry.py`) are unpadded, and the global
//! adaptive pools lower to teeny-kernels' `GlobalAvgPool`/`GlobalMaxPool`.
//!
//! **Layout**: input `[B, C, IH, IW]`, output `[B, C, OH, OW]`, both row-major.
//!
//! Grid: `B * C * OH * ceil(OW / BLOCK_W)` — each CTA covers `BLOCK_W` adjacent output
//! columns of one output row, so the input row index is a scalar and every lane's
//! column is `ox*S - P + kx`: index arithmetic is i32 multiply-add only.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// Zero-padded square average pooling (`count_include_pad=True`).
///
/// Input:  `x_ptr`   — `[B, C, IH, IW]`
/// Output: `out_ptr` — `[B, C, OH, OW]`
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn avg_pool2d_padded_forward<T: Triton, D: Float, const BLOCK_W: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    IH: i32,
    IW: i32,
    OH: i32,
    OW: i32,
    K: i32,
    S: i32,
    P: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    // One CTA per (instance, output row, column tile).
    let tiles = (OW + BLOCK_W - 1) / BLOCK_W;
    let pid = T::program_id(Axis::X);
    let tile = pid % tiles;
    let r1 = pid / tiles;
    let oy = r1 % OH;
    let inst = r1 / OH;

    let ox = T::arange(0, BLOCK_W) + tile * BLOCK_W;
    let omask = ox.lt(OW);
    let zeros = T::zeros::<D>(&[BLOCK_W]);
    let izero = T::zeros::<i32>(&[BLOCK_W]);

    let in_base = inst * IH * IW;
    let ix0 = ox * S + (0 - P);

    // One flat loop over the K*K taps. A nested `for ky { for kx { .. } }` with the
    // accumulator carried through both levels compiles under teenyc but returns
    // all-zero output (checked on CUDA), so the taps are enumerated as one loop.
    let mut acc = T::zeros::<D>(&[BLOCK_W]);
    for t in 0..K * K {
        let ky = t / K;
        let kx = t % K;
        // Scalar input row, broadcast so it can be range-checked per lane.
        let iy = izero + (oy * S - P + ky);
        let iy_c = T::maximum(T::minimum(iy, izero + (IH - 1)), izero);
        let ix = ix0 + kx;
        let ix_c = T::maximum(T::minimum(ix, izero + (IW - 1)), izero);
        // Clamped indices keep every load in bounds; out-of-range taps are then
        // zeroed, which is the padding.
        let v = T::load(
            x_ptr.add_offsets(iy_c * IW + ix_c + in_base),
            Some(omask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let v = T::where_(iy.ge(0), v, zeros);
        let v = T::where_(iy.lt(IH), v, zeros);
        let v = T::where_(ix.ge(0), v, zeros);
        let v = T::where_(ix.lt(IW), v, zeros);
        acc = acc + v;
    }

    // count_include_pad=True: always divide by the full window.
    let area = T::cast::<i32, D>(izero + K * K, None, false);
    T::store(
        out_ptr.add_offsets(ox + (inst * OH + oy) * OW),
        acc / area,
        Some(omask),
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

/// Output extent of a pooled axis (`ceil_mode=False`).
pub fn pooled_extent(input: usize, k: usize, s: usize, p: usize) -> usize {
    (input + 2 * p - k) / s + 1
}

/// Graph-level representation of zero-padded square average pooling.
///
/// Input `[B, C, IH, IW]`, output `[B, C, OH, OW]` with
/// `OH = (IH + 2P - K) / S + 1`, likewise `OW`.
pub struct AvgPool2dPaddedOp<D: FloatBytes + Send + Sync + 'static> {
    /// Window side.
    pub k: usize,
    /// Stride.
    pub s: usize,
    /// Zero padding on each side.
    pub p: usize,
    /// Kernel launch block size along the output column axis.
    pub block_w: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> AvgPool2dPaddedOp<D> {
    /// Creates the op for window `k`, stride `s`, padding `p`.
    pub fn new(k: usize, s: usize, p: usize, block_w: i32) -> Self {
        assert!(2 * p <= k, "avg_pool2d requires padding <= kernel_size / 2");
        Self {
            k,
            s,
            p,
            block_w,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for AvgPool2dPaddedOp<D> {
    fn name(&self) -> &str {
        "stereo.avg_pool2d_padded"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        // [B, C, IH, IW] -> [B, C, OH, OW]; an unknown extent stays unknown.
        let x = input_shapes[0];
        let pool = |d: Option<usize>| d.map(|v| pooled_extent(v, self.k, self.s, self.p));
        vec![x[0], x[1], pool(x[2]), pool(x[3])]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = AvgPool2dPaddedForward::<D>::new(self.block_w);
        Some((
            "avg_pool2d_padded_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(AvgPool2dPaddedRuntimeOp::<D>::new(
                self.k,
                self.s,
                self.p,
                self.block_w,
            )),
        ))
    }
}

/// Runtime dispatch for zero-padded square average pooling.
pub struct AvgPool2dPaddedRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    k: usize,
    s: usize,
    p: usize,
    block_w: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> AvgPool2dPaddedRuntimeOp<D> {
    /// Creates the runtime op for window `k`, stride `s`, padding `p`.
    pub fn new(k: usize, s: usize, p: usize, block_w: i32) -> Self {
        Self {
            k,
            s,
            p,
            block_w,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for AvgPool2dPaddedRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
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
        let in_shape = inputs[0].1;
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(in_shape[2] as i32);
        visitor.visit_i32(in_shape[3] as i32);
        visitor.visit_i32(output_shape[2] as i32);
        visitor.visit_i32(output_shape[3] as i32);
        visitor.visit_i32(self.k as i32);
        visitor.visit_i32(self.s as i32);
        visitor.visit_i32(self.p as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let bw = self.block_w as usize;
        let tiles = output_shape[3].div_ceil(bw);
        [
            (output_shape[0] * output_shape[1] * output_shape[2] * tiles) as u32,
            1,
            1,
        ]
    }
}
