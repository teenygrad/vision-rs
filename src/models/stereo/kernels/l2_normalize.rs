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

    // max(||x||, eps) — a clamp, not an addition, exactly as F.normalize specifies.
    let denom = T::maximum(T::sqrt(T::sum(x * x, Some(0), true)), eps);

    T::store(
        out_ptr.add_offsets(offs),
        x / denom,
        Some(cmask),
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

/// Graph-level representation of channel-axis L2 normalisation.
///
/// Shape-preserving. The input may have any rank >= 2: axis 1 is the channel axis
/// and every axis after it is flattened into `S`.
pub struct L2NormalizeOp<D: FloatBytes + Send + Sync + 'static> {
    /// Kernel launch block size along the channel axis; must be >= `C`.
    pub block_c: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> L2NormalizeOp<D> {
    /// Creates the op with the launch block size along the channel axis.
    pub fn new(block_c: i32) -> Self {
        Self {
            block_c,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for L2NormalizeOp<D> {
    fn name(&self) -> &str {
        "stereo.l2_normalize"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = L2NormalizeForward::<D>::new(self.block_c);
        Some((
            "l2_normalize_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(L2NormalizeRuntimeOp::<D>::new(self.block_c)),
        ))
    }
}

/// Runtime dispatch for channel-axis L2 normalisation.
pub struct L2NormalizeRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    block_c: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> L2NormalizeRuntimeOp<D> {
    /// Creates the runtime op with the launch block size along the channel axis.
    pub fn new(block_c: i32) -> Self {
        Self {
            block_c,
            _phantom: PhantomData,
        }
    }
}

/// Channel count and flattened trailing extent for a `[B, C, ...]` shape.
fn c_and_s(shape: &[usize]) -> (i32, i32) {
    let c = shape[1] as i32;
    let s: usize = shape[2..].iter().product::<usize>().max(1);
    (c, s as i32)
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for L2NormalizeRuntimeOp<D> {
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
        let (c, s) = c_and_s(output_shape);
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(c);
        visitor.visit_i32(s);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let _ = self.block_c;
        let (_, s) = c_and_s(output_shape);
        [(output_shape[0] * s as usize) as u32, 1, 1]
    }
}
