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

//! Average adjacent pairs along one axis:
//!
//! ```text
//! out[p, j, q] = (x[p, 2j, q] + x[p, 2j + 1, q]) / 2,   j < L / 2  (floor)
//! ```
//!
//! This is `F.avg_pool2d(x, [1, 2], stride=[1, 2])` on the last axis, which
//! FoundationStereo uses to build the second level of its geometry pyramid
//! (`core/geometry.py`). PyTorch applies it after permuting the cost volume to
//! `[B*H*W, C, 1, D]`; here it runs on the native layouts instead: the disparity axis
//! of the cost volume `[B, C, D, H, W]`, and the last axis of the correlation
//! `[B, H, W, W2]`. A trailing odd element is dropped, as `avg_pool2d` does.
//!
//! **Layout**: `x` viewed as `[P, L, Q]` around the pooled axis.
//!
//! Grid: `P * (L / 2) * ceil(Q / BLOCK)`. For `Q = 1` each CTA does one element, which
//! is correct but slow; it runs once per image pair.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// Average adjacent pairs along the middle axis of `[P, L, Q]`.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn pool_pairs_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    L: i32,
    Q: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let lh = L / 2;
    let tiles = (Q + BLOCK - 1) / BLOCK;
    let pid = T::program_id(Axis::X);
    let tile = pid % tiles;
    let r1 = pid / tiles;
    let j = r1 % lh;
    let p = r1 / lh;

    let q = T::arange(0, BLOCK) + tile * BLOCK;
    let m = q.lt(Q);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let half = T::full(&[BLOCK], D::from_f64(0.5));

    let src = q + (p * L + 2 * j) * Q;
    let a = T::load(
        x_ptr.add_offsets(src),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let b = T::load(
        x_ptr.add_offsets(src + Q),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );

    T::store(
        out_ptr.add_offsets(q + (p * lh + j) * Q),
        (a + b) * half,
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

/// Graph-level pair averaging along `axis`; that axis halves (floor), others keep.
pub struct PoolPairsOp<D: FloatBytes + Send + Sync + 'static> {
    /// The pooled axis.
    pub axis: usize,
    /// Kernel launch block size along the flattened trailing axes.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> PoolPairsOp<D> {
    /// Creates the op pooling along `axis`.
    pub fn new(axis: usize, block: i32) -> Self {
        Self {
            axis,
            block,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for PoolPairsOp<D> {
    fn name(&self) -> &str {
        "stereo.pool_pairs"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        let mut s = input_shapes[0].clone();
        s[self.axis] = s[self.axis].map(|l| l / 2);
        s
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = PoolPairsForward::<D>::new(self.block);
        Some((
            "pool_pairs_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(PoolPairsRuntimeOp::<D>::new(self.axis, self.block)),
        ))
    }
}

/// Runtime dispatch for [`PoolPairsOp`].
pub struct PoolPairsRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    axis: usize,
    block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> PoolPairsRuntimeOp<D> {
    /// Creates the runtime op pooling along `axis`.
    pub fn new(axis: usize, block: i32) -> Self {
        Self {
            axis,
            block,
            _phantom: PhantomData,
        }
    }

    /// `(P, L, Q)` for an input shape.
    fn plq(&self, input: &[usize]) -> (usize, usize, usize) {
        let p = input[..self.axis].iter().product::<usize>();
        let q = input[self.axis + 1..].iter().product::<usize>();
        (p, input[self.axis], q)
    }
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for PoolPairsRuntimeOp<D> {
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
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        let (_, l, q) = self.plq(inputs[0].1);
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(l as i32);
        visitor.visit_i32(q as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        // Output shape has L/2 on the axis; recover P and Q from it.
        let p = output_shape[..self.axis].iter().product::<usize>();
        let lh = output_shape[self.axis];
        let q = output_shape[self.axis + 1..].iter().product::<usize>();
        [(p * lh * q.div_ceil(self.block as usize)) as u32, 1, 1]
    }
}
