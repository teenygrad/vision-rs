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

//! Softmax over the channel axis, `F.softmax(x, dim=1)` for `[B, C, ...]`.
//!
//! teeny-kernels lowers `Op::Softmax` as a last-axis softmax and drops its `dim`
//! (teenygrad/teenygrad#152), so a channel softmax cannot go through it.
//! FoundationStereo applies one to its 9 upsampling weights per pixel
//! (`upsample_disp`, `core/foundation_stereo.py`).
//!
//! Max-subtracted, as PyTorch computes it. **Layout**: `[B, C, S]`, `S` flattening the
//! trailing axes. Grid: `B * ceil(S / BLOCK)`, lanes on positions; the channel axis is
//! walked three times (max, sum of exponentials, normalised write) in consecutive flat
//! loops.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// `out[b, c, s] = exp(x[b, c, s] - max_c) / sum_c exp(x[b, c, s] - max_c)`.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn channel_softmax_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    C: i32,
    S: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let tiles = (S + BLOCK - 1) / BLOCK;
    let pid = T::program_id(Axis::X);
    let tile = pid % tiles;
    let b = pid / tiles;

    let s = T::arange(0, BLOCK) + tile * BLOCK;
    let m = s.lt(S);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let base = s + b * C * S;

    // teenyc's RISC-V backend panics ("codegen_goto: target bbN not in basic_blocks")
    // when one loop follows another with no real op between them. Each loop's result is
    // therefore passed through an identity op (`+ 0`, exact) before the next loop.
    let mut mx = T::full(&[BLOCK], D::from_f64(-3.0e38));
    for c in 0..C {
        let x = T::load(
            x_ptr.add_offsets(base + c * S),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        mx = T::maximum(mx, x);
    }
    let mx = mx + zeros;
    let mut sum = T::zeros::<D>(&[BLOCK]);
    for c in 0..C {
        let x = T::load(
            x_ptr.add_offsets(base + c * S),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        sum = sum + T::exp(x - mx);
    }
    let sum = sum + zeros;
    for c in 0..C {
        let x = T::load(
            x_ptr.add_offsets(base + c * S),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        T::store(
            out_ptr.add_offsets(base + c * S),
            T::exp(x - mx) / sum,
            Some(m),
            &[],
            None,
            None,
        );
    }
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

/// Graph-level softmax over axis 1. Shape-preserving, rank >= 2.
pub struct ChannelSoftmaxOp<D: FloatBytes + Send + Sync + 'static> {
    /// Kernel launch block size along the flattened trailing axes.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> ChannelSoftmaxOp<D> {
    /// Creates the op.
    pub fn new(block: i32) -> Self {
        Self {
            block,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for ChannelSoftmaxOp<D> {
    fn name(&self) -> &str {
        "stereo.channel_softmax"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = ChannelSoftmaxForward::<D>::new(self.block);
        Some((
            "channel_softmax_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(ChannelSoftmaxRuntimeOp::<D> {
                block: self.block,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`ChannelSoftmaxOp`].
pub struct ChannelSoftmaxRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    block: i32,
    _phantom: PhantomData<D>,
}

fn trailing(shape: &[usize]) -> usize {
    shape[2..].iter().product::<usize>().max(1)
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for ChannelSoftmaxRuntimeOp<D> {
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
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(output_shape[1] as i32);
        visitor.visit_i32(trailing(output_shape) as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let tiles = trailing(output_shape).div_ceil(self.block as usize);
        [(output_shape[0] * tiles) as u32, 1, 1]
    }
}
