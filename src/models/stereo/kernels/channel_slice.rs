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

//! A contiguous range of channels, `x[:, start:start + len]` for `[B, C, ...]`:
//! `torch.chunk` / `torch.split` along `dim=1`, one output per call.
//!
//! teeny-kernels' `Op::Slice` and `Op::Split` are TODO in the graph lowering.
//! EdgeNeXt's `SplitTransposeBlock` chunks its input along channels.
//!
//! **Layout**: `[B, C, S]` in, `[B, len, S]` out, `S` flattening the trailing axes.
//! Grid: `B * len * ceil(S / BLOCK)`, one row of `S` per program.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// `out[b, c, s] = x[b, start + c, s]`.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn channel_slice_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    C: i32,
    START: i32,
    LEN: i32,
    S: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let tiles = (S + BLOCK - 1) / BLOCK;
    let pid = T::program_id(Axis::X);
    let tile = pid % tiles;
    let row = pid / tiles;
    let c = row % LEN;
    let b = row / LEN;

    let s = T::arange(0, BLOCK) + tile * BLOCK;
    let m = s.lt(S);
    let x = T::load(
        x_ptr.add_offsets(s + (b * C + START + c) * S),
        Some(m),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(
        out_ptr.add_offsets(s + row * S),
        x,
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

/// Graph-level channel slice. Rank >= 2.
pub struct ChannelSliceOp<D: FloatBytes + Send + Sync + 'static> {
    /// First channel.
    pub start: usize,
    /// Number of channels.
    pub len: usize,
    /// Launch block size along the trailing axes.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> ChannelSliceOp<D> {
    /// Creates the op.
    pub fn new(start: usize, len: usize, block: i32) -> Self {
        Self {
            start,
            len,
            block,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for ChannelSliceOp<D> {
    fn name(&self) -> &str {
        "stereo.channel_slice"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        let mut s = input_shapes[0].clone();
        s[1] = Some(self.len);
        s
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = ChannelSliceForward::<D>::new(self.block);
        Some((
            "channel_slice_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(ChannelSliceRuntimeOp::<D> {
                start: self.start,
                len: self.len,
                block: self.block,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`ChannelSliceOp`].
pub struct ChannelSliceRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    start: usize,
    len: usize,
    block: i32,
    _phantom: PhantomData<D>,
}

fn trailing(shape: &[usize]) -> usize {
    shape[2..].iter().product::<usize>().max(1)
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for ChannelSliceRuntimeOp<D> {
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
        let x = inputs[0].1;
        assert!(self.start + self.len <= x[1], "channel_slice: range");
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(x[1] as i32);
        visitor.visit_i32(self.start as i32);
        visitor.visit_i32(self.len as i32);
        visitor.visit_i32(trailing(output_shape) as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let tiles = trailing(output_shape).div_ceil(self.block as usize);
        [(output_shape[0] * self.len * tiles) as u32, 1, 1]
    }
}
