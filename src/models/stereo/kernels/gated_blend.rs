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

//! Gated blend of two tensors, with the gate optionally broadcast over channels:
//!
//! ```text
//! out[b, c, s] = a[b, c, s] * g[b, c', s] + b_[b, c, s] * (1 - g[b, c', s])
//! c' = c if the gate has C channels, 0 if it has one
//! ```
//!
//! FoundationStereo's refinement head blends in exactly this form twice
//! (`core/update.py`):
//!
//! - the GRU state update `h = (1 - z) * h + z * q`, i.e. `gated_blend(q, h, z)` with a
//!   full-channel gate;
//! - `SelectiveConvGRU`'s mix of its two GRUs,
//!   `small * att + large * (1 - att)`, i.e. `gated_blend(small, large, att)`, where
//!   `att` is a one-channel spatial attention map `[B, 1, H, W]`.
//!
//! teeny-kernels' element-wise `Mul`/`Sub` read both operands at the same flat offset
//! with no broadcasting, so the attention case cannot be written with them. The two
//! products are evaluated in PyTorch's order to keep rounding comparable.
//!
//! **Layout**: `a`, `b_` and `out` are `[B, C, S]`; `g` is `[B, GC, S]` with `GC` equal
//! to `C` or 1. `S` flattens the trailing axes.
//!
//! Grid: `B * C * ceil(S / BLOCK)`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// `a * g + b * (1 - g)`, with `g` either full-channel or broadcast over channels.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn gated_blend_forward<T: Triton, D: Float, const BLOCK: i32>(
    a_ptr: InPtr<T::Pointer<D>>,
    b_ptr: InPtr<T::Pointer<D>>,
    g_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    C: i32,
    GC: i32,
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
    let r1 = pid / tiles;
    let c = r1 % C;
    let b = r1 / C;
    // GC == C selects the matching gate channel, GC == 1 always channel 0.
    let gc = c % GC;

    let s = T::arange(0, BLOCK) + tile * BLOCK;
    let m = s.lt(S);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let ones = T::full(&[BLOCK], D::from_f64(1.0));

    let off = s + (b * C + c) * S;
    let a = T::load(
        a_ptr.add_offsets(off),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let bv = T::load(
        b_ptr.add_offsets(off),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let g = T::load(
        g_ptr.add_offsets(s + (b * GC + gc) * S),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );

    T::store(
        out_ptr.add_offsets(off),
        a * g + bv * (ones - g),
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

/// Graph-level `a * g + b * (1 - g)`.
///
/// Inputs, in order: `a` `[B, C, ...]`, `b` `[B, C, ...]`, `g` `[B, C or 1, ...]`.
/// Output has `a`'s shape.
pub struct GatedBlendOp<D: FloatBytes + Send + Sync + 'static> {
    /// Kernel launch block size along the flattened trailing axis.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> GatedBlendOp<D> {
    /// Creates the op with the launch block size along the trailing axis.
    pub fn new(block: i32) -> Self {
        Self {
            block,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for GatedBlendOp<D> {
    fn name(&self) -> &str {
        "stereo.gated_blend"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = GatedBlendForward::<D>::new(self.block);
        Some((
            "gated_blend_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(GatedBlendRuntimeOp::<D>::new(self.block)),
        ))
    }
}

/// Runtime dispatch for the gated blend.
pub struct GatedBlendRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> GatedBlendRuntimeOp<D> {
    /// Creates the runtime op with the launch block size along the trailing axis.
    pub fn new(block: i32) -> Self {
        Self {
            block,
            _phantom: PhantomData,
        }
    }
}

/// Trailing extent of a `[B, C, ...]` shape.
fn trailing(shape: &[usize]) -> usize {
    shape[2..].iter().product::<usize>().max(1)
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for GatedBlendRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        3
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
        let gate_c = inputs[2].1[1];
        assert!(
            gate_c == output_shape[1] || gate_c == 1,
            "gated_blend: gate has {gate_c} channels, expected {} or 1",
            output_shape[1]
        );
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(inputs[2].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(output_shape[1] as i32);
        visitor.visit_i32(gate_c as i32);
        visitor.visit_i32(trailing(output_shape) as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let tiles = trailing(output_shape).div_ceil(self.block as usize);
        [(output_shape[0] * output_shape[1] * tiles) as u32, 1, 1]
    }
}
