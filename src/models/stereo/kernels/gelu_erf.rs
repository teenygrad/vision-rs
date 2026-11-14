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

//! Exact GELU, equivalent to `torch.nn.GELU()` with its default
//! `approximate='none'`:
//!
//! ```text
//! y = 0.5 * x * (1 + erf(x / sqrt(2)))
//! ```
//!
//! teeny-kernels lowers `Op::Gelu` to the tanh approximation
//! (`0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`), which differs from the
//! exact form by up to ~5e-4 (teenygrad/teenygrad#157). FoundationStereo's
//! `EdgeNextConvEncoder` uses the default exact GELU, so the refinement head records
//! this op instead.
//!
//! Element-wise over a flat buffer. Grid: `ceil(n / BLOCK)`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// `y = 0.5 * x * (1 + erf(x / sqrt(2)))`.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn gelu_erf_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    N: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let offs = T::arange(0, BLOCK) + pid * BLOCK;
    let m = offs.lt(N);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let x = T::load(
        x_ptr.add_offsets(offs),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );

    let half = T::full(&[BLOCK], D::from_f64(0.5));
    let one = T::full(&[BLOCK], D::from_f64(1.0));
    // 1/sqrt(2) as a literal: teenyc's generated kernel core has no `core::f64::consts`.
    #[allow(clippy::approx_constant)]
    let inv_sqrt2 = T::full(&[BLOCK], D::from_f64(0.7071067811865476));
    let y = half * x * (one + T::erf(x * inv_sqrt2));

    T::store(out_ptr.add_offsets(offs), y, Some(m), &[], None, None);
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

/// Graph-level exact GELU. Shape-preserving, any rank.
pub struct GeluErfOp<D: FloatBytes + Send + Sync + 'static> {
    /// Kernel launch block size.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> GeluErfOp<D> {
    /// Creates the op with the given launch block size.
    pub fn new(block: i32) -> Self {
        Self {
            block,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for GeluErfOp<D> {
    fn name(&self) -> &str {
        "stereo.gelu_erf"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = GeluErfForward::<D>::new(self.block);
        Some((
            "gelu_erf_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(GeluErfRuntimeOp::<D>::new(self.block)),
        ))
    }
}

/// Runtime dispatch for exact GELU.
pub struct GeluErfRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> GeluErfRuntimeOp<D> {
    /// Creates the runtime op with the given launch block size.
    pub fn new(block: i32) -> Self {
        Self {
            block,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for GeluErfRuntimeOp<D> {
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
        visitor.visit_i32(output_shape.iter().product::<usize>() as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        [n.div_ceil(self.block as usize) as u32, 1, 1]
    }
}
