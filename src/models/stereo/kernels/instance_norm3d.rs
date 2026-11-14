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

//! Instance normalisation over every trailing axis, equivalent to
//! `torch.nn.InstanceNorm3d(C)` with its defaults (`affine=False`,
//! `track_running_stats=False`, `eps=1e-5`):
//!
//! ```text
//! mean[b, c] = sum_s x[b, c, s] / N
//! var[b, c]  = sum_s (x[b, c, s] - mean[b, c])^2 / N      (biased, as PyTorch)
//! out[b, c, s] = (x[b, c, s] - mean[b, c]) / sqrt(var[b, c] + eps)
//! ```
//!
//! With no affine parameters and no running statistics the op has no weights, and at
//! inference it behaves exactly as in training: statistics always come from the input.
//!
//! **Layout**: `x` is `[B, C, N]` row-major, where `N` flattens every trailing axis
//! (`D*H*W` for a cost volume). Because the reduction covers all of them, the same
//! kernel is also exact for `InstanceNorm1d`/`InstanceNorm2d`.
//!
//! In FoundationStereo's shipped configurations this runs as `InstanceNorm2d`
//! (18 modules, same defaults). Its `InstanceNorm3d` sites (`core/submodule.py:64`,
//! `:328`) are only built for 3-D layers with `norm='instance'`, which neither
//! released checkpoint uses: their 3-D layers are normalised with `BatchNorm3d`. Were
//! one used, this would be the one 3-D op the rank-4 conversion cannot remove, since
//! folding the disparity axis into batch splits each `(b, c)` normalisation set into
//! `D` separate ones (`core/rank4.py:169`).
//!
//! Grid: `B * C` — one CTA per instance. Each CTA walks its `N` elements in
//! `BLOCK`-sized chunks three times: once for the mean, once for the variance about
//! that mean, and once to write the output. The two-pass variance is used rather than
//! `E[x^2] - E[x]^2` because a cost volume's `N` runs to hundreds of thousands of
//! elements, where the single-pass form loses precision in f32.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// Instance-normalise each `(b, c)` slice over its trailing elements.
///
/// Input:  `x_ptr`   — `[B, C, N]` row-major
/// Output: `out_ptr` — `[B, C, N]`
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn instance_norm3d_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    N: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    // One CTA per (batch, channel) instance.
    let pid = T::program_id(Axis::X);
    let base = pid * N;
    let n_chunks = (N + BLOCK - 1) / BLOCK;

    let offs = T::arange(0, BLOCK);
    let zeros = T::zeros::<D>(&[BLOCK]);
    // N as a float, broadcast across the block.
    let n_f = T::cast::<i32, D>(T::zeros::<i32>(&[BLOCK]) + N, None, false);

    // Pass 1: mean. Masked lanes load 0 and add nothing.
    let mut acc = T::zeros::<D>(&[BLOCK]);
    for i in 0..n_chunks {
        let idx = offs + i * BLOCK;
        let x = T::load(
            x_ptr.add_offsets(idx + base),
            Some(idx.lt(N)),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        acc = acc + x;
    }
    let mean = T::sum(acc, Some(0), true) / n_f;

    // Pass 2: biased variance about that mean. Masked lanes must contribute 0, not
    // (0 - mean)^2, so the deviation is selected against the mask.
    let mut acc2 = T::zeros::<D>(&[BLOCK]);
    for i in 0..n_chunks {
        let idx = offs + i * BLOCK;
        let m = idx.lt(N);
        let x = T::load(
            x_ptr.add_offsets(idx + base),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let d = T::where_(m, x - mean, zeros);
        acc2 = acc2 + d * d;
    }
    let var = T::sum(acc2, Some(0), true) / n_f;
    let inv_std = T::rsqrt(var + T::full(&[BLOCK], D::from_f64(1e-5)));

    // Pass 3: normalise and write.
    for i in 0..n_chunks {
        let idx = offs + i * BLOCK;
        let m = idx.lt(N);
        let x = T::load(
            x_ptr.add_offsets(idx + base),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        T::store(
            out_ptr.add_offsets(idx + base),
            (x - mean) * inv_std,
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

/// Graph-level representation of `InstanceNorm3d` (no affine, no running stats).
///
/// Shape-preserving. The input may have any rank >= 2: axes 0 and 1 are batch and
/// channel, and every axis after them is reduced over.
pub struct InstanceNorm3dOp<D: FloatBytes + Send + Sync + 'static> {
    /// Kernel launch block size along the flattened trailing axis.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> InstanceNorm3dOp<D> {
    /// Creates the op with the launch block size along the flattened trailing axis.
    pub fn new(block: i32) -> Self {
        Self {
            block,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for InstanceNorm3dOp<D> {
    fn name(&self) -> &str {
        "stereo.instance_norm3d"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = InstanceNorm3dForward::<D>::new(self.block);
        Some((
            "instance_norm3d_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(InstanceNorm3dRuntimeOp::<D>::new(self.block)),
        ))
    }
}

/// Runtime dispatch for `InstanceNorm3d`.
pub struct InstanceNorm3dRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> InstanceNorm3dRuntimeOp<D> {
    /// Creates the runtime op with the launch block size along the trailing axis.
    pub fn new(block: i32) -> Self {
        Self {
            block,
            _phantom: PhantomData,
        }
    }
}

/// Number of instances (`B*C`) and elements per instance for a `[B, C, ...]` shape.
fn instances_and_n(shape: &[usize]) -> (usize, i32) {
    let n: usize = shape[2..].iter().product::<usize>().max(1);
    (shape[0] * shape[1], n as i32)
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for InstanceNorm3dRuntimeOp<D> {
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
        let (_, n) = instances_and_n(output_shape);
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(n);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let _ = self.block;
        let (instances, _) = instances_and_n(output_shape);
        [instances as u32, 1, 1]
    }
}
