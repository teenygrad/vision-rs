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

//! All-pairs cosine correlation volume for stereo matching.
//!
//! The einsum FoundationStereo builds its initial correlation volume with
//! (`core/geometry.py:74`):
//!
//! ```python
//! corr = torch.einsum('aijk,aijh->ajkh',
//!                     F.normalize(fmap1.float(), dim=1),
//!                     F.normalize(fmap2.float(), dim=1))
//! ```
//!
//! With `a=batch`, `i=channel`, `j=row`, `k=left column`, `h=right column`, that is
//!
//! ```text
//! out[a, j, k, h] = sum_i  n1[a, i, j, k] * n2[a, i, j, h]
//! ```
//!
//! where `n1`, `n2` are `fmap1`, `fmap2` L2-normalised along the channel axis. Since
//! both operands are unit vectors this is a cosine similarity, so the result is in
//! [-1, 1] regardless of feature scale.
//!
//! **Inputs must already be L2-normalised along the channel axis** — use
//! [`super::l2_normalize`] first. This mirrors FoundationStereo, which normalises whole
//! feature maps before the einsum, and keeps the normalised maps reusable.
//!
//! **Layout**: `fmap1` is `[B, C, H, W1]` and `fmap2` is `[B, C, H, W2]`, both
//! row-major. Output is `[B, H, W1, W2]`, which is exactly the layout the 1-D bilinear
//! sampler then consumes as `[N = B*H*W1, 1, W2]`.
//!
//! Grid: `B * H * W1` — one CTA per left pixel, sweeping every right column, so the
//! left column is loaded once and reused across all `W2` dot products.
//!
//! `BLOCK_C` must be >= `C`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// All-pairs correlation of two channel-normalised feature maps.
///
/// Inputs:
///   `fmap1_ptr` — `[B, C, H, W1]` row-major, channel-normalised
///   `fmap2_ptr` — `[B, C, H, W2]` row-major, channel-normalised
///
/// Output:
///   `out_ptr`   — `[B, H, W1, W2]`
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn correlation_forward<T: Triton, D: Float, const BLOCK_C: i32>(
    fmap1_ptr: InPtr<T::Pointer<D>>,
    fmap2_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    C: i32,
    H: i32,
    W1: i32,
    W2: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let k = pid % W1;
    let rest = pid / W1;
    let y = rest % H;
    let b = rest / H;

    let c_offs = T::arange(0, BLOCK_C);
    let cmask = c_offs.lt(C);
    let zeros = T::zeros::<D>(&[BLOCK_C]);
    let s1 = H * W1;
    let s2 = H * W2;

    let f1 = T::load(
        fmap1_ptr.add_offsets(c_offs * s1 + (b * C * s1 + y * W1 + k)),
        Some(cmask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let out_base = ((b * H + y) * W1 + k) * W2;

    for h in 0..W2 {
        let f2 = T::load(
            fmap2_ptr.add_offsets(c_offs * s2 + (b * C * s2 + y * W2 + h)),
            Some(cmask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        // Keepdim reduction, broadcast across the channel lanes and written by
        // lane 0 only. (teenyc 0.3.0 ICEd on this form, teenygrad/teenygrad#119;
        // the branch reduced to a 1-element tensor and stored it unmasked until
        // that was fixed.)
        let dot = T::sum(f1 * f2, Some(0), true);
        T::store(
            out_ptr.add_offsets(c_offs + (out_base + h)),
            dot,
            Some(c_offs.lt(1)),
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

/// Graph-level representation of the all-pairs correlation volume.
///
/// Inputs, in order: `fmap1` `[B, C, H, W1]`, `fmap2` `[B, C, H, W2]`, both already
/// channel-normalised. Output `[B, H, W1, W2]`.
pub struct CorrelationOp<D: FloatBytes + Send + Sync + 'static> {
    /// Kernel launch block size along the channel axis; must be >= `C`.
    pub block_c: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> CorrelationOp<D> {
    /// Creates the op with the launch block size along the channel axis.
    pub fn new(block_c: i32) -> Self {
        Self {
            block_c,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for CorrelationOp<D> {
    fn name(&self) -> &str {
        "stereo.correlation"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        // [B, C, H, W1] + [B, C, H, W2] -> [B, H, W1, W2]
        let f1 = input_shapes[0];
        let f2 = input_shapes[1];
        vec![f1[0], f1[2], f1[3], f2[3]]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = CorrelationForward::<D>::new(self.block_c);
        Some((
            "correlation_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(CorrelationRuntimeOp::<D>::new(self.block_c)),
        ))
    }
}

/// Runtime dispatch for the correlation volume.
pub struct CorrelationRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    block_c: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> CorrelationRuntimeOp<D> {
    /// Creates the runtime op with the launch block size along the channel axis.
    pub fn new(block_c: i32) -> Self {
        Self {
            block_c,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for CorrelationRuntimeOp<D> {
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
        let c = inputs[0].1[1] as i32; // fmap1 [B, C, H, W1]
        let h = output_shape[1] as i32;
        let w1 = output_shape[2] as i32;
        let w2 = output_shape[3] as i32;
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(c);
        visitor.visit_i32(h);
        visitor.visit_i32(w1);
        visitor.visit_i32(w2);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        // One CTA per left pixel: B * H * W1.
        let _ = self.block_c;
        [
            (output_shape[0] * output_shape[1] * output_shape[2]) as u32,
            1,
            1,
        ]
    }
}
