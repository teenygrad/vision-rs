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

//! 1-D bilinear sampler — the correlation-volume lookup used by stereo matching.
//!
//! This is the stereo special case of `grid_sample`. FoundationStereo's
//! `bilinear_sampler` (core/utils/utils.py) asserts
//!
//! ```text
//! assert torch.unique(ygrid).numel() == 1 and H == 1  # This is a stereo problem
//! ```
//!
//! so the sampled signal is always height-1 with a constant y grid. A full 2-D
//! `GridSample` is therefore unnecessary: the operation reduces to a linear
//! interpolation between two adjacent columns.
//!
//! **Layout**: `img` is `[N, C, W]` row-major, where `N` folds every batch/pixel
//! index (`B*H*W` in the caller). `coords` is `[N, K]` holding sample positions in
//! **pixel** units along `W`. Output is `[N, C, K]`.
//!
//! Positions outside `[0, W-1]` contribute zero, matching `grid_sample`'s
//! `padding_mode="zeros"` with `align_corners=True`.
//!
//! Grid: `N * C` — one CTA per `(n, channel)` pair. `K` is small in practice
//! (`2*radius+1`, radius 4 → 9), so a single `BLOCK_K` tile covers it.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// 1-D bilinear gather along the last axis of `img`.
///
/// Inputs:
///   `img_ptr`    — `[N, C, W]` row-major
///   `coords_ptr` — `[N, K]` sample positions in pixel units
///
/// Output:
///   `out_ptr`    — `[N, C, K]`
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn bilinear_sampler_1d_forward<T: Triton, D: Float, const BLOCK_K: i32>(
    img_ptr: InPtr<T::Pointer<D>>,
    coords_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    C: i32,
    W: i32,
    K: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Tensor<D>: Comparison<T::Tensor<D>, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let n = pid / C;
    let c = pid % C;

    let k_offs = T::arange(0, BLOCK_K);
    let kmask = k_offs.lt(K);

    let fzero = T::zeros::<D>(&[BLOCK_K]);
    let one = T::full(&[BLOCK_K], D::from_f64(1.0));
    let wmax = T::full(&[BLOCK_K], D::from_f64((W - 1) as f64));

    // Sample positions, in pixel units along W.
    let x = T::load(
        coords_ptr.add_offsets(k_offs + (n * K)),
        Some(kmask),
        Some(fzero),
        &[],
        None,
        None,
        None,
        false,
    );

    // Left/right taps and their interpolation weights.
    let x0f = T::floor(x);
    let x1f = x0f + one;
    let w1 = x - x0f;
    let w0 = one - w1;

    // grid_sample(padding_mode="zeros"): a tap outside [0, W-1] contributes nothing.
    let in0 = x0f.ge(fzero);
    let in0b = x0f.le(wmax);
    let in1 = x1f.ge(fzero);
    let in1b = x1f.le(wmax);

    // Clamp in float so every load is in bounds, then zero the invalid taps below.
    let x0c = T::minimum(T::maximum(x0f, fzero), wmax);
    let x1c = T::minimum(T::maximum(x1f, fzero), wmax);
    let i0 = T::cast::<D, i32>(x0c, None, false);
    let i1 = T::cast::<D, i32>(x1c, None, false);

    let base = (n * C + c) * W;
    let v0 = T::load(
        img_ptr.add_offsets(i0 + base),
        Some(kmask),
        Some(fzero),
        &[],
        None,
        None,
        None,
        false,
    );
    let v1 = T::load(
        img_ptr.add_offsets(i1 + base),
        Some(kmask),
        Some(fzero),
        &[],
        None,
        None,
        None,
        false,
    );

    let v0 = T::where_(in0, v0, fzero);
    let v0 = T::where_(in0b, v0, fzero);
    let v1 = T::where_(in1, v1, fzero);
    let v1 = T::where_(in1b, v1, fzero);

    let acc = v0 * w0 + v1 * w1;

    T::store(
        out_ptr.add_offsets(k_offs + ((n * C + c) * K)),
        acc,
        Some(kmask),
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

/// Graph-level representation of the 1-D bilinear sampler.
///
/// Inputs, in order: `img` `[N, C, W]`, `coords` `[N, K]`. Output `[N, C, K]`.
pub struct BilinearSampler1dOp<D: FloatBytes + Send + Sync + 'static> {
    /// Kernel launch block size along the sample axis; must be >= `K`.
    pub block_k: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> BilinearSampler1dOp<D> {
    /// Creates the op with the launch block size along the sample axis.
    pub fn new(block_k: i32) -> Self {
        Self {
            block_k,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for BilinearSampler1dOp<D> {
    fn name(&self) -> &str {
        "stereo.bilinear_sampler_1d"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        // img [N, C, W] + coords [N, K] -> [N, C, K]
        let img = input_shapes[0];
        let coords = input_shapes[1];
        vec![img[0], img[1], coords[1]]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = BilinearSampler1dForward::<D>::new(self.block_k);
        Some((
            "bilinear_sampler_1d_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(BilinearSampler1dRuntimeOp::<D>::new(self.block_k)),
        ))
    }
}

/// Runtime dispatch for the 1-D bilinear sampler.
pub struct BilinearSampler1dRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    block_k: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> BilinearSampler1dRuntimeOp<D> {
    /// Creates the runtime op with the launch block size along the sample axis.
    pub fn new(block_k: i32) -> Self {
        Self {
            block_k,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for BilinearSampler1dRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        2 // img, coords
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        Vec::new() // no learned parameters
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
        let w = inputs[0].1[2] as i32; // img [N, C, W]
        let c = output_shape[1] as i32;
        let k = output_shape[2] as i32;
        visitor.visit_ptr(inputs[0].0); // img_ptr
        visitor.visit_ptr(inputs[1].0); // coords_ptr
        visitor.visit_ptr(output); // out_ptr
        visitor.visit_i32(c);
        visitor.visit_i32(w);
        visitor.visit_i32(k);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        // One CTA per (n, channel); K is covered by a single BLOCK_K tile.
        let _ = self.block_k;
        [(output_shape[0] * output_shape[1]) as u32, 1, 1]
    }
}
