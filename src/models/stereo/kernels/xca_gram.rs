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

//! The attention logits of timm's `CrossCovarianceAttn` (XCA, EdgeNeXt):
//!
//! ```python
//! q, k, v = qkv.reshape(B, N, 3, heads, d).permute(2, 0, 3, 4, 1)   # [B, heads, d, N]
//! attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)) * temperature
//! ```
//!
//! Attention is across channels: each head's `d x d` map compares channel rows over
//! all `N` positions. On NCHW data the 1x1 `qkv` conv already lays `q`, `k`, `v` out as
//! `[B, 3C, N]` with channel `t*C + h*d + i`, so no permute is needed.
//!
//! Each program computes one logit, `<q_i, k_j> / (max(|q_i|, eps) * max(|k_j|, eps))`
//! times the head's temperature, accumulating `q.q`, `k.k` and `q.k` in one flat pass
//! over `N` (dividing by the norms after the dot product rather than normalising
//! first, which is the same up to rounding). Grid: `B * heads * d * d`. Output
//! `[B, heads, d, d]`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// `attn[b, h, i, j]`, as above.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn xca_gram_forward<T: Triton, D: Float, const BLOCK: i32>(
    qkv_ptr: InPtr<T::Pointer<D>>,
    temp_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    C: i32,
    HEADS: i32,
    N: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let d = C / HEADS;
    let pid = T::program_id(Axis::X);
    let j = pid % d;
    let i = (pid / d) % d;
    let h = (pid / (d * d)) % HEADS;
    let b = pid / (d * d * HEADS);

    let offs = T::arange(0, BLOCK);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let q_row = (b * 3 * C + h * d + i) * N;
    let k_row = (b * 3 * C + C + h * d + j) * N;
    let mut qq = T::zeros::<D>(&[BLOCK]);
    let mut kk = T::zeros::<D>(&[BLOCK]);
    let mut qk = T::zeros::<D>(&[BLOCK]);
    for t in 0..(N + BLOCK - 1) / BLOCK {
        let n = offs + t * BLOCK;
        let m = n.lt(N);
        let q = T::load(
            qkv_ptr.add_offsets(n + q_row),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let k = T::load(
            qkv_ptr.add_offsets(n + k_row),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        qq = qq + q * q;
        kk = kk + k * k;
        qk = qk + q * k;
    }
    let eps = T::full(&[BLOCK], D::from_f64(1e-12));
    let nq = T::maximum(T::sqrt(T::sum(qq, Some(0), true)), eps);
    let nk = T::maximum(T::sqrt(T::sum(kk, Some(0), true)), eps);
    let temp = T::load(
        temp_ptr.add_offsets(T::zeros::<i32>(&[BLOCK]) + h),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let logit = T::sum(qk, Some(0), true) / (nq * nk) * temp;
    // Broadcast across the lanes, written by lane 0 only.
    T::store(
        out_ptr.add_offsets(offs + pid),
        logit,
        Some(offs.lt(1)),
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

/// Graph-level XCA logits: `[B, 3C, H, W]` to `[B, heads, d, d]`. Weight slot `temperature`, `[heads]` (PyTorch's `[heads, 1, 1]`, same data).
pub struct XcaGramOp<D: FloatBytes + Send + Sync + 'static> {
    /// Channels `C` (the input has `3C`).
    pub c: usize,
    /// Attention heads.
    pub heads: usize,
    /// Launch block size along `N`.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> XcaGramOp<D> {
    /// Creates the op.
    pub fn new(c: usize, heads: usize, block: i32) -> Self {
        Self {
            c,
            heads,
            block,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for XcaGramOp<D> {
    fn name(&self) -> &str {
        "stereo.xca_gram"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        let d = self.c / self.heads;
        vec![input_shapes[0][0], Some(self.heads), Some(d), Some(d)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = XcaGramForward::<D>::new(self.block);
        Some((
            "xca_gram_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(XcaGramRuntimeOp::<D> {
                c: self.c,
                heads: self.heads,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`XcaGramOp`].
pub struct XcaGramRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    c: usize,
    heads: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for XcaGramRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![self.heads]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["temperature"]
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        params: &[RawPtr],
        output: RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        let x = inputs[0].1;
        assert_eq!(x[1], 3 * self.c, "xca_gram: qkv channels");
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.c as i32);
        visitor.visit_i32(self.heads as i32);
        visitor.visit_i32(x[2..].iter().product::<usize>() as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [output_shape.iter().product::<usize>() as u32, 1, 1]
    }
}
