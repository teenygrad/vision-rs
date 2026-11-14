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

//! The second half of timm's `CrossCovarianceAttn` (XCA, EdgeNeXt):
//!
//! ```python
//! x = attn.softmax(dim=-1) @ v               # [B, heads, d, N]
//! x = x.permute(0, 3, 1, 2).reshape(B, N, C)
//! ```
//!
//! On NCHW data the output channel is `h*d + i`, so the result is written straight
//! to `[B, C, N]`. Fuses the softmax over `j` with the product: each program walks its
//! head's `d` logits once for the max, then once more accumulating both
//! `sum_j exp(l_j - max)` and `sum_j exp(l_j - max) * v_j`, dividing at the end
//! (equal to normalising first, up to rounding). Logits are loaded broadcast across
//! the lanes, which run along `N`. Grid: `B * C * ceil(N / BLOCK)`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// `out[b, h*d + i, n] = sum_j softmax_j(attn[b, h, i, :]) * v[b, h*d + j, n]`.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn xca_apply_forward<T: Triton, D: Float, const BLOCK: i32>(
    attn_ptr: InPtr<T::Pointer<D>>,
    qkv_ptr: InPtr<T::Pointer<D>>,
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
    let tiles = (N + BLOCK - 1) / BLOCK;
    let pid = T::program_id(Axis::X);
    let tile = pid % tiles;
    let row = pid / tiles; // b * C + h * d + i
    let ch = row % C;
    let b = row / C;
    let h = ch / d;
    let i = ch % d;

    let n = T::arange(0, BLOCK) + tile * BLOCK;
    let m = n.lt(N);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let zero_idx = T::zeros::<i32>(&[BLOCK]);
    let logits = ((b * HEADS + h) * d + i) * d;
    let v_row = (b * 3 * C + 2 * C + h * d) * N;

    let mut mx = T::full(&[BLOCK], D::from_f64(-3.0e38));
    for j in 0..d {
        let l = T::load(
            attn_ptr.add_offsets(zero_idx + (logits + j)),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        mx = T::maximum(mx, l);
    }
    // An exact identity op between the loops: teenyc's RISC-V backend panics on
    // back-to-back loops (teenygrad/teenygrad#156).
    let mx = mx + zeros;
    let mut sum = T::zeros::<D>(&[BLOCK]);
    let mut acc = T::zeros::<D>(&[BLOCK]);
    for j in 0..d {
        let l = T::load(
            attn_ptr.add_offsets(zero_idx + (logits + j)),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let v = T::load(
            qkv_ptr.add_offsets(n + (v_row + j * N)),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let e = T::exp(l - mx);
        sum = sum + e;
        acc = acc + e * v;
    }
    T::store(
        out_ptr.add_offsets(n + row * N),
        acc / sum,
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

/// Graph-level XCA output: inputs `attn` `[B, heads, d, d]` and `qkv` `[B, 3C, H, W]`, output `[B, C, H, W]`.
pub struct XcaApplyOp<D: FloatBytes + Send + Sync + 'static> {
    /// Channels `C`.
    pub c: usize,
    /// Attention heads.
    pub heads: usize,
    /// Launch block size along `N`.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> XcaApplyOp<D> {
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

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for XcaApplyOp<D> {
    fn name(&self) -> &str {
        "stereo.xca_apply"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        let mut s = input_shapes[1].clone();
        s[1] = Some(self.c);
        s
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = XcaApplyForward::<D>::new(self.block);
        Some((
            "xca_apply_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(XcaApplyRuntimeOp::<D> {
                c: self.c,
                heads: self.heads,
                block: self.block,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`XcaApplyOp`].
pub struct XcaApplyRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    c: usize,
    heads: usize,
    block: i32,
    _phantom: PhantomData<D>,
}

fn n_of(shape: &[usize]) -> usize {
    shape[2..].iter().product::<usize>().max(1)
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for XcaApplyRuntimeOp<D> {
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
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.c as i32);
        visitor.visit_i32(self.heads as i32);
        visitor.visit_i32(n_of(output_shape) as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let tiles = n_of(output_shape).div_ceil(self.block as usize);
        [(output_shape[0] * self.c * tiles) as u32, 1, 1]
    }
}
