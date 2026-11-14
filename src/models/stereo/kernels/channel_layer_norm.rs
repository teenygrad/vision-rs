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

//! LayerNorm over the channel axis of `[B, C, ...]`, with a per-channel affine:
//! timm's `LayerNorm2d`, and equally timm's channels-last `nn.LayerNorm(C)` as EdgeNeXt
//! applies it after permuting NCHW to NHWC.
//!
//! teeny-kernels' `Op::LayerNorm` normalises the last axis, and `Transpose` cannot be
//! lowered, so a channel LayerNorm on NCHW data needs its own kernel. EdgeNeXt uses one
//! after its stem, before every downsample and inside every block.
//!
//! **Layout**: `[B, C, S]`, `S` flattening the trailing axes. Grid:
//! `B * ceil(S / BLOCK)`, lanes on positions; the channel axis is walked three times
//! (mean, biased variance about it, normalised affine write) in consecutive flat loops.
//! Two-pass variance, as PyTorch computes it.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// `out[b, c, s] = (x - mean_c) / sqrt(var_c + eps) * weight[c] + bias[c]`.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn channel_layer_norm_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    w_ptr: InPtr<T::Pointer<D>>,
    b_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    C: i32,
    S: i32,
    eps: f32,
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
    let b = pid / tiles;

    let s = T::arange(0, BLOCK) + tile * BLOCK;
    let m = s.lt(S);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let zero_idx = T::zeros::<i32>(&[BLOCK]);
    let base = s + b * C * S;
    let c_f = T::cast::<i32, D>(zero_idx + C, None, false);

    // Each loop's result goes through an exact identity op before the next loop:
    // teenyc's RISC-V backend panics on back-to-back loops (teenygrad/teenygrad#156).
    let mut acc = T::zeros::<D>(&[BLOCK]);
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
        acc = acc + x;
    }
    let mean = acc / c_f;
    let mut acc2 = T::zeros::<D>(&[BLOCK]);
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
        let d = x - mean;
        acc2 = acc2 + d * d;
    }
    let eps_t = T::cast::<f32, D>(T::full::<f32>(&[BLOCK], eps), None, false);
    let inv_std = T::rsqrt(acc2 / c_f + eps_t);
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
        // Per-channel affine, broadcast across the block.
        let w = T::load(
            w_ptr.add_offsets(zero_idx + c),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let bb = T::load(
            b_ptr.add_offsets(zero_idx + c),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        T::store(
            out_ptr.add_offsets(base + c * S),
            (x - mean) * inv_std * w + bb,
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

/// Graph-level channel LayerNorm. Shape-preserving, rank >= 2. Weight slots `weight`
/// and `bias`, both `[C]`.
pub struct ChannelLayerNormOp<D: FloatBytes + Send + Sync + 'static> {
    /// Channels.
    pub c: usize,
    /// Variance epsilon (1e-6 in EdgeNeXt).
    pub eps: f32,
    /// Kernel launch block size along the flattened trailing axes.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> ChannelLayerNormOp<D> {
    /// Creates the op.
    pub fn new(c: usize, eps: f32, block: i32) -> Self {
        Self {
            c,
            eps,
            block,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for ChannelLayerNormOp<D> {
    fn name(&self) -> &str {
        "stereo.channel_layer_norm"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = ChannelLayerNormForward::<D>::new(self.block);
        Some((
            "channel_layer_norm_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(ChannelLayerNormRuntimeOp::<D> {
                c: self.c,
                eps: self.eps,
                block: self.block,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`ChannelLayerNormOp`].
pub struct ChannelLayerNormRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    c: usize,
    eps: f32,
    block: i32,
    _phantom: PhantomData<D>,
}

fn trailing(shape: &[usize]) -> usize {
    shape[2..].iter().product::<usize>().max(1)
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for ChannelLayerNormRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![self.c], vec![self.c]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight", "bias"]
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        params: &[RawPtr],
        output: RawPtr,
        output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        assert_eq!(output_shape[1], self.c, "channel_layer_norm: channels");
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(params[1]);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.c as i32);
        visitor.visit_i32(trailing(output_shape) as i32);
        visitor.visit_f32(self.eps);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let tiles = trailing(output_shape).div_ceil(self.block as usize);
        [(output_shape[0] * tiles) as u32, 1, 1]
    }
}
