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

//! The Fourier features of timm's `PositionalEncodingFourier` (EdgeNeXt, stage 1's
//! `SplitTransposeBlock`), before its `token_projection` 1x1 conv:
//!
//! ```python
//! y_embed = (y + 1) / (H + 1e-6) * 2 * pi          # likewise x_embed over W
//! dim_t = 10000 ** (2 * (i // 2) / 32)             # i in 0..32
//! pos_y = stack(sin(y_embed / dim_t[0::2]), cos(y_embed / dim_t[1::2])).flatten()
//! pos = cat(pos_y, pos_x)                          # [B, 64, H, W]
//! ```
//!
//! The map depends only on the shape, so the kernel has no inputs: it is recorded on
//! the tensor whose shape it takes. Channel `c < 32` is `pos_y[c]`, `c >= 32` is
//! `pos_x[c - 32]`; even `i` takes `sin`, odd `cos`. Grid: `B * 64 * H`, lanes along
//! `W`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// Fourier position features, `[B, 64, H, W]`.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn pos_fourier_forward<T: Triton, D: Float, const BLOCK: i32>(
    out_ptr: OutPtr<T::Pointer<D>>,
    H: i32,
    W: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let y = pid % H;
    let c = (pid / H) % 64;
    let x = T::arange(0, BLOCK);
    let m = x.lt(W);
    let zero_idx = T::zeros::<i32>(&[BLOCK]);
    let is_y = (zero_idx + c).lt(32);
    let i = c % 32;

    let coord = T::where_(
        is_y,
        T::cast::<i32, D>(zero_idx + (y + 1), None, false),
        T::cast::<i32, D>(x + 1, None, false),
    );
    let extent = T::where_(
        is_y,
        T::cast::<i32, D>(zero_idx + H, None, false),
        T::cast::<i32, D>(zero_idx + W, None, false),
    );
    // 2 * pi and ln(10000) / 32 as literals: teenyc's kernel core has no
    // `core::f64::consts`.
    #[allow(clippy::approx_constant)]
    let two_pi = T::full(&[BLOCK], D::from_f64(6.283185307179586));
    let embed = coord / (extent + T::full(&[BLOCK], D::from_f64(1e-6))) * two_pi;
    let dim_t = T::exp(
        T::cast::<i32, D>(zero_idx + 2 * (i / 2), None, false)
            * T::full(&[BLOCK], D::from_f64(9.210340371976184 / 32.0)),
    );
    let v = embed / dim_t;
    let out = T::where_((zero_idx + i % 2).lt(1), T::sin(v), T::cos(v));
    T::store(
        out_ptr.add_offsets(x + pid * W),
        out,
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

/// Graph-level Fourier position features: from a `[B, C, H, W]` tensor (only its shape is used) to `[B, 64, H, W]`.
pub struct PosFourierOp<D: FloatBytes + Send + Sync + 'static> {
    /// Launch block size along `W`; at least `W`.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> PosFourierOp<D> {
    /// Creates the op.
    pub fn new(block: i32) -> Self {
        Self {
            block,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for PosFourierOp<D> {
    fn name(&self) -> &str {
        "stereo.pos_fourier"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        let s = input_shapes[0];
        vec![s[0], Some(64), s[2], s[3]]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = PosFourierForward::<D>::new(self.block);
        Some((
            "pos_fourier_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(PosFourierRuntimeOp::<D> {
                block: self.block,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`PosFourierOp`].
pub struct PosFourierRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for PosFourierRuntimeOp<D> {
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
        let _ = inputs;
        assert!(
            output_shape[3] <= self.block as usize,
            "pos_fourier: W exceeds the block"
        );
        visitor.visit_ptr(output);
        visitor.visit_i32(output_shape[2] as i32);
        visitor.visit_i32(output_shape[3] as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [(output_shape[0] * 64 * output_shape[2]) as u32, 1, 1]
    }
}
