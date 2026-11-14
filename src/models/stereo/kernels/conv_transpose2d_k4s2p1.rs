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

//! `ConvTranspose2d(kernel_size=4, stride=2, padding=1)` — an exact 2x upsample.
//!
//! This is the only transposed convolution shape FoundationStereo uses
//! (`core/foundation_stereo.py:166`):
//!
//! ```python
//! nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1)
//! ```
//!
//! Output size is `(IH-1)*2 - 2 + 4 = 2*IH`, so it doubles both axes exactly.
//!
//! **Written as a gather, not a scatter.** The scatter form that the name suggests
//! needs atomics; the gather form does not. For an output row `oy`, the contributing
//! kernel rows satisfy `oy = iy*2 - 1 + ky`, so `ky` must match the parity of `oy + 1`.
//! With `ky in 0..4` that leaves exactly two choices, `ky = p + 2j` for `j in {0,1}`
//! and `p = (oy + 1) % 2`, each giving `iy = (oy + 1 - ky) / 2`. The same holds per
//! column, so every output draws from exactly a 2x2 neighbourhood of the input,
//! reduced over `Cin`. Taps whose `iy`/`ix` fall outside the input contribute zero.
//!
//! **Layout**: input `[B, Cin, IH, IW]`, weight `[Cin, Cout, 4, 4]` (PyTorch's
//! transposed-conv layout, in-channels first), bias `[Cout]`, output
//! `[B, Cout, 2*IH, 2*IW]`.
//!
//! Grid: `B * Cout * OH * OW` — one CTA per output element, lanes over `Cin`.
//! `BLOCK_CI` must be >= `Cin`. The four taps accumulate elementwise into a register
//! tile and are reduced once at the end, so the body contains a single reduction.
//!
//! As with the other stereo kernels, index arithmetic is done in float and cast once:
//! integer `maximum`/`minimum` ICE in teenyc 0.3.0 while the float overloads are fine.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// Transposed convolution with `kernel_size=4, stride=2, padding=1`.
///
/// Inputs:  `x_ptr` `[B, Cin, IH, IW]`, `w_ptr` `[Cin, Cout, 4, 4]`, `b_ptr` `[Cout]`
/// (read only when `HAS_BIAS` is 1)
/// Output:  `out_ptr` `[B, Cout, 2*IH, 2*IW]`
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn conv_transpose2d_k4s2p1_forward<T: Triton, D: Float, const BLOCK_CI: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    w_ptr: InPtr<T::Pointer<D>>,
    b_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    CIN: i32,
    COUT: i32,
    IH: i32,
    IW: i32,
    HAS_BIAS: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let oh = IH * 2;
    let ow = IW * 2;

    // One CTA per (batch, out channel, out row, out column).
    let pid = T::program_id(Axis::X);
    let ox = pid % ow;
    let r1 = pid / ow;
    let oy = r1 % oh;
    let r2 = r1 / oh;
    let co = r2 % COUT;
    let b = r2 / COUT;

    let ci = T::arange(0, BLOCK_CI);
    let cimask = ci.lt(CIN);
    let zeros = T::zeros::<D>(&[BLOCK_CI]);
    let izero = T::zeros::<i32>(&[BLOCK_CI]);

    // Index arithmetic is done in i32. An earlier revision computed these offsets in
    // float and cast once, to work around an ICE on integer maximum/minimum; fp32 is
    // only exact for integers below 2^24, and `ci * IH * IW` exceeds that for this
    // model (Cin reaches 512), so the float route had a silent correctness ceiling.
    let plane = IH * IW;
    let ci_off = ci * plane;
    let w_off = ci * (COUT * 16);

    // Only kernel taps matching the parity of oy+1 / ox+1 contribute.
    let py = (oy + 1) % 2;
    let px = (ox + 1) % 2;

    // Four taps, written straight-line and summed into one expression so the body
    // holds a single reduction.
    let xbase = b * CIN * IH * IW;

    // ---- tap (j=0, i=0) ----
    let ky00 = py;
    let kx00 = px;
    let iy00 = izero + ((oy + 1 - ky00) / 2);
    let ix00 = izero + ((ox + 1 - kx00) / 2);
    let xo00 = ci_off
        + (T::maximum(T::minimum(iy00, izero + (IH - 1)), izero) * IW
            + T::maximum(T::minimum(ix00, izero + (IW - 1)), izero));
    let xv00 = T::load(
        x_ptr.add_offsets(xo00 + xbase),
        Some(cimask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let wv00 = T::load(
        w_ptr.add_offsets(w_off + (co * 16 + ky00 * 4 + kx00)),
        Some(cimask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    // Taps whose source falls outside the input contribute nothing.
    let xv00 = T::where_(iy00.ge(0), xv00, zeros);
    let xv00 = T::where_(iy00.lt(IH), xv00, zeros);
    let xv00 = T::where_(ix00.ge(0), xv00, zeros);
    let xv00 = T::where_(ix00.lt(IW), xv00, zeros);
    let t00 = xv00 * wv00;

    // ---- tap (j=0, i=1) ----
    let ky01 = py;
    let kx01 = px + 2;
    let iy01 = izero + ((oy + 1 - ky01) / 2);
    let ix01 = izero + ((ox + 1 - kx01) / 2);
    let xo01 = ci_off
        + (T::maximum(T::minimum(iy01, izero + (IH - 1)), izero) * IW
            + T::maximum(T::minimum(ix01, izero + (IW - 1)), izero));
    let xv01 = T::load(
        x_ptr.add_offsets(xo01 + xbase),
        Some(cimask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let wv01 = T::load(
        w_ptr.add_offsets(w_off + (co * 16 + ky01 * 4 + kx01)),
        Some(cimask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    // Taps whose source falls outside the input contribute nothing.
    let xv01 = T::where_(iy01.ge(0), xv01, zeros);
    let xv01 = T::where_(iy01.lt(IH), xv01, zeros);
    let xv01 = T::where_(ix01.ge(0), xv01, zeros);
    let xv01 = T::where_(ix01.lt(IW), xv01, zeros);
    let t01 = xv01 * wv01;

    // ---- tap (j=1, i=0) ----
    let ky10 = py + 2;
    let kx10 = px;
    let iy10 = izero + ((oy + 1 - ky10) / 2);
    let ix10 = izero + ((ox + 1 - kx10) / 2);
    let xo10 = ci_off
        + (T::maximum(T::minimum(iy10, izero + (IH - 1)), izero) * IW
            + T::maximum(T::minimum(ix10, izero + (IW - 1)), izero));
    let xv10 = T::load(
        x_ptr.add_offsets(xo10 + xbase),
        Some(cimask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let wv10 = T::load(
        w_ptr.add_offsets(w_off + (co * 16 + ky10 * 4 + kx10)),
        Some(cimask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    // Taps whose source falls outside the input contribute nothing.
    let xv10 = T::where_(iy10.ge(0), xv10, zeros);
    let xv10 = T::where_(iy10.lt(IH), xv10, zeros);
    let xv10 = T::where_(ix10.ge(0), xv10, zeros);
    let xv10 = T::where_(ix10.lt(IW), xv10, zeros);
    let t10 = xv10 * wv10;

    // ---- tap (j=1, i=1) ----
    let ky11 = py + 2;
    let kx11 = px + 2;
    let iy11 = izero + ((oy + 1 - ky11) / 2);
    let ix11 = izero + ((ox + 1 - kx11) / 2);
    let xo11 = ci_off
        + (T::maximum(T::minimum(iy11, izero + (IH - 1)), izero) * IW
            + T::maximum(T::minimum(ix11, izero + (IW - 1)), izero));
    let xv11 = T::load(
        x_ptr.add_offsets(xo11 + xbase),
        Some(cimask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let wv11 = T::load(
        w_ptr.add_offsets(w_off + (co * 16 + ky11 * 4 + kx11)),
        Some(cimask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    // Taps whose source falls outside the input contribute nothing.
    let xv11 = T::where_(iy11.ge(0), xv11, zeros);
    let xv11 = T::where_(iy11.lt(IH), xv11, zeros);
    let xv11 = T::where_(ix11.ge(0), xv11, zeros);
    let xv11 = T::where_(ix11.lt(IW), xv11, zeros);
    let t11 = xv11 * wv11;

    // Single reduction over Cin, then bias. With HAS_BIAS = 0 the caller passes any
    // valid buffer as `b_ptr` (the graph op passes the weights) and the value is
    // multiplied by zero, which is exact for finite weights.
    let prod = t00 + t01 + t10 + t11;
    let dot = T::sum(prod, Some(0), false) / T::full(&[1], D::from_f64(1.0));
    let has_bias = T::cast::<i32, D>(T::zeros::<i32>(&[1]) + HAS_BIAS, None, false);
    let bias = T::load(
        b_ptr.add_offsets(T::arange(0, 1) + co),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );

    T::store(
        out_ptr.add_offsets(T::arange(0, 1) + pid),
        dot + bias * has_bias,
        None,
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

/// Graph-level `nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1)`.
///
/// Input `[B, cin, IH, IW]`, output `[B, cout, 2*IH, 2*IW]`. Weight slots are
/// `weight` `[cin, cout, 4, 4]` (PyTorch's transposed-conv layout) and, with
/// `has_bias`, `bias` `[cout]`, so checkpoint tensors load unchanged.
pub struct ConvTranspose2dK4s2p1Op<D: FloatBytes + Send + Sync + 'static> {
    /// Input channels.
    pub cin: usize,
    /// Output channels.
    pub cout: usize,
    /// Whether the layer has a bias.
    pub has_bias: bool,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> ConvTranspose2dK4s2p1Op<D> {
    /// Creates the op.
    pub fn new(cin: usize, cout: usize, has_bias: bool) -> Self {
        Self {
            cin,
            cout,
            has_bias,
            _phantom: PhantomData,
        }
    }

    /// Channel block: the smallest power of two covering `cin`.
    fn block_ci(&self) -> i32 {
        self.cin.next_power_of_two().max(1) as i32
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for ConvTranspose2dK4s2p1Op<D> {
    fn name(&self) -> &str {
        "stereo.conv_transpose2d_k4s2p1"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        let x = input_shapes[0];
        vec![
            x[0],
            Some(self.cout),
            x[2].map(|h| 2 * h),
            x[3].map(|w| 2 * w),
        ]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = ConvTranspose2dK4s2p1Forward::<D>::new(self.block_ci());
        Some((
            "conv_transpose2d_k4s2p1_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(ConvTranspose2dK4s2p1RuntimeOp::<D> {
                cin: self.cin,
                cout: self.cout,
                has_bias: self.has_bias,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`ConvTranspose2dK4s2p1Op`].
pub struct ConvTranspose2dK4s2p1RuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    cin: usize,
    cout: usize,
    has_bias: bool,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for ConvTranspose2dK4s2p1RuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        let mut p = vec![vec![self.cin, self.cout, 4, 4]];
        if self.has_bias {
            p.push(vec![self.cout]);
        }
        p
    }

    fn param_names(&self) -> &'static [&'static str] {
        if self.has_bias {
            &["weight", "bias"]
        } else {
            &["weight"]
        }
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
        assert_eq!(x[1], self.cin, "conv_transpose2d_k4s2p1: input channels");
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        // Without a bias the kernel multiplies the loaded value by zero; any valid
        // buffer will do, so pass the weights.
        visitor.visit_ptr(if self.has_bias { params[1] } else { params[0] });
        visitor.visit_ptr(output);
        visitor.visit_i32(self.cin as i32);
        visitor.visit_i32(self.cout as i32);
        visitor.visit_i32(x[2] as i32);
        visitor.visit_i32(x[3] as i32);
        visitor.visit_i32(self.has_bias as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [output_shape.iter().product::<usize>() as u32, 1, 1]
    }
}
