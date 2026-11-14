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

//! One pyramid level of FoundationStereo's geometry-encoding lookup
//! (`Combined_Geo_Encoding_Volume.__call__`, `core/geometry.py`), fused into a single
//! kernel that writes channel-first output.
//!
//! For each pixel `(b, h, w)` with current disparity `d = disp[b, 0, h, w]`, level `i`
//! (`s = 2^i`) and offsets `dx_k = k - R` for `k in 0..2R+1`:
//!
//! ```text
//! geo:  out[b, c*K + k, h, w] = sample(geo[b, c, :, h, w],  dx_k + d / s)
//! corr: out[b, C*K + k, h, w] = sample(corr[b, h, w, :],     w / s - d / s + dx_k)
//! ```
//!
//! `sample` is 1-D linear interpolation in pixel units with zero padding, which is
//! what FoundationStereo's `bilinear_sampler` does through
//! `grid_sample(align_corners=True)` on a height-1 image (the same rule as
//! [`super::bilinear_sampler_1d`]).
//!
//! PyTorch gets there by permuting the cost volume to `[B*H*W, C, 1, D]`, sampling to
//! `[B*H*W, C, 1, K]`, reshaping to `[B, H, W, C*K]` and permuting back to channels
//! first. teeny-kernels cannot lower `Transpose`, so this kernel reads the cost volume
//! in its native `[B, C, D, H, W]` layout and the correlation in the
//! `[B, H, W, W2]` layout [`super::correlation`] produces, and writes
//! `[B, C*K + K, H, W]` directly, with the channel order PyTorch's reshape gives.
//!
//! Grid: `B * H * ceil(W / BLOCK_W)`, lanes on adjacent columns. The `C*K` cost-volume
//! samples and the `K` correlation samples are two consecutive flat loops (a nested
//! loop is miscompiled by teenyc, teenygrad/teenygrad#155).

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// Geometry-encoding lookup for one pyramid level.
///
/// Inputs: `geo_ptr` `[B, C, DL, H, W]`, `corr_ptr` `[B, H, W, W2L]`,
/// `disp_ptr` `[B, 1, H, W]`. Output: `out_ptr` `[B, (C + 1) * (2R + 1), H, W]`.
/// `inv_scale` is `1 / 2^level`.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn geo_lookup_forward<T: Triton, D: Float, const BLOCK_W: i32>(
    geo_ptr: InPtr<T::Pointer<D>>,
    corr_ptr: InPtr<T::Pointer<D>>,
    disp_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    C: i32,
    DL: i32,
    H: i32,
    W: i32,
    W2L: i32,
    R: i32,
    inv_scale: f32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    // One CTA per (batch, row, column tile).
    let tiles = (W + BLOCK_W - 1) / BLOCK_W;
    let pid = T::program_id(Axis::X);
    let tile = pid % tiles;
    let r1 = pid / tiles;
    let h = r1 % H;
    let b = r1 / H;

    let K = 2 * R + 1;
    let CO = (C + 1) * K;
    let HW = H * W;

    let ws = T::arange(0, BLOCK_W) + tile * BLOCK_W;
    let wmask = ws.lt(W);
    let zeros = T::zeros::<D>(&[BLOCK_W]);
    let one = T::full(&[BLOCK_W], D::from_f64(1.0));
    let izero = T::zeros::<i32>(&[BLOCK_W]);
    let inv = T::full(&[BLOCK_W], D::from_f64(inv_scale as f64));

    // d / s, per lane.
    let d = T::load(
        disp_ptr.add_offsets(ws + (b * HW + h * W)),
        Some(wmask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let d_s = d * inv;
    // w / s - d / s: the left pixel's column, mapped onto the right image.
    let wf = T::cast::<i32, D>(ws, None, false);
    let xr = wf * inv - d_s;

    // ---- cost volume: C * K samples along D ---------------------------------
    for t in 0..C * K {
        let c = t / K;
        let k = t % K;
        let dxk = T::cast::<i32, D>(izero + (k - R), None, false);
        let x = dxk + d_s;

        let x0f = T::floor(x);
        let w1 = x - x0f;
        let w0 = one - w1;
        let i0 = T::cast::<D, i32>(x0f, None, false);
        let i1 = i0 + 1;
        let i0c = T::maximum(T::minimum(i0, izero + (DL - 1)), izero);
        let i1c = T::maximum(T::minimum(i1, izero + (DL - 1)), izero);

        // geo[b, c, i, h, w]
        let base = (b * C + c) * DL * HW + h * W;
        let v0 = T::load(
            geo_ptr.add_offsets(i0c * HW + ws + base),
            Some(wmask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let v1 = T::load(
            geo_ptr.add_offsets(i1c * HW + ws + base),
            Some(wmask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        // Zero padding: a tap outside [0, DL - 1] contributes nothing.
        let v0 = T::where_(i0.ge(0), v0, zeros);
        let v0 = T::where_(i0.lt(DL), v0, zeros);
        let v1 = T::where_(i1.ge(0), v1, zeros);
        let v1 = T::where_(i1.lt(DL), v1, zeros);

        T::store(
            out_ptr.add_offsets(ws + ((b * CO + c * K + k) * HW + h * W)),
            v0 * w0 + v1 * w1,
            Some(wmask),
            &[],
            None,
            None,
        );
    }

    // ---- correlation: K samples along W2 ------------------------------------
    // corr[b, h, w, j] is at ((b * H + h) * W + w) * W2L + j.
    let row = ws * W2L + (b * H + h) * W * W2L;
    for k in 0..K {
        let dxk = T::cast::<i32, D>(izero + (k - R), None, false);
        let x = xr + dxk;

        let x0f = T::floor(x);
        let w1 = x - x0f;
        let w0 = one - w1;
        let i0 = T::cast::<D, i32>(x0f, None, false);
        let i1 = i0 + 1;
        let i0c = T::maximum(T::minimum(i0, izero + (W2L - 1)), izero);
        let i1c = T::maximum(T::minimum(i1, izero + (W2L - 1)), izero);

        let v0 = T::load(
            corr_ptr.add_offsets(row + i0c),
            Some(wmask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let v1 = T::load(
            corr_ptr.add_offsets(row + i1c),
            Some(wmask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let v0 = T::where_(i0.ge(0), v0, zeros);
        let v0 = T::where_(i0.lt(W2L), v0, zeros);
        let v1 = T::where_(i1.ge(0), v1, zeros);
        let v1 = T::where_(i1.lt(W2L), v1, zeros);

        T::store(
            out_ptr.add_offsets(ws + ((b * CO + C * K + k) * HW + h * W)),
            v0 * w0 + v1 * w1,
            Some(wmask),
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

/// Graph-level geometry lookup for one pyramid level.
///
/// Inputs, in order: `geo` `[B, C, DL, H, W]`, `corr` `[B, H, W, W2L]`,
/// `disp` `[B, 1, H, W]`. Output `[B, (C + 1) * (2R + 1), H, W]`.
pub struct GeoLookupOp<D: FloatBytes + Send + Sync + 'static> {
    /// Pyramid level `i`; samples are taken at `disp / 2^i`.
    pub level: u32,
    /// Lookup radius `R` (`corr_radius`).
    pub radius: usize,
    /// Kernel launch block size along the column axis.
    pub block_w: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> GeoLookupOp<D> {
    /// Creates the op for `level` and `radius`.
    pub fn new(level: u32, radius: usize, block_w: i32) -> Self {
        Self {
            level,
            radius,
            block_w,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for GeoLookupOp<D> {
    fn name(&self) -> &str {
        "stereo.geo_lookup"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        let geo = input_shapes[0];
        let k = 2 * self.radius + 1;
        let c = geo[1].map(|c| (c + 1) * k);
        vec![geo[0], c, geo[3], geo[4]]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = GeoLookupForward::<D>::new(self.block_w);
        Some((
            "geo_lookup_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(GeoLookupRuntimeOp::<D>::new(
                self.level,
                self.radius,
                self.block_w,
            )),
        ))
    }
}

/// Runtime dispatch for [`GeoLookupOp`].
pub struct GeoLookupRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    level: u32,
    radius: usize,
    block_w: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> GeoLookupRuntimeOp<D> {
    /// Creates the runtime op.
    pub fn new(level: u32, radius: usize, block_w: i32) -> Self {
        Self {
            level,
            radius,
            block_w,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for GeoLookupRuntimeOp<D> {
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
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        let geo = inputs[0].1;
        let corr = inputs[1].1;
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(inputs[2].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(geo[1] as i32); // C
        visitor.visit_i32(geo[2] as i32); // DL
        visitor.visit_i32(geo[3] as i32); // H
        visitor.visit_i32(geo[4] as i32); // W
        visitor.visit_i32(corr[3] as i32); // W2L
        visitor.visit_i32(self.radius as i32);
        visitor.visit_f32(1.0 / (1u32 << self.level) as f32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let tiles = output_shape[3].div_ceil(self.block_w as usize);
        [(output_shape[0] * output_shape[2] * tiles) as u32, 1, 1]
    }
}
