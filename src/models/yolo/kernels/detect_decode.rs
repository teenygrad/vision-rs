//! Fused dist2bbox decode Triton kernel for YOLO detection post-processing.
//!
//! Converts raw LTRB distance predictions from a detection head into
//! XYWH world-coordinate boxes, fusing the dist2bbox conversion and
//! stride-scaling into a single kernel pass.
//!
//! Layout:
//!   - `boxes`:    `[B, 4, A]` — raw LTRB distances
//!   - `anchor_x`: `[A]`       — anchor centre x per anchor
//!   - `anchor_y`: `[A]`       — anchor centre y per anchor
//!   - `strides`:  `[A]`       — stride scale per anchor
//!   - `out`:      `[B, 4, A]` — decoded XYWH boxes in world coordinates
//!
//! Parallelism: one CTA per (batch, BLOCK_A-wide anchor tile).
//! Grid: `B * cdiv(A, BLOCK_A)` flat CTAs.

#![allow(non_snake_case)]

use std::{any::Any, marker::PhantomData, sync::Arc};

use teeny_core::{
    device::program::ArgVisitor,
    dtype::{Float, FloatBytes},
    graph::{CustomOp, Shape},
    model::{RawPtr, RuntimeOp},
};
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

/// Fused dist2bbox + stride-scale decode: LTRB distances → XYWH world coords.
///
/// Grid: `B * cdiv(A, BLOCK_A)` — one CTA per (batch element, anchor tile).
#[allow(clippy::erasing_op, clippy::identity_op)]
#[kernel]
pub fn detect_decode_forward<T: Triton, D: Float, const BLOCK_A: i32>(
    boxes_ptr: InPtr<T::Pointer<D>>,
    anchor_x_ptr: InPtr<T::Pointer<D>>,
    anchor_y_ptr: InPtr<T::Pointer<D>>,
    strides_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    _B: i32,
    A: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let a_tiles = T::cdiv(A, BLOCK_A);
    let pid_b = T::program_id(Axis::X) / a_tiles;
    let a_tile = T::program_id(Axis::X) % a_tiles;
    let a_start = a_tile * BLOCK_A;

    let a_offs = T::arange(0, BLOCK_A) + a_start;
    let mask = a_offs.lt(A);

    let zeros = T::zeros::<D>(&[BLOCK_A]);

    let anchor_x = T::load(
        anchor_x_ptr.add_offsets(a_offs),
        Some(mask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let anchor_y = T::load(
        anchor_y_ptr.add_offsets(a_offs),
        Some(mask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let strides = T::load(
        strides_ptr.add_offsets(a_offs),
        Some(mask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );

    let base = pid_b * 4 * A;
    let dx1 = T::load(
        boxes_ptr.add_offsets(a_offs + (base + 0 * A)),
        Some(mask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let dy1 = T::load(
        boxes_ptr.add_offsets(a_offs + (base + 1 * A)),
        Some(mask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let dx2 = T::load(
        boxes_ptr.add_offsets(a_offs + (base + 2 * A)),
        Some(mask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let dy2 = T::load(
        boxes_ptr.add_offsets(a_offs + (base + 3 * A)),
        Some(mask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );

    let x1 = anchor_x - dx1;
    let x2 = anchor_x + dx2;
    let y1 = anchor_y - dy1;
    let y2 = anchor_y + dy2;

    let half = T::full(&[BLOCK_A], D::from_f64(0.5));
    let cx = (x1 + x2) * half * strides;
    let cy = (y1 + y2) * half * strides;
    let w = (x2 - x1) * strides;
    let h = (y2 - y1) * strides;

    T::store(
        out_ptr.add_offsets(a_offs + (base + 0 * A)),
        cx,
        Some(mask),
        &[],
        None,
        None,
    );
    T::store(
        out_ptr.add_offsets(a_offs + (base + 1 * A)),
        cy,
        Some(mask),
        &[],
        None,
        None,
    );
    T::store(
        out_ptr.add_offsets(a_offs + (base + 2 * A)),
        w,
        Some(mask),
        &[],
        None,
        None,
    );
    T::store(
        out_ptr.add_offsets(a_offs + (base + 3 * A)),
        h,
        Some(mask),
        &[],
        None,
        None,
    );
}

// ---------------------------------------------------------------------------
// DetectDecodeOp — the CustomOp for graph-level representation
// ---------------------------------------------------------------------------

/// Graph-level representation of the detect_decode op.
///
/// Stores precomputed anchor grid and stride data used to build the
/// `DetectDecodeRuntimeOp` at lowering time via `CustomOp::lower()`.
pub struct DetectDecodeOp<D: FloatBytes + Send + Sync + 'static> {
    /// Anchor point x-coordinates, one per anchor.
    pub anchor_x: Vec<f32>,
    /// Anchor point y-coordinates, one per anchor.
    pub anchor_y: Vec<f32>,
    /// Per-anchor stride (8/16/32 for the respective FPN scale).
    pub strides: Vec<f32>,
    /// Kernel launch block size along the anchor dimension.
    pub block_a: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> DetectDecodeOp<D> {
    /// Creates the op with a precomputed anchor grid and launch block size.
    pub fn new(anchor_x: Vec<f32>, anchor_y: Vec<f32>, strides: Vec<f32>, block_a: i32) -> Self {
        Self {
            anchor_x,
            anchor_y,
            strides,
            block_a,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for DetectDecodeOp<D> {
    fn name(&self) -> &str {
        "yolo.detect_decode"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        // boxes [B, 4, A] → [B, 4, A]: shape-preserving
        input_shapes[0].clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = DetectDecodeForward::<D>::new(self.block_a);
        let runtime_op: Arc<dyn RuntimeOp> = Arc::new(DetectDecodeRuntimeOp::<D>::new(
            self.anchor_x.clone(),
            self.anchor_y.clone(),
            self.strides.clone(),
            self.block_a,
        ));
        Some((
            "detect_decode_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            runtime_op,
        ))
    }
}

// ---------------------------------------------------------------------------
// DetectDecodeRuntimeOp — the RuntimeOp for kernel dispatch
// ---------------------------------------------------------------------------

/// Runtime dispatch for the detect_decode kernel.
///
/// Anchor grid and strides are stored here so they can be uploaded to device
/// parameter buffers via [`RuntimeOp::param_init_data`] at model-load time.
pub struct DetectDecodeRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    anchor_x: Vec<f32>,
    anchor_y: Vec<f32>,
    strides: Vec<f32>,
    block_a: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> DetectDecodeRuntimeOp<D> {
    /// Creates the runtime op with a precomputed anchor grid and launch block size.
    pub fn new(anchor_x: Vec<f32>, anchor_y: Vec<f32>, strides: Vec<f32>, block_a: i32) -> Self {
        Self {
            anchor_x,
            anchor_y,
            strides,
            block_a,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for DetectDecodeRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        // input_shapes[0] is boxes: [B, 4, A]
        let a = input_shapes[0][2];
        vec![vec![a], vec![a], vec![a]] // anchor_x, anchor_y, strides
    }

    fn param_init_data(&self, param_idx: usize) -> Option<Vec<u8>> {
        let data: &[f32] = match param_idx {
            0 => &self.anchor_x,
            1 => &self.anchor_y,
            2 => &self.strides,
            _ => return None,
        };
        // Upload the anchor grid / strides in the device buffer's element type
        // `D`, converting from the host-side f32 geometry generically.
        Some(
            data.iter()
                .flat_map(|&f| D::from_f64(f as f64).to_le_bytes())
                .collect(),
        )
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
        let b = output_shape[0] as i32;
        let a = output_shape[2] as i32;
        visitor.visit_ptr(inputs[0].0); // boxes_ptr
        visitor.visit_ptr(params[0]); // anchor_x_ptr
        visitor.visit_ptr(params[1]); // anchor_y_ptr
        visitor.visit_ptr(params[2]); // strides_ptr
        visitor.visit_ptr(output); // out_ptr
        visitor.visit_i32(b); // _B
        visitor.visit_i32(a); // A
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let b = output_shape[0];
        let a = output_shape[2];
        let a_tiles = a.div_ceil(self.block_a as usize);
        [(b * a_tiles) as u32, 1, 1]
    }
}
