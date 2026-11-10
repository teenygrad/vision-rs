//! BCE classification loss Triton kernel for YOLO detection training.
//!
//! Computes numerically-stable binary cross-entropy loss between predicted
//! class logits and soft target labels, summed over all C classes per anchor.
//!
//! Stable BCE formula (avoids log(0) and exp overflow):
//!   loss = relu(x) − x·t + log(1 + exp(−|x|))
//!        = max(x, 0) − x·t + log(1 + exp(−|x|))
//!
//! Layout:
//!   - `pred`:   `[C, N]` — predicted class logits per anchor
//!   - `target`: `[C, N]` — soft class labels ∈ [0, 1] per anchor
//!   - `loss`:   `[N]`    — per-anchor BCE loss (summed over C classes)
//!
//! Parallelism: one CTA per BLOCK_N-wide anchor tile; classes iterated
//! sequentially inside each CTA.
//! Grid: `cdiv(N, BLOCK_N)` flat CTAs.

#![allow(non_snake_case)]

use teeny_core::dtype::Float;
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

/// BCE classification loss forward: class logits + soft targets → per-anchor loss.
///
/// Grid: `cdiv(N, BLOCK_N)` — one CTA per anchor tile.
#[allow(clippy::erasing_op, clippy::identity_op)]
#[kernel]
pub fn yolo_bce_cls_loss_forward<T: Triton, D: Float, const BLOCK_N: i32>(
    pred_ptr: InPtr<T::Pointer<D>>,
    target_ptr: InPtr<T::Pointer<D>>,
    loss_ptr: OutPtr<T::Pointer<D>>,
    N: i32,
    C: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let n_start = T::program_id(Axis::X) * BLOCK_N;
    let n_offs = T::arange(0, BLOCK_N) + n_start;
    let mask = n_offs.lt(N);
    let zeros = T::zeros::<D>(&[BLOCK_N]);
    let ones = T::full(&[BLOCK_N], D::from_f64(1.0));

    // Accumulate BCE over all C classes for this anchor tile.
    let mut acc = zeros;
    let mut c: i32 = 0;
    while c < C {
        // Layout [C, N]: class c occupies row c, at base offset c*N.
        let base = c * N;
        let x = T::load(
            pred_ptr.add_offsets(n_offs + base),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let t = T::load(
            target_ptr.add_offsets(n_offs + base),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );

        // Numerically-stable BCE: max(x, 0) − x·t + log(1 + exp(−|x|))
        let relu_x = T::maximum(x, zeros);
        let log1p_exp = T::log(ones + T::exp(zeros - T::abs(x)));
        let bce = relu_x - x * t + log1p_exp;

        acc = acc + bce;
        c += 1;
    }

    T::store(
        loss_ptr.add_offsets(n_offs),
        acc,
        Some(mask),
        &[],
        None,
        None,
    );
}

/// BCE classification loss backward: upstream gradient + forward inputs → ∂L/∂pred.
///
/// Per element: `d_pred[c,n] = dy[n] · (sigmoid(pred[c,n]) − target[c,n])`
///
/// Uses the numerically stable sigmoid: `σ(x) = 1 / (1 + exp(−x))`.
/// No saved activations are needed because the gradient depends only on the
/// forward inputs (pred logits and targets).
///
/// Grid: `cdiv(N, BLOCK_N)` — one CTA per anchor tile.
#[allow(clippy::erasing_op, clippy::identity_op)]
#[kernel]
pub fn yolo_bce_cls_loss_backward<T: Triton, D: Float, const BLOCK_N: i32>(
    dy_ptr: InPtr<T::Pointer<D>>,
    pred_ptr: InPtr<T::Pointer<D>>,
    target_ptr: InPtr<T::Pointer<D>>,
    d_pred_ptr: OutPtr<T::Pointer<D>>,
    N: i32,
    C: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let n_start = T::program_id(Axis::X) * BLOCK_N;
    let n_offs = T::arange(0, BLOCK_N) + n_start;
    let mask = n_offs.lt(N);
    let zeros = T::zeros::<D>(&[BLOCK_N]);
    let ones = T::full(&[BLOCK_N], D::from_f64(1.0));
    let neg_one = T::full(&[BLOCK_N], D::from_f64(-1.0));

    let dy = T::load(
        dy_ptr.add_offsets(n_offs),
        Some(mask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );

    let mut c: i32 = 0;
    while c < C {
        let base = c * N;
        let x = T::load(
            pred_ptr.add_offsets(n_offs + base),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let t = T::load(
            target_ptr.add_offsets(n_offs + base),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );

        // σ(x) = 1 / (1 + exp(−x))
        let sig = ones / (ones + T::exp(neg_one * x));
        let grad = dy * (sig - t);

        T::store(
            d_pred_ptr.add_offsets(n_offs + base),
            grad,
            Some(mask),
            &[],
            None,
            None,
        );
        c += 1;
    }
}
