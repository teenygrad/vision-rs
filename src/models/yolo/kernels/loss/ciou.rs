//! Fused CIoU loss Triton kernels for YOLO detection training.
//!
//! Computes Complete-IoU (CIoU) loss between predicted and target boxes,
//! both in XYWH world-coordinate format, in a single fused kernel pass.
//!
//! CIoU = 1 − IoU + d²/c² + α·v
//!
//! where:
//!   - IoU  = intersection / union
//!   - d²   = squared Euclidean distance between box centers
//!   - c²   = squared diagonal of the smallest enclosing box
//!   - v    = (4/π²) · (atan(tw/th) − atan(pw/ph))²  (aspect-ratio consistency)
//!   - α    = v / (1 − IoU + v + ε)
//!
//! Layout:
//!   - `pred`:   `[4, N]` — predicted (cx, cy, w, h) per anchor
//!   - `target`: `[4, N]` — target (cx, cy, w, h) per anchor
//!   - `loss`:   `[N]`    — per-anchor CIoU loss
//!   - `iou`:    `[N]`    — saved IoU (forward → backward)
//!   - `v`:      `[N]`    — saved aspect-ratio term (forward → backward)
//!   - `alpha`:  `[N]`    — saved α coefficient (forward → backward)
//!
//! Parallelism: one CTA per BLOCK_N-wide anchor tile.
//! Grid: `cdiv(N, BLOCK_N)` flat CTAs.

#![allow(non_snake_case, clippy::erasing_op, clippy::identity_op)]

use teeny_core::dtype::Float;
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

/// Fused CIoU loss forward: predicted and target XYWH boxes → per-anchor loss
/// plus saved activations (iou, v, alpha) needed by the backward pass.
///
/// Grid: `cdiv(N, BLOCK_N)` — one CTA per anchor tile.
#[kernel]
pub fn yolo_ciou_loss_forward<T: Triton, D: Float, const BLOCK_N: i32>(
    pred_ptr: InPtr<T::Pointer<D>>,
    target_ptr: InPtr<T::Pointer<D>>,
    loss_ptr: OutPtr<T::Pointer<D>>,
    iou_ptr: OutPtr<T::Pointer<D>>,
    v_ptr: OutPtr<T::Pointer<D>>,
    alpha_ptr: OutPtr<T::Pointer<D>>,
    N: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let n_start = T::program_id(Axis::X) * BLOCK_N;
    let n_offs = T::arange(0, BLOCK_N) + n_start;
    let mask = n_offs.lt(N);
    let zeros = T::zeros::<D>(&[BLOCK_N]);

    // Pred XYWH — layout [4, N]: channel c is at base offset c*N.
    let px = T::load(
        pred_ptr.add_offsets(n_offs + 0 * N),
        Some(mask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let py = T::load(
        pred_ptr.add_offsets(n_offs + 1 * N),
        Some(mask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let pw = T::load(
        pred_ptr.add_offsets(n_offs + 2 * N),
        Some(mask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let ph = T::load(
        pred_ptr.add_offsets(n_offs + 3 * N),
        Some(mask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );

    // Target XYWH.
    let tx = T::load(
        target_ptr.add_offsets(n_offs + 0 * N),
        Some(mask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let ty = T::load(
        target_ptr.add_offsets(n_offs + 1 * N),
        Some(mask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let tw = T::load(
        target_ptr.add_offsets(n_offs + 2 * N),
        Some(mask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let th = T::load(
        target_ptr.add_offsets(n_offs + 3 * N),
        Some(mask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );

    let half = T::full(&[BLOCK_N], D::from_f64(0.5));
    let eps = T::full(&[BLOCK_N], D::from_f64(1e-7));
    let ones = T::full(&[BLOCK_N], D::from_f64(1.0));

    // Pred corners.
    let px1 = px - pw * half;
    let px2 = px + pw * half;
    let py1 = py - ph * half;
    let py2 = py + ph * half;

    // Target corners.
    let tx1 = tx - tw * half;
    let tx2 = tx + tw * half;
    let ty1 = ty - th * half;
    let ty2 = ty + th * half;

    // Intersection.
    let ix1 = T::maximum(px1, tx1);
    let ix2 = T::minimum(px2, tx2);
    let iy1 = T::maximum(py1, ty1);
    let iy2 = T::minimum(py2, ty2);
    let inter_w = T::maximum(ix2 - ix1, zeros);
    let inter_h = T::maximum(iy2 - iy1, zeros);
    let inter = inter_w * inter_h;

    // Union.
    let pred_area = pw * ph;
    let target_area = tw * th;
    let union = pred_area + target_area - inter;

    // IoU.
    let iou = inter / (union + eps);

    // Center distance squared.
    let dx = px - tx;
    let dy = py - ty;
    let d2 = dx * dx + dy * dy;

    // Smallest enclosing box diagonal squared.
    let ex1 = T::minimum(px1, tx1);
    let ex2 = T::maximum(px2, tx2);
    let ey1 = T::minimum(py1, ty1);
    let ey2 = T::maximum(py2, ty2);
    let ecw = ex2 - ex1;
    let ech = ey2 - ey1;
    let c2 = ecw * ecw + ech * ech;

    // Aspect-ratio consistency term (requires atan).
    let atan_t = T::atan(tw / (th + eps));
    let atan_p = T::atan(pw / (ph + eps));
    let diff = atan_t - atan_p;
    // 4 / π²  ≈ 0.405284734
    let pi2_inv4 = T::full(&[BLOCK_N], D::from_f64(0.405_284_73));
    let v = pi2_inv4 * diff * diff;

    // α = v / (1 − IoU + v + ε)
    let alpha = v / (ones - iou + v + eps);

    // Save activations for the backward pass.
    T::store(
        iou_ptr.add_offsets(n_offs),
        iou,
        Some(mask),
        &[],
        None,
        None,
    );
    T::store(v_ptr.add_offsets(n_offs), v, Some(mask), &[], None, None);
    T::store(
        alpha_ptr.add_offsets(n_offs),
        alpha,
        Some(mask),
        &[],
        None,
        None,
    );

    // CIoU loss = 1 − IoU + d²/(c² + ε) + α·v
    let loss = ones - iou + d2 / (c2 + eps) + alpha * v;

    T::store(
        loss_ptr.add_offsets(n_offs),
        loss,
        Some(mask),
        &[],
        None,
        None,
    );
}

/// Fused CIoU loss backward: computes ∂L/∂pred given the upstream gradient
/// and the saved activations from the forward pass.
///
/// Only produces gradients w.r.t. `pred`; target boxes are treated as constants.
///
/// Gradient decomposes into three independent parts:
///
///   (a) IoU term:            ∂(−IoU)/∂pred
///   (b) Center-distance term: ∂(d²/(c²+ε))/∂pred
///   (c) α·v term:            ∂(α·v)/∂(pw,ph)  — zero for (px,py)
///
/// The min/max branching for intersection and enclosing-box corners is
/// re-derived from (pred, target) rather than saved, since pred+target
/// fully determine which branch was active.
///
/// Grid: `cdiv(N, BLOCK_N)` — one CTA per anchor tile.
#[kernel]
pub fn yolo_ciou_loss_backward<T: Triton, D: Float, const BLOCK_N: i32>(
    dy_ptr: InPtr<T::Pointer<D>>,
    pred_ptr: InPtr<T::Pointer<D>>,
    target_ptr: InPtr<T::Pointer<D>>,
    iou_ptr: InPtr<T::Pointer<D>>,
    v_ptr: InPtr<T::Pointer<D>>,
    alpha_ptr: InPtr<T::Pointer<D>>,
    d_pred_ptr: OutPtr<T::Pointer<D>>,
    N: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let n_start = T::program_id(Axis::X) * BLOCK_N;
    let n_offs = T::arange(0, BLOCK_N) + n_start;
    let mask = n_offs.lt(N);
    let zeros = T::zeros::<D>(&[BLOCK_N]);

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

    // Reload pred XYWH.
    let px = T::load(
        pred_ptr.add_offsets(n_offs + 0 * N),
        Some(mask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let py = T::load(
        pred_ptr.add_offsets(n_offs + 1 * N),
        Some(mask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let pw = T::load(
        pred_ptr.add_offsets(n_offs + 2 * N),
        Some(mask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let ph = T::load(
        pred_ptr.add_offsets(n_offs + 3 * N),
        Some(mask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );

    // Reload target XYWH.
    let tx = T::load(
        target_ptr.add_offsets(n_offs + 0 * N),
        Some(mask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let ty = T::load(
        target_ptr.add_offsets(n_offs + 1 * N),
        Some(mask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let tw = T::load(
        target_ptr.add_offsets(n_offs + 2 * N),
        Some(mask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let th = T::load(
        target_ptr.add_offsets(n_offs + 3 * N),
        Some(mask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );

    // Saved activations.
    let iou = T::load(
        iou_ptr.add_offsets(n_offs),
        Some(mask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let v = T::load(
        v_ptr.add_offsets(n_offs),
        Some(mask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let alpha = T::load(
        alpha_ptr.add_offsets(n_offs),
        Some(mask),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );

    let half = T::full(&[BLOCK_N], D::from_f64(0.5));
    let eps = T::full(&[BLOCK_N], D::from_f64(1e-7));
    let ones = T::full(&[BLOCK_N], D::from_f64(1.0));
    let two = T::full(&[BLOCK_N], D::from_f64(2.0));

    // Re-derive corners.
    let px1 = px - pw * half;
    let px2 = px + pw * half;
    let py1 = py - ph * half;
    let py2 = py + ph * half;
    let tx1 = tx - tw * half;
    let tx2 = tx + tw * half;
    let ty1 = ty - th * half;
    let ty2 = ty + th * half;

    // ── Intersection geometry ────────────────────────────────────────────────

    let ix1 = T::maximum(px1, tx1);
    let ix2 = T::minimum(px2, tx2);
    let iy1 = T::maximum(py1, ty1);
    let iy2 = T::minimum(py2, ty2);
    let inter_w = T::maximum(ix2 - ix1, zeros);
    let inter_h = T::maximum(iy2 - iy1, zeros);
    let inter = inter_w * inter_h;
    let union = pw * ph + tw * th - inter;

    let union_eps = union + eps;

    // Boolean masks for min/max branch routing (1.0 if that branch was active).
    let iw_pos = T::gt(inter_w, zeros);
    let px2_wins = T::lt(px2, tx2); // ix2 = px2 (pred right edge tighter)
    let px1_loses = T::gt(px1, tx1); // ix1 = px1 (pred left  edge tighter)
    let py2_wins = T::lt(py2, ty2);
    let py1_loses = T::gt(py1, ty1);
    let ih_pos = T::gt(inter_h, zeros);

    // ∂inter_w/∂px = ±1 gated by iw_pos; ∂inter_w/∂pw = ½·(wins+loses) gated by iw_pos
    let diw_dpx = T::where_(
        iw_pos,
        T::where_(px2_wins, ones, zeros) - T::where_(px1_loses, ones, zeros),
        zeros,
    );
    let dih_dpy = T::where_(
        ih_pos,
        T::where_(py2_wins, ones, zeros) - T::where_(py1_loses, ones, zeros),
        zeros,
    );
    let diw_dpw = T::where_(
        iw_pos,
        half * (T::where_(px2_wins, ones, zeros) + T::where_(px1_loses, ones, zeros)),
        zeros,
    );
    let dih_dph = T::where_(
        ih_pos,
        half * (T::where_(py2_wins, ones, zeros) + T::where_(py1_loses, ones, zeros)),
        zeros,
    );

    let di_dpx = inter_h * diw_dpx;
    let di_dpy = inter_w * dih_dpy;
    let di_dpw = inter_h * diw_dpw;
    let di_dph = inter_w * dih_dph;

    // ∂union/∂pred: union = pw·ph + tw·th − inter
    let du_dpx = zeros - di_dpx;
    let du_dpy = zeros - di_dpy;
    let du_dpw = ph - di_dpw;
    let du_dph = pw - di_dph;

    // ∂IoU/∂pred_i = (∂inter/∂pred_i − iou·∂union/∂pred_i) / (union+ε)
    //
    // The total IoU contribution to ∂loss/∂pred_i also includes a cross-term
    // from ∂α/∂IoU (since α = v/D, D = 1−IoU+v+ε):
    //
    //   ∂loss/∂pred_i|_{IoU} = (v²/D² − 1) · ∂IoU/∂pred_i
    //
    // The (−1) is the direct ∂(−IoU)/∂IoU = −1 term.
    // The (+v²/D²) is the cross-term from ∂(α·v)/∂IoU = v²/D².
    // D is computed below alongside the α·v section; we use d_denom for it.
    let diou_dpx = (di_dpx - iou * du_dpx) / union_eps;
    let diou_dpy = (di_dpy - iou * du_dpy) / union_eps;
    let diou_dpw = (di_dpw - iou * du_dpw) / union_eps;
    let diou_dph = (di_dph - iou * du_dph) / union_eps;

    // ── Enclosing-box diagonal squared ───────────────────────────────────────

    let ex1 = T::minimum(px1, tx1);
    let ex2 = T::maximum(px2, tx2);
    let ey1 = T::minimum(py1, ty1);
    let ey2 = T::maximum(py2, ty2);
    let ecw = ex2 - ex1;
    let ech = ey2 - ey1;
    let c2 = ecw * ecw + ech * ech;

    // Center distance squared.
    let dcx = px - tx;
    let dcy = py - ty;
    let d2 = dcx * dcx + dcy * dcy;

    let c2_eps = c2 + eps;

    // Boolean masks for enclosing-box corner routing.
    let ex2_from_pred = T::gt(px2, tx2); // ex2 = px2
    let ex1_from_pred = T::lt(px1, tx1); // ex1 = px1
    let ey2_from_pred = T::gt(py2, ty2);
    let ey1_from_pred = T::lt(py1, ty1);

    // ∂c²/∂px  = 2·ecw·((ex2_from_pred?1:0) − (ex1_from_pred?1:0))
    // ∂c²/∂pw  = 2·ecw·(½·ex2_from_pred + ½·ex1_from_pred)  = ecw·(ex2+ex1 flags)
    //   [px2 = px+pw/2 → ∂px2/∂pw = ½;  px1 = px−pw/2 → ∂ex1/∂pw = −∂px1/∂pw = +½]
    let dc2_dpx =
        two * ecw * (T::where_(ex2_from_pred, ones, zeros) - T::where_(ex1_from_pred, ones, zeros));
    let dc2_dpy =
        two * ech * (T::where_(ey2_from_pred, ones, zeros) - T::where_(ey1_from_pred, ones, zeros));
    let dc2_dpw =
        ecw * (T::where_(ex2_from_pred, ones, zeros) + T::where_(ex1_from_pred, ones, zeros));
    let dc2_dph =
        ech * (T::where_(ey2_from_pred, ones, zeros) + T::where_(ey1_from_pred, ones, zeros));

    // ∂(d²/(c²+ε))/∂px = 2·(px−tx)/(c²+ε) − d²/(c²+ε)²·∂c²/∂px
    let inv_c2 = ones / c2_eps;
    let d2_over_c2sq = d2 / (c2_eps * c2_eps);
    let d_dist_dpx = two * dcx * inv_c2 - d2_over_c2sq * dc2_dpx;
    let d_dist_dpy = two * dcy * inv_c2 - d2_over_c2sq * dc2_dpy;
    let d_dist_dpw = zeros - d2_over_c2sq * dc2_dpw;
    let d_dist_dph = zeros - d2_over_c2sq * dc2_dph;

    // ── α·v term ─────────────────────────────────────────────────────────────
    // Only pw and ph carry gradients (atan terms depend on w/h only).
    //
    // v = (4/π²)·(atan(tw/(th+ε)) − atan(pw/(ph+ε)))²
    // Let Δ = atan_t − atan_p,  u_p = pw/(ph+ε)
    // ∂v/∂pw = (4/π²)·2Δ·(−1/(1+u_p²))·(1/(ph+ε))
    //        = −(8/π²)·Δ·(ph+ε)/((ph+ε)²+pw²)
    // ∂v/∂ph = (4/π²)·2Δ·(−1/(1+u_p²))·(−pw/(ph+ε)²)
    //        = +(8/π²)·Δ·pw/((ph+ε)²+pw²)
    //
    // ∂(α·v)/∂pw = ∂v/∂pw · [alpha + v·(1−iou+ε)/(1−iou+v+ε)²]
    //            = ∂v/∂pw · [alpha + v·(1−iou+ε)/((1−iou+v+ε)·(1−iou+v+ε))]
    // but  alpha = v/(1−iou+v+ε), so (1−iou+v+ε) = v/alpha  (when alpha≠0)
    // Simpler: let D = 1−iou+v+ε
    //   ∂(α·v)/∂pw = ∂v/∂pw · (alpha + v·(D−v)/D²)
    //              = ∂v/∂pw · (alpha + v·(1−iou+ε)/D²)

    let atan_p = T::atan(pw / (ph + eps));
    let atan_t = T::atan(tw / (th + eps));
    let diff = atan_t - atan_p;

    let ph_eps = ph + eps;
    let denom_uv = ph_eps * ph_eps + pw * pw; // (ph+ε)² + pw²

    // 8/π²  ≈ 0.810569466
    let eight_pi2_inv = T::full(&[BLOCK_N], D::from_f64(0.810_569_46));

    let dv_dpw = zeros - eight_pi2_inv * diff * ph_eps / denom_uv;
    let dv_dph = eight_pi2_inv * diff * pw / denom_uv;

    // D = (1 − iou + v + ε)
    let d_denom = ones - iou + v + eps;
    // ∂(α·v)/∂pw = ∂v/∂pw · (alpha + v·(1−iou+ε)/D²)
    let one_minus_iou_eps = ones - iou + eps;
    let factor = alpha + v * one_minus_iou_eps / (d_denom * d_denom);

    let d_av_dpw = dv_dpw * factor;
    let d_av_dph = dv_dph * factor;

    // ── IoU scale: combines direct ∂(−IoU) and the cross-term ∂(α·v)/∂IoU ────
    // Full IoU contribution: (v²/D² − 1) · ∂IoU/∂pred_i
    let iou_scale = v * v / (d_denom * d_denom) - ones;

    // ── Combine and write ─────────────────────────────────────────────────────

    let g_px = dy * (iou_scale * diou_dpx + d_dist_dpx);
    let g_py = dy * (iou_scale * diou_dpy + d_dist_dpy);
    let g_pw = dy * (iou_scale * diou_dpw + d_dist_dpw + d_av_dpw);
    let g_ph = dy * (iou_scale * diou_dph + d_dist_dph + d_av_dph);

    T::store(
        d_pred_ptr.add_offsets(n_offs + 0 * N),
        g_px,
        Some(mask),
        &[],
        None,
        None,
    );
    T::store(
        d_pred_ptr.add_offsets(n_offs + 1 * N),
        g_py,
        Some(mask),
        &[],
        None,
        None,
    );
    T::store(
        d_pred_ptr.add_offsets(n_offs + 2 * N),
        g_pw,
        Some(mask),
        &[],
        None,
        None,
    );
    T::store(
        d_pred_ptr.add_offsets(n_offs + 3 * N),
        g_ph,
        Some(mask),
        &[],
        None,
        None,
    );
}
