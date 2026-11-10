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

//! Anchor grid generation for YOLO detection heads.
//!
//! YOLO26 uses three FPN levels with strides [8, 16, 32].  For each level,
//! anchor centres are placed at the centre of every grid cell.

/// Flattened anchor descriptors for all FPN levels.
///
/// Anchors are ordered level-by-level (stride 8 first, then 16, then 32),
/// and within each level in row-major order (y, x).
pub struct AnchorGrid {
    /// Anchor centre x in input-image pixel coordinates. Shape: `[A]`.
    pub cx: Vec<f32>,
    /// Anchor centre y in input-image pixel coordinates. Shape: `[A]`.
    pub cy: Vec<f32>,
    /// Stride (scale factor from feature to input). Shape: `[A]`.
    pub strides: Vec<f32>,
    /// Total number of anchors A.
    pub n_anchors: usize,
}

impl AnchorGrid {
    /// Build the anchor grid for YOLO26 given the input image dimensions.
    ///
    /// `img_h` and `img_w` must be divisible by 32.
    pub fn yolo26(img_h: usize, img_w: usize) -> Self {
        Self::new(img_h, img_w, &[8, 16, 32])
    }

    /// Build the anchor grid for arbitrary strides.
    pub fn new(img_h: usize, img_w: usize, strides: &[usize]) -> Self {
        let total: usize = strides.iter().map(|&s| (img_h / s) * (img_w / s)).sum();
        let mut cx = Vec::with_capacity(total);
        let mut cy = Vec::with_capacity(total);
        let mut stride_v = Vec::with_capacity(total);

        for &s in strides {
            let fh = img_h / s;
            let fw = img_w / s;
            for gy in 0..fh {
                for gx in 0..fw {
                    cx.push((gx as f32 + 0.5) * s as f32);
                    cy.push((gy as f32 + 0.5) * s as f32);
                    stride_v.push(s as f32);
                }
            }
        }

        AnchorGrid {
            n_anchors: total,
            cx,
            cy,
            strides: stride_v,
        }
    }

    /// Decode raw LTRB predictions into absolute XYWH boxes.
    ///
    /// Input `ltrb`: `[4, A]` channels-first — (l, t, r, b) distances.
    /// Output: `[4, A]` channels-first — (cx, cy, w, h) in pixel coords.
    pub fn decode_ltrb_to_xywh(&self, ltrb: &[f32]) -> Vec<f32> {
        let a = self.n_anchors;
        assert_eq!(ltrb.len(), 4 * a);
        let mut out = vec![0.0f32; 4 * a];
        for i in 0..a {
            let l = ltrb[i];
            let t = ltrb[a + i];
            let r = ltrb[2 * a + i];
            let b = ltrb[3 * a + i];
            let s = self.strides[i];
            out[i] = self.cx[i] + s * (r - l) * 0.5; // cx
            out[a + i] = self.cy[i] + s * (b - t) * 0.5; // cy
            out[2 * a + i] = s * (l + r); // w
            out[3 * a + i] = s * (t + b); // h
        }
        out
    }

    /// Backward through the LTRB → XYWH decode.
    ///
    /// `d_xywh`: `[4, A]` gradient w.r.t. decoded (cx, cy, w, h).
    /// Returns: `[4, A]` gradient w.r.t. raw (l, t, r, b) predictions.
    pub fn decode_backward(&self, d_xywh: &[f32]) -> Vec<f32> {
        let a = self.n_anchors;
        assert_eq!(d_xywh.len(), 4 * a);
        let mut d_ltrb = vec![0.0f32; 4 * a];
        for i in 0..a {
            let d_cx = d_xywh[i];
            let d_cy = d_xywh[a + i];
            let d_w = d_xywh[2 * a + i];
            let d_h = d_xywh[3 * a + i];
            let s = self.strides[i];
            d_ltrb[i] = d_cx * (-s * 0.5) + d_w * s; // d_l
            d_ltrb[a + i] = d_cy * (-s * 0.5) + d_h * s; // d_t
            d_ltrb[2 * a + i] = d_cx * (s * 0.5) + d_w * s; // d_r
            d_ltrb[3 * a + i] = d_cy * (s * 0.5) + d_h * s; // d_b
        }
        d_ltrb
    }
}
