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

//! Object detection interface.
//!
//! # Example
//!
//! ```rust,no_run
//! use vision_rs::detect::{DetectorConfig, ObjectDetector, Yolo26DetectorConfig};
//! use vision_rs::models::yolo::yolo26::Yolo26Variant;
//!
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! let config = DetectorConfig::Yolo26(Yolo26DetectorConfig::new(
//!     Yolo26Variant::N,
//!     "weights/yolo26n.bin",
//!     vec!["car".into(), "truck".into(), "person".into()],
//! ));
//!
//! let detector = ObjectDetector::new(config)?;
//! let image_bytes = std::fs::read("frame.jpg")?;
//! let detections = detector.detect(&image_bytes).await?;
//!
//! for d in &detections {
//!     println!("{} {:.2} {:?}", d.class, d.confidence, d.bbox);
//! }
//! # Ok(())
//! # }
//! ```

use std::path::PathBuf;

use crate::models::yolo::yolo26::Yolo26Variant;

// ── Detection result ──────────────────────────────────────────────────────────

/// A single detected object.
#[derive(Clone, Debug)]
pub struct Detection {
    /// Bounding box as `[cx, cy, w, h]` normalised to `[0, 1]`.
    pub bbox: [f32; 4],
    /// Class label resolved from the detector's `class_names` list.
    pub class: String,
    /// Confidence score in `[0, 1]`.
    pub confidence: f32,
}

// ── YOLO26 config ─────────────────────────────────────────────────────────────

/// Configuration for a YOLO26-backed [`ObjectDetector`].
#[derive(Clone, Debug)]
pub struct Yolo26DetectorConfig {
    /// Which YOLO26 model size to use (N/S/M/L/X).
    pub variant: Yolo26Variant,
    /// Path to the model weights file.
    pub weights: PathBuf,
    /// Class labels, indexed by the model's output class index.
    pub class_names: Vec<String>,
    /// Minimum confidence to retain a detection. Default: `0.25`.
    pub conf_threshold: f32,
    /// IoU threshold for non-maximum suppression. Default: `0.45`.
    pub nms_iou_threshold: f32,
    /// Square input resolution the model was trained at (e.g. `640`).
    pub img_size: usize,
}

impl Yolo26DetectorConfig {
    /// Creates a config with default `conf_threshold` (`0.25`), `nms_iou_threshold`
    /// (`0.45`), and `img_size` (`640`).
    pub fn new(
        variant: Yolo26Variant,
        weights: impl Into<PathBuf>,
        class_names: Vec<String>,
    ) -> Self {
        Self {
            variant,
            weights: weights.into(),
            class_names,
            conf_threshold: 0.25,
            nms_iou_threshold: 0.45,
            img_size: 640,
        }
    }
}

// ── Top-level config enum ─────────────────────────────────────────────────────

/// Detector configuration, tagged by model family.
#[derive(Clone, Debug)]
pub enum DetectorConfig {
    /// Use a YOLO26 model.
    Yolo26(Yolo26DetectorConfig),
}

// ── ObjectDetector ────────────────────────────────────────────────────────────

/// A model-agnostic object detector, dispatching to the configured model family.
pub struct ObjectDetector {
    inner: DetectorInner,
}

impl ObjectDetector {
    /// Build a detector from the given config.
    pub fn new(config: DetectorConfig) -> anyhow::Result<Self> {
        let inner = match config {
            DetectorConfig::Yolo26(cfg) => DetectorInner::Yolo26(Yolo26Detector { config: cfg }),
        };
        Ok(Self { inner })
    }

    /// Run inference on raw JPEG or PNG `image_bytes`.
    ///
    /// Returns every detection that clears the config's `conf_threshold`,
    /// after non-maximum suppression.
    pub async fn detect(&self, image_bytes: &[u8]) -> anyhow::Result<Vec<Detection>> {
        match &self.inner {
            DetectorInner::Yolo26(d) => d.detect(image_bytes).await,
        }
    }
}

// ── Private per-model implementations ────────────────────────────────────────

enum DetectorInner {
    Yolo26(Yolo26Detector),
}

struct Yolo26Detector {
    config: Yolo26DetectorConfig,
}

impl Yolo26Detector {
    async fn detect(&self, _image_bytes: &[u8]) -> anyhow::Result<Vec<Detection>> {
        let _ = &self.config;
        todo!("YOLO26 inference")
    }
}
