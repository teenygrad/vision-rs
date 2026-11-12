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

use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct SpaceInfo {
    pub id: usize,
    pub bbox: [f64; 4],
    pub occupied: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ParkingLotSnapshot {
    pub lot: String,
    pub timestamp: String,
    pub image_b64: String,
    pub total_spaces: usize,
    pub occupied_ids: Vec<usize>,
    pub spaces: Vec<SpaceInfo>,
    /// Inference timing over every image the server has run the model on so far, across all
    /// lots; absent when the server runs without `--model`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference: Option<InferenceStats>,
    /// What the model found in this image; absent when the server runs without `--model`.
    /// `occupied_ids`/`spaces[].occupied` above are the ground truth from the annotations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detections: Option<Detections>,
}

/// COCO classes that count as a parked vehicle: car, bus, truck.
pub const VEHICLE_CLASSES: &[usize] = &[2, 5, 7];

/// One object the model detected.
#[derive(Clone, Debug, Serialize)]
pub struct Detection {
    /// COCO class id.
    pub class: usize,
    pub score: f32,
    /// `[x, y, w, h]` in image pixels, like [`SpaceInfo::bbox`].
    pub bbox: [f64; 4],
}

impl Detection {
    fn centre(&self) -> (f64, f64) {
        let [x, y, w, h] = self.bbox;
        (x + w / 2.0, y + h / 2.0)
    }

    fn is_vehicle(&self) -> bool {
        VEHICLE_CLASSES.contains(&self.class)
    }
}

/// The model's view of one image: everything it detected, and the spaces that makes occupied.
#[derive(Clone, Debug, Serialize)]
pub struct Detections {
    pub boxes: Vec<LabelledDetection>,
    /// Spaces with a vehicle's centre inside their bounding box.
    pub occupied_ids: Vec<usize>,
}

/// A [`Detection`] plus its class name, so the page needn't carry the COCO label table.
#[derive(Clone, Debug, Serialize)]
pub struct LabelledDetection {
    #[serde(flatten)]
    pub detection: Detection,
    pub label: &'static str,
}

impl Detections {
    /// Marks a space occupied when any vehicle detection's centre falls inside its bounding box.
    pub fn from_boxes(spaces: &[SpaceInfo], boxes: Vec<Detection>) -> Self {
        let occupied_ids = spaces
            .iter()
            .filter(|space| {
                let [sx, sy, sw, sh] = space.bbox;
                boxes.iter().filter(|d| d.is_vehicle()).any(|d| {
                    let (cx, cy) = d.centre();
                    cx >= sx && cx <= sx + sw && cy >= sy && cy <= sy + sh
                })
            })
            .map(|space| space.id)
            .collect();
        let boxes = boxes
            .into_iter()
            .map(|detection| LabelledDetection {
                label: coco_label(detection.class),
                detection,
            })
            .collect();
        Self {
            boxes,
            occupied_ids,
        }
    }
}

/// The COCO-80 class name for `class`, or `"object"` past the end of the table.
pub fn coco_label(class: usize) -> &'static str {
    const LABELS: [&str; 80] = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ];
    LABELS.get(class).copied().unwrap_or("object")
}

/// Per-image inference time statistics, in milliseconds.
#[derive(Clone, Debug, Default, Serialize)]
pub struct InferenceStats {
    /// Number of images timed.
    pub count: u64,
    /// The most recent image's time.
    pub last_ms: f64,
    pub mean_ms: f64,
    /// Sample standard deviation (0 until there are two images).
    pub std_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    /// Welford's running sum of squared deviations from the mean.
    #[serde(skip)]
    m2: f64,
}

impl InferenceStats {
    /// Adds one image's inference time.
    pub fn record(&mut self, ms: f64) {
        self.count += 1;
        self.last_ms = ms;
        if self.count == 1 {
            self.min_ms = ms;
            self.max_ms = ms;
        } else {
            self.min_ms = self.min_ms.min(ms);
            self.max_ms = self.max_ms.max(ms);
        }
        let delta = ms - self.mean_ms;
        self.mean_ms += delta / self.count as f64;
        self.m2 += delta * (ms - self.mean_ms);
        self.std_ms = if self.count > 1 {
            (self.m2 / (self.count - 1) as f64).sqrt()
        } else {
            0.0
        };
    }
}

#[cfg(test)]
mod tests {
    use super::{Detection, Detections, InferenceStats, ParkingLotSnapshot, SpaceInfo};

    #[test]
    fn inference_stats_match_the_two_pass_formulas() {
        let samples = [2100.0, 2050.0, 2180.0, 1990.0, 2120.0];
        let mut stats = InferenceStats::default();
        for s in samples {
            stats.record(s);
        }
        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let var = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1.0);
        assert_eq!(stats.count, 5);
        assert_eq!(stats.last_ms, 2120.0);
        assert_eq!((stats.min_ms, stats.max_ms), (1990.0, 2180.0));
        assert!((stats.mean_ms - mean).abs() < 1e-9);
        assert!((stats.std_ms - var.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn one_sample_has_zero_std() {
        let mut stats = InferenceStats::default();
        stats.record(42.0);
        assert_eq!(
            (stats.mean_ms, stats.std_ms, stats.min_ms, stats.max_ms),
            (42.0, 0.0, 42.0, 42.0)
        );
    }

    fn space(id: usize, bbox: [f64; 4]) -> SpaceInfo {
        SpaceInfo {
            id,
            bbox,
            occupied: false,
        }
    }

    fn detection(class: usize, bbox: [f64; 4]) -> Detection {
        Detection {
            class,
            score: 0.6,
            bbox,
        }
    }

    #[test]
    fn a_space_is_occupied_when_a_vehicle_centre_falls_inside_it() {
        let spaces = [
            space(1, [0.0, 0.0, 10.0, 10.0]),
            space(2, [20.0, 0.0, 10.0, 10.0]),
            space(3, [40.0, 0.0, 10.0, 10.0]),
        ];
        let boxes = [
            // A car centred in space 1, spilling over its edges.
            detection(2, [-5.0, -5.0, 20.0, 20.0]),
            // A truck whose box overlaps space 2 but whose centre is outside it.
            detection(7, [28.0, 0.0, 10.0, 10.0]),
            // A person (not a vehicle) centred in space 3.
            detection(0, [42.0, 2.0, 6.0, 6.0]),
        ];
        let detections = Detections::from_boxes(&spaces, boxes.to_vec());
        assert_eq!(detections.occupied_ids, vec![1]);
        assert_eq!(
            detections.boxes.len(),
            3,
            "every detection is kept for display"
        );
    }

    fn snapshot(detections: Option<Detections>) -> ParkingLotSnapshot {
        ParkingLotSnapshot {
            lot: "UFPR04".into(),
            timestamp: "2012-12-14T09:35:05Z".into(),
            image_b64: String::new(),
            total_spaces: 1,
            occupied_ids: vec![],
            spaces: vec![space(1, [0.0, 0.0, 10.0, 10.0])],
            inference: None,
            detections,
        }
    }

    #[test]
    fn snapshot_carries_the_models_detections() {
        let dets = Detections::from_boxes(
            &[space(1, [0.0, 0.0, 10.0, 10.0])],
            vec![detection(2, [1.0, 2.0, 6.0, 4.0])],
        );
        let json: serde_json::Value =
            serde_json::to_value(snapshot(Some(dets))).expect("serialise snapshot");
        let dets = &json["detections"];
        assert_eq!(dets["occupied_ids"], serde_json::json!([1]));
        let b = &dets["boxes"][0];
        assert_eq!(b["class"], 2);
        assert_eq!(b["label"], "car");
        assert_eq!(b["bbox"], serde_json::json!([1.0, 2.0, 6.0, 4.0]));
        assert!((b["score"].as_f64().unwrap() - 0.6).abs() < 1e-6);
    }

    #[test]
    fn snapshot_without_a_model_has_no_detections() {
        let json = serde_json::to_value(snapshot(None)).expect("serialise snapshot");
        assert!(json.get("detections").is_none(), "{json}");
    }
}
