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
    use super::InferenceStats;

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
}
