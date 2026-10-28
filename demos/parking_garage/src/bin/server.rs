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

/*
 * Usage (no inference):
 *   parking-garage-server [dataset_root] [--port N]
 *
 * Usage (with YOLO inference, requires --features cuda):
 *   MODELS_CACHE_DIR=... TEENYC_PATH=... TEENYC_CACHE_DIR=... \
 *     parking-garage-server [dataset_root] [--port N] --model ultralytics/yolo26n
 *
 * Weights are downloaded automatically on first run (pre-converted safetensors from
 * https://huggingface.co/datasets/teenygrad/ultralytics-yolo26) and cached under
 * MODELS_CACHE_DIR.
 *
 * dataset_root defaults to $DATASETS_CACHE_DIR/PKLot/PKLot (falling back to
 * $HOME/.cache/vision-rs/datasets/PKLot/PKLot, matching .env.dev's own default, if
 * DATASETS_CACHE_DIR isn't set). If it doesn't exist yet, PKLot.tar.gz is downloaded (with
 * checksum verification and resumable retries) from our HF mirror
 * (https://huggingface.co/datasets/teenygrad/pklot) and extracted automatically — see
 * ensure_dataset/PKLOT_URL below; CC BY 4.0, ~4.6GB.
 */

use anyhow::{Context, Result};
use axum::{
    Router,
    extract::{
        State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    response::IntoResponse,
    routing::get,
};
use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use dotenv::dotenv;
use futures_util::{SinkExt, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use parking_garage::{ParkingLotSnapshot, SpaceInfo};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use tokio::net::TcpListener;
use tokio::sync::broadcast;
use tokio::time::{Duration, interval};

const TICK_SECS: u64 = 2;
const BROADCAST_CAPACITY: usize = 64;

// ── In-memory lot model ────────────────────────────────────────────────────

struct Space {
    id: usize,
    bbox: [f64; 4], // [x, y, w, h] axis-aligned, derived from XML contour
    occupied: bool,
}

struct ParkingImage {
    filename: String, // basename only, used for timestamp + display
    path: PathBuf,    // full path to .jpg; .xml lives at the same stem
}

struct Lot {
    name: String,
    images: Vec<ParkingImage>,
    cursor: usize,
}

// ── Axum state ────────────────────────────────────────────────────────────

#[derive(Clone)]
struct AppState {
    tx: broadcast::Sender<String>,
}

// ── PKLot XML annotation parsing ───────────────────────────────────────────

// Parse a PKLot XML annotation file into a Vec of Spaces.
// Bounding box is the axis-aligned envelope of the contour corner points.
fn parse_xml_spaces(xml: &str) -> Result<Vec<Space>> {
    let doc = roxmltree::Document::parse(xml).context("parsing XML")?;
    let mut spaces = Vec::new();

    for node in doc.root_element().children() {
        if !node.has_tag_name("space") {
            continue;
        }
        let id = node.attribute("id").and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
        let occupied = node.attribute("occupied").map(|v| v != "0").unwrap_or(false);

        // Prefer contour (4 rotated corners) for exact AABB.
        // Some dataset files use <Point> (capital P) — match case-insensitively.
        let mut xs: Vec<f64> = Vec::new();
        let mut ys: Vec<f64> = Vec::new();

        if let Some(contour) = node.children().find(|n| n.has_tag_name("contour")) {
            for pt in contour.children().filter(|n| {
                n.tag_name().name().eq_ignore_ascii_case("point")
            }) {
                if let (Some(x), Some(y)) = (
                    pt.attribute("x").and_then(|v| v.parse().ok()),
                    pt.attribute("y").and_then(|v| v.parse().ok()),
                ) {
                    xs.push(x);
                    ys.push(y);
                }
            }
        }

        // Fallback: derive AABB from rotatedRect using the angle so the box
        // is correct even for heavily-angled spaces (side rows of the lot).
        if xs.is_empty() {
            if let Some(rr) = node.children().find(|n| n.has_tag_name("rotatedRect")) {
                let cx = rr.children().find(|n| n.has_tag_name("center"))
                    .and_then(|c| c.attribute("x").and_then(|v| v.parse::<f64>().ok()))
                    .unwrap_or(0.0);
                let cy = rr.children().find(|n| n.has_tag_name("center"))
                    .and_then(|c| c.attribute("y").and_then(|v| v.parse::<f64>().ok()))
                    .unwrap_or(0.0);
                let w = rr.children().find(|n| n.has_tag_name("size"))
                    .and_then(|s| s.attribute("w").and_then(|v| v.parse::<f64>().ok()))
                    .unwrap_or(0.0);
                let h = rr.children().find(|n| n.has_tag_name("size"))
                    .and_then(|s| s.attribute("h").and_then(|v| v.parse::<f64>().ok()))
                    .unwrap_or(0.0);
                let angle_deg = rr.children().find(|n| n.has_tag_name("angle"))
                    .and_then(|a| a.attribute("d").and_then(|v| v.parse::<f64>().ok()))
                    .unwrap_or(0.0);
                let (sin_a, cos_a) = angle_deg.to_radians().sin_cos();
                // AABB half-extents of a rotated rectangle.
                let hw = (w * cos_a.abs() + h * sin_a.abs()) / 2.0;
                let hh = (w * sin_a.abs() + h * cos_a.abs()) / 2.0;
                xs = vec![cx - hw, cx + hw];
                ys = vec![cy - hh, cy + hh];
            }
        }

        if xs.is_empty() {
            continue;
        }

        let x_min = xs.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y_min = ys.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        spaces.push(Space {
            id,
            bbox: [x_min, y_min, x_max - x_min, y_max - y_min],
            occupied,
        });
    }

    Ok(spaces)
}

// ── Dataset loading ────────────────────────────────────────────────────────

// Return subdirectories of `dir`, sorted alphabetically.
fn sorted_subdirs(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .map(|e| e.path())
        .collect();
    dirs.sort();
    Ok(dirs)
}

// Load a single PKLot lot from its root directory.
// Structure: {lot}/{Weather}/{YYYY-MM-DD}/{timestamp}.jpg + .xml
fn load_lot(lot_dir: &Path) -> Result<Lot> {
    let name = lot_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_owned();

    let mut images: Vec<ParkingImage> = Vec::new();

    for weather_dir in sorted_subdirs(lot_dir)? {
        for date_dir in sorted_subdirs(&weather_dir)? {
            let mut entries: Vec<_> = std::fs::read_dir(&date_dir)
                .with_context(|| format!("reading {}", date_dir.display()))?
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path().extension().and_then(|x| x.to_str()) == Some("jpg")
                })
                .collect();
            entries.sort_by_key(|e| e.file_name());

            for entry in entries {
                let jpg_path = entry.path();
                if jpg_path.with_extension("xml").exists() {
                    images.push(ParkingImage {
                        filename: entry.file_name().to_string_lossy().into_owned(),
                        path: jpg_path,
                    });
                }
            }
        }
    }

    anyhow::ensure!(!images.is_empty(), "no images found under {}", lot_dir.display());

    println!("lot {:>8}: {} images", name, images.len());
    Ok(Lot { name, images, cursor: 0 })
}

/// Source archive for the PKLot dataset: an unmodified mirror of the original UFPR VRI lab
/// archive, hosted on our own HF org since the original host (inf.ufpr.br) isn't always
/// reachable — see https://huggingface.co/datasets/teenygrad/pklot. Released under CC BY 4.0 —
/// see Almeida et al., "PKLot – A robust dataset for parking lot classification", Expert
/// Systems with Applications, 2015. Attribute the paper if you redistribute this data.
const PKLOT_URL: &str = "https://huggingface.co/datasets/teenygrad/pklot/resolve/main/PKLot.tar.gz";

/// Expected sha256 of `PKLot.tar.gz`, pinned to what we uploaded to the HF mirror above —
/// this is a fixed, immutable artifact, not a moving target. Guards against a corrupt/partial
/// download (e.g. a botched resume) being extracted silently.
const PKLOT_SHA256: &str = "e89bbc1dc735298c478688d50c7a682fb3b0076a87b6634923132709f2d2fa9b";

/// Exact size of `PKLot.tar.gz`, pinned alongside its checksum — drives the progress bar's
/// total (we poll the partial file's size on disk rather than parsing curl's own output).
const PKLOT_SIZE_BYTES: u64 = 4_898_276_304;

/// A ~4.6GB download over a real network will drop sometimes — worth a few attempts before
/// giving up.
const PKLOT_MAX_ATTEMPTS: u32 = 5;

fn sha256_hex(path: &Path) -> Result<String> {
    let output = std::process::Command::new("sha256sum")
        .arg(path)
        .output()
        .context("running sha256sum")?;
    anyhow::ensure!(output.status.success(), "sha256sum exited with {}", output.status);
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .split_whitespace()
        .next()
        .map(str::to_owned)
        .ok_or_else(|| anyhow::anyhow!("unexpected sha256sum output: {stdout}"))
}

/// Downloads and extracts `PKLot.tar.gz` if `root` doesn't already exist.
///
/// The archive's own top-level directory (named `PKLot`) contains two siblings:
/// `PKLot/PKLot/{PUCPR,UFPR04,UFPR05}/...` (the full-frame images + XML annotations this demo
/// reads) and `PKLot/PKLotSegmented/...` (pre-cropped per-space images, unused here). So
/// extracting the archive two directories above `root` reproduces `root` exactly — true of
/// both the `$DATASETS_CACHE_DIR/PKLot/PKLot` default and the classic `.../PKLot/PKLot`
/// layout, which is why `root` must end in `PKLot/PKLot`.
///
/// Shells out to `curl`/`tar`/`sha256sum` rather than pulling in an HTTP client crate, so this
/// works with or without `--features cuda`.
fn ensure_dataset(root: &Path) -> Result<()> {
    if root.exists() {
        return Ok(());
    }

    let extract_into = root
        .parent()
        .and_then(Path::parent)
        .filter(|p| !p.as_os_str().is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "dataset root {} must be nested as \"<parent>/PKLot/PKLot\" for auto-download \
                 to know where to extract PKLot.tar.gz",
                root.display()
            )
        })?;
    std::fs::create_dir_all(extract_into).with_context(|| format!("creating {}", extract_into.display()))?;

    let tarball = extract_into.join("PKLot.tar.gz");
    println!("dataset not found at {}; downloading from {PKLOT_URL} (CC BY 4.0, ~4.6GB) …", root.display());

    let pb = ProgressBar::new(PKLOT_SIZE_BYTES);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("  [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, eta {eta})")?
            .progress_chars("█▉▊▋▌▍▎▏  "),
    );

    let mut verified = false;
    for attempt in 1..=PKLOT_MAX_ATTEMPTS {
        if attempt > 1 {
            pb.println(format!("retrying download (attempt {attempt}/{PKLOT_MAX_ATTEMPTS}) …"));
        }

        // `-C -` resumes from wherever a previous attempt left off (curl auto-detects the
        // partial file's size and issues a Range request) instead of restarting a multi-GB
        // download from zero; `--retry`/`--retry-all-errors` additionally retries transient
        // failures (timeouts, connection resets, 5xx) within a single invocation. `-s` silences
        // curl's own progress meter since we're driving our own bar below (polling the partial
        // file's size on disk, which also naturally reflects a resumed attempt's starting point).
        let mut child = std::process::Command::new("curl")
            .args(["-sfL", "--retry", "5", "--retry-delay", "5", "--retry-all-errors", "-C", "-", "-o"])
            .arg(&tarball)
            .arg(PKLOT_URL)
            .spawn()
            .context("spawning curl to download PKLot.tar.gz")?;

        let status = loop {
            if let Ok(meta) = std::fs::metadata(&tarball) {
                pb.set_position(meta.len());
            }
            if let Some(status) = child.try_wait().context("waiting for curl")? {
                break status;
            }
            std::thread::sleep(std::time::Duration::from_millis(300));
        };

        if !status.success() {
            pb.println(format!("curl exited with {status}"));
            continue;
        }

        match sha256_hex(&tarball) {
            Ok(hash) if hash == PKLOT_SHA256 => {
                verified = true;
                break;
            }
            Ok(hash) => {
                pb.println(format!(
                    "checksum mismatch (got {hash}, expected {PKLOT_SHA256}) — download is \
                     corrupt (possibly a bad resume); discarding and restarting from scratch"
                ));
                let _ = std::fs::remove_file(&tarball);
                pb.set_position(0);
            }
            Err(e) => {
                pb.println(format!("failed to checksum downloaded file: {e:#}"));
                let _ = std::fs::remove_file(&tarball);
                pb.set_position(0);
            }
        }
    }
    if verified {
        pb.finish_with_message("PKLot.tar.gz downloaded");
    } else {
        pb.finish_and_clear();
    }
    anyhow::ensure!(
        verified,
        "failed to download a valid PKLot.tar.gz from {PKLOT_URL} after {PKLOT_MAX_ATTEMPTS} attempts"
    );

    println!("extracting PKLot.tar.gz into {} …", extract_into.display());
    let status = std::process::Command::new("tar")
        .args(["-xzf"])
        .arg(&tarball)
        .args(["-C"])
        .arg(extract_into)
        .status()
        .context("running tar to extract PKLot.tar.gz")?;
    anyhow::ensure!(status.success(), "tar failed to extract PKLot.tar.gz");

    let _ = std::fs::remove_file(&tarball);

    anyhow::ensure!(
        root.exists(),
        "expected {} to exist after extracting PKLot.tar.gz — archive layout may have changed",
        root.display()
    );
    Ok(())
}

fn scan_lots(root: &Path) -> Result<Vec<Lot>> {
    let mut lots = Vec::new();
    for lot_dir in sorted_subdirs(root)? {
        match load_lot(&lot_dir) {
            Ok(lot) => lots.push(lot),
            Err(e) => eprintln!("skipping {}: {e:#}", lot_dir.display()),
        }
    }
    if lots.is_empty() {
        anyhow::bail!("no lots found under {}", root.display());
    }
    Ok(lots)
}

// ── Helpers ────────────────────────────────────────────────────────────────

// "2012-09-11_15_16_58.jpg" → "2012-09-11T15:16:58Z"
fn parse_timestamp(filename: &str) -> String {
    let mut parts = filename.splitn(5, '_');
    let date = parts.next().unwrap_or("");
    let hh = parts.next().unwrap_or("00");
    let mm = parts.next().unwrap_or("00");
    let ss = parts.next().unwrap_or("00");
    format!("{date}T{hh}:{mm}:{ss}Z")
}

// ── CUDA inference (requires --features cuda) ──────────────────────────────

#[cfg(feature = "cuda")]
const VEHICLE_CLASSES: &[usize] = &[2, 5, 7]; // COCO: car, bus, truck


// Letterbox-resize image to img_size × img_size, normalise to [0,1], return NCHW f32.
#[cfg(feature = "cuda")]
fn preprocess_image_raw(img: &image::RgbImage, img_size: usize) -> Vec<f32> {
    let (orig_w, orig_h) = (img.width() as usize, img.height() as usize);
    let scale = img_size as f64 / orig_w.max(orig_h) as f64;
    let new_w = (orig_w as f64 * scale).round() as u32;
    let new_h = (orig_h as f64 * scale).round() as u32;
    let resized = image::imageops::resize(img, new_w, new_h, image::imageops::FilterType::Triangle);
    let pad_x = (img_size - new_w as usize) / 2;
    let pad_y = (img_size - new_h as usize) / 2;
    let s = img_size;
    let plane = s * s;
    let mut out = vec![114.0f32 / 255.0; 3 * s * s];
    for py in 0..new_h as usize {
        for px in 0..new_w as usize {
            let p = resized.get_pixel(px as u32, py as u32);
            let idx = (pad_y + py) * s + (pad_x + px);
            out[idx] = p[0] as f32 / 255.0;
            out[plane + idx] = p[1] as f32 / 255.0;
            out[2 * plane + idx] = p[2] as f32 / 255.0;
        }
    }
    out
}

// IoU of two CxCyWH normalised boxes.
#[cfg(feature = "cuda")]
fn box_iou_cxcywh(a: [f32; 4], b: [f32; 4]) -> f32 {
    let ax1 = a[0] - a[2] * 0.5; let ax2 = a[0] + a[2] * 0.5;
    let ay1 = a[1] - a[3] * 0.5; let ay2 = a[1] + a[3] * 0.5;
    let bx1 = b[0] - b[2] * 0.5; let bx2 = b[0] + b[2] * 0.5;
    let by1 = b[1] - b[3] * 0.5; let by2 = b[1] + b[3] * 0.5;
    let inter = (ax2.min(bx2) - ax1.max(bx1)).max(0.0)
              * (ay2.min(by2) - ay1.max(by1)).max(0.0);
    let union = a[2] * a[3] + b[2] * b[3] - inter;
    inter / (union + 1e-7)
}

/// Base URL for pre-converted YOLO26 safetensors weights on Hugging Face.
#[cfg(feature = "cuda")]
const HF_YOLO26_BASE_URL: &str =
    "https://huggingface.co/datasets/teenygrad/ultralytics-yolo26/resolve/main/ultralytics/yolo26";

/// Download a single file from `url` to `dest`.
#[cfg(feature = "cuda")]
async fn download_weights(url: &str, dest: &Path) -> Result<()> {
    use futures_util::StreamExt;
    use tokio::io::AsyncWriteExt;

    let resp = reqwest::Client::builder()
        .user_agent("vision-rs-parking-garage-demo")
        .build()?
        .get(url)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?
        .error_for_status()
        .with_context(|| format!("HTTP error for {url}"))?;

    let mut file = tokio::fs::File::create(dest).await?;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("reading response chunk")?;
        file.write_all(&chunk).await.context("writing to disk")?;
    }
    Ok(())
}

// Load safetensors weights into a compiled CUDA model.
// Handles BatchNorm folding required by Conv2dBnSilu fused nodes.
#[cfg(feature = "cuda")]
fn load_weights_for_model(
    model: &mut teeny_cuda::model::LoadedModel,
    path: &Path,
) -> Result<()> {
    use teeny_data::safetensors::SafeTensors;

    let st = SafeTensors::from_pretrained(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let tensors = st.tensors().context("deserialising safetensors header")?;

    let named_params: Vec<(String, usize, usize)> = model.param_info_named().collect();
    if named_params.is_empty() {
        println!("warning: 0 named params — optimise() may have dropped node names; weights not loaded");
        return Ok(());
    }

    let load_f32 = |key: &str| -> Result<Vec<f32>> {
        let tv = tensors.tensor(key)
            .map_err(|_| anyhow::anyhow!("key '{}' not found", key))?;
        let bytes = tv.data();
        anyhow::ensure!(bytes.len() % 4 == 0, "tensor '{}' not f32", key);
        Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    };

    let mut loaded = 0usize;
    let mut missing: Vec<String> = Vec::new();

    for (key, node_idx, param_idx) in &named_params {
        if key.ends_with(".bn_scale") {
            let prefix = &key[..key.len() - ".bn_scale".len()];
            match (load_f32(&format!("{prefix}.bn.weight")), load_f32(&format!("{prefix}.bn.running_var"))) {
                (Ok(gamma), Ok(var)) => {
                    let eps = 1e-3f32;
                    let v: Vec<f32> = gamma.iter().zip(var.iter()).map(|(&g, &v)| g / (v + eps).sqrt()).collect();
                    model.load_param_f32(*node_idx, *param_idx, &v)
                        .with_context(|| format!("uploading bn_scale '{key}'"))?;
                    loaded += 1;
                }
                _ => missing.push(key.clone()),
            }
            continue;
        }

        if key.ends_with(".bn_shift") {
            let prefix = &key[..key.len() - ".bn_shift".len()];
            match (
                load_f32(&format!("{prefix}.bn.bias")),
                load_f32(&format!("{prefix}.bn.weight")),
                load_f32(&format!("{prefix}.bn.running_mean")),
                load_f32(&format!("{prefix}.bn.running_var")),
            ) {
                (Ok(beta), Ok(gamma), Ok(mean), Ok(var)) => {
                    let eps = 1e-3f32;
                    let v: Vec<f32> = beta.iter().zip(gamma.iter()).zip(mean.iter()).zip(var.iter())
                        .map(|(((&b, &g), &m), &vv)| b - (g / (vv + eps).sqrt()) * m)
                        .collect();
                    model.load_param_f32(*node_idx, *param_idx, &v)
                        .with_context(|| format!("uploading bn_shift '{key}'"))?;
                    loaded += 1;
                }
                _ => missing.push(key.clone()),
            }
            continue;
        }

        let tv_result = match tensors.tensor(key.as_str()) {
            ok @ Ok(_) => ok,
            Err(_) if key.ends_with(".weight") => {
                let prefix = &key[..key.len() - ".weight".len()];
                tensors.tensor(&format!("{prefix}.conv.weight"))
            }
            err => err,
        };
        match tv_result {
            Ok(tv) => {
                let bytes = tv.data();
                anyhow::ensure!(bytes.len() % 4 == 0, "tensor '{key}': not f32");
                let data: Vec<f32> = bytes.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                model.load_param_f32(*node_idx, *param_idx, &data)
                    .with_context(|| format!("uploading '{key}'"))?;
                loaded += 1;
            }
            Err(_) => missing.push(key.clone()),
        }
    }

    if !missing.is_empty() {
        println!("warning: {}/{} params not found (showing first 5):", missing.len(), named_params.len());
        for k in missing.iter().take(5) { println!("  missing: {k}"); }
    }
    println!("loaded {loaded}/{} named params from {}", named_params.len(),
        path.file_name().unwrap_or_default().to_string_lossy());
    Ok(())
}

// Build and return an inference closure for the given model spec.
// The closure takes an image path and returns COCO detections:
//   Vec<(class_id, score, [cx_norm, cy_norm, w_norm, h_norm])>
#[cfg(feature = "cuda")]
fn build_infer_fn(
    model_spec: &str,
    img_size: usize,
) -> Result<Box<dyn FnMut(&Path) -> Result<Vec<(usize, f32, [f32; 4])>>>> {
    use teeny_compiler::compiler::backend::llvm::compiler::LlvmCompiler;
    use teeny_core::{graph::{DtypeRepr, SymTensor}, model::LoweringMode};
    use teeny_cuda::{compiler::{graph::CudaGraphCompiler, target::Target}, model::TensorRef};
    use teeny_test::cuda as testing;
    use teeny_kernels::graph::{Anduin, TritonLowering};
    use vision_rs::models::yolo::{
        loss::anchor::AnchorGrid,
        yolo26::{Yolo26Variant, blocks::detect::DetectHead, yolo26},
    };

    let name = Path::new(model_spec).file_name().and_then(|n| n.to_str()).unwrap_or(model_spec);
    let (nc, variant) = if name.ends_with("xl") {
        (80usize, Yolo26Variant::XL)
    } else {
        match name.chars().last() {
            Some('n') => (80, Yolo26Variant::N),
            Some('s') => (80, Yolo26Variant::S),
            Some('m') => (80, Yolo26Variant::M),
            Some('l') => (80, Yolo26Variant::L),
            _ => anyhow::bail!("cannot infer variant from '{}'; expected suffix n/s/m/l/xl", name),
        }
    };

    let models_dir: PathBuf = std::env::var("MODELS_CACHE_DIR")
        .context("MODELS_CACHE_DIR not set")?
        .into();
    let model_dir = models_dir.join(model_spec);
    let st_path = model_dir.join(format!("{name}.safetensors"));
    if !st_path.exists() {
        std::fs::create_dir_all(&model_dir)
            .with_context(|| format!("creating {}", model_dir.display()))?;
        let url = format!("{HF_YOLO26_BASE_URL}/{name}.safetensors");
        println!("downloading {name}.safetensors from Hugging Face …");
        // build_infer_fn runs synchronously inside the #[tokio::main] runtime
        // already driving main(), so block_on a nested runtime here instead
        // of building a second one (which panics: "Cannot start a runtime
        // from within a runtime").
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(download_weights(&url, &st_path))
        })?;
    }

    let env = testing::setup_cuda_env()?;
    let target = Target::new(env.capability);
    let device = &env.device;

    let teenyc_path = std::env::var("TEENYC_PATH").unwrap_or_else(|_| "teenyc".into());
    let kern_cache = teeny_compiler::compiler::default_cache_dir();

    println!("compiling YOLO26{} (nc={nc}, {img_size}×{img_size}) — first run builds kernel cache …",
        format!("{variant:?}").to_uppercase());

    let (input_sym, _graph_rc) = SymTensor::input(
        DtypeRepr::F32,
        vec![None, Some(3), Some(img_size), Some(img_size)],
    );
    let out = yolo26::<f32>(nc, &variant, DetectHead::OneToOne)(input_sym);
    let graph_rc = out.boxes.graph.clone();
    // examples/yolo26.rs's vetted (mAP-validated) inference path compiles the
    // *optimised* graph and loads weights via the same named-param lookup
    // afterward without issue, so do the same here (faster + fewer DAG nodes).
    let graph_to_compile = graph_rc.borrow().clone();

    let compiler = LlvmCompiler::new(teenyc_path, kern_cache)?;
    let graph_cmp = CudaGraphCompiler::new(compiler);
    let lowering = TritonLowering::new().with_optimizer(Anduin);
    let cuda_model =
        graph_cmp.compile_model(&graph_to_compile, &lowering, &target, LoweringMode::Inference, false)?;
    println!("compiled {} DAG nodes", cuda_model.dag.len());

    let mut model = cuda_model.load(device, 1)?;
    println!("loading weights from {} …", st_path.display());
    load_weights_for_model(&mut model, &st_path)?;
    println!("model ready\n");

    let grid = AnchorGrid::yolo26(img_size, img_size);
    let a = grid.n_anchors;
    let a_per_scale: Vec<usize> = [8usize, 16, 32].iter().map(|&s| (img_size / s).pow(2)).collect();
    let box_off: Vec<usize> = {
        let mut o = vec![0usize]; for &n in &a_per_scale { o.push(o.last().unwrap() + 4 * n); } o
    };
    let score_off: Vec<usize> = {
        let mut o = vec![0usize]; for &n in &a_per_scale { o.push(o.last().unwrap() + nc * n); } o
    };
    let anchor_base: Vec<usize> = {
        let mut o = vec![0usize]; for &n in &a_per_scale { o.push(o.last().unwrap() + n); } o
    };
    let anchor_scale: Vec<(usize, usize)> = a_per_scale.iter().enumerate()
        .flat_map(|(si, &a_s)| (0..a_s).map(move |j| (si, j))).collect();

    let terminals = model.terminal_node_indices_sorted_by_size();
    anyhow::ensure!(terminals.len() >= 2, "model must have 2 terminal nodes");
    let (boxes_tidx, scores_tidx) = (terminals[0], terminals[1]);

    let grid_cx = grid.cx;
    let grid_cy = grid.cy;
    let grid_strides = grid.strides;

    let graph_model = model.capture_graph(device, 1, &[vec![1, 3, img_size, img_size]], &[boxes_tidx, scores_tidx])?;

    let f = move |path: &Path| -> Result<Vec<(usize, f32, [f32; 4])>> {
        let _ = (&env, &model);
        let img = image::open(path).with_context(|| format!("opening {:?}", path))?.to_rgb8();
        let (orig_w, orig_h) = (img.width() as usize, img.height() as usize);
        let pixels = preprocess_image_raw(&img, img_size);

        let outputs = graph_model.run(&[pixels.as_slice()])?;
        let ltrb = &outputs[0];
        let logits = &outputs[1];

        // Decode LTRB → CxCyWH in letterbox-pixel coords.
        let mut xywh = vec![0.0f32; 4 * a];
        for (si, &a_s) in a_per_scale.iter().enumerate() {
            let bb = box_off[si];
            let ab = anchor_base[si];
            for j in 0..a_s {
                let (l, t, r, b) = (ltrb[bb+j], ltrb[bb+a_s+j], ltrb[bb+2*a_s+j], ltrb[bb+3*a_s+j]);
                let ai = ab + j;
                let s = grid_strides[ai];
                xywh[ai]       = grid_cx[ai] + s * (r - l) * 0.5;
                xywh[a + ai]   = grid_cy[ai] + s * (b - t) * 0.5;
                xywh[2*a + ai] = s * (l + r);
                xywh[3*a + ai] = s * (t + b);
            }
        }

        const SCORE_THRESH: f32 = 0.25;
        let mut cands: Vec<(f32, usize, [f32; 4])> = Vec::new();
        for ai in 0..a {
            let (si, j) = anchor_scale[ai];
            let a_s = a_per_scale[si];
            let sb = score_off[si];
            let (best_s, best_c) = (0..nc)
                .map(|c| { let sig = 1.0f32 / (1.0 + (-logits[sb + c*a_s + j]).exp()); (sig, c) })
                .max_by(|(s1,_),(s2,_)| s1.partial_cmp(s2).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            if best_s >= SCORE_THRESH {
                cands.push((best_s, best_c, [xywh[ai], xywh[a+ai], xywh[2*a+ai], xywh[3*a+ai]]));
            }
        }
        cands.sort_by(|(s1,..),(s2,..)| s2.partial_cmp(s1).unwrap_or(std::cmp::Ordering::Equal));

        const NMS_THRESH: f32 = 0.65;
        let mut suppressed = vec![false; cands.len()];
        for i in 0..cands.len() {
            if suppressed[i] { continue; }
            for jj in (i+1)..cands.len() {
                if suppressed[jj] || cands[i].1 != cands[jj].1 { continue; }
                if box_iou_cxcywh(cands[i].2, cands[jj].2) > NMS_THRESH { suppressed[jj] = true; }
            }
        }

        // Un-letterbox → normalised original-image coords.
        let lb_scale = img_size as f32 / orig_w.max(orig_h) as f32;
        let lb_new_w = (orig_w as f32 * lb_scale).round() as usize;
        let lb_new_h = (orig_h as f32 * lb_scale).round() as usize;
        let lb_pad_x = ((img_size - lb_new_w) / 2) as f32;
        let lb_pad_y = ((img_size - lb_new_h) / 2) as f32;
        let sx = orig_w as f32 * lb_scale;
        let sy = orig_h as f32 * lb_scale;

        let mut dets = Vec::new();
        for (i, &(score, cls, [cx, cy, w, h])) in cands.iter().enumerate() {
            if suppressed[i] { continue; }
            dets.push((cls, score, [(cx-lb_pad_x)/sx, (cy-lb_pad_y)/sy, w/sx, h/sy]));
        }
        Ok(dets)
    };

    Ok(Box::new(f))
}

// ── AOT kernel compile (`cargo teeny aot`/`package --device ...`) ──────────
//
// Detected before the normal `--port`/`--model` arg loop runs (which doesn't
// know about `--device` and would otherwise misparse its value as the
// dataset root — see the loop's `other => eprintln!("unknown flag: ...")`
// fallthrough). Mirrors `examples/yolo26.rs`'s `is_aot_invocation`/`run_aot`.

#[cfg(any(feature = "cuda", feature = "riscv"))]
fn is_aot_invocation(raw_args: &[String]) -> bool {
    raw_args.iter().any(|a| a == "--device")
}

#[cfg(any(feature = "cuda", feature = "riscv"))]
fn run_aot(raw_args: &[String]) -> Result<()> {
    use clap::Parser;
    use teeny_core::graph::DtypeRepr;
    use teeny_core::model::LoweringMode;
    use teeny_kernels::graph::TritonLowering;
    use vision_rs::models::yolo::yolo26::{Yolo26Variant, blocks::detect::DetectHead, yolo26};

    /// Matches this demo's own default (`--model ultralytics/yolo26n`, 640×640 — see
    /// `build_infer_fn`'s variant-from-name-suffix inference and its `main()` call site).
    const NC: usize = 80;
    const IMG_SIZE: usize = 640;

    #[derive(Parser)]
    struct AotCli {
        #[command(flatten)]
        aot: teeny_cli::AotArgs,
    }

    let cli = AotCli::parse_from(raw_args);

    let model = yolo26::<f32>(NC, &Yolo26Variant::N, DetectHead::OneToOne);
    let lowering = TritonLowering::new();

    teeny_cli::aot_compile(
        &model,
        DtypeRepr::F32,
        vec![None, Some(3), Some(IMG_SIZE), Some(IMG_SIZE)],
        &lowering,
        LoweringMode::Inference,
        &cli.aot,
    )?;

    println!(
        "AOT compile complete (device={}, cache={})",
        cli.aot.device,
        cli.aot.resolve_cache_dir().display()
    );
    Ok(())
}

// ── WebSocket handler ──────────────────────────────────────────────────────

async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: AppState) {
    let mut rx = state.tx.subscribe();
    let (mut sender, mut receiver) = socket.split();

    loop {
        tokio::select! {
            msg = rx.recv() => {
                match msg {
                    Ok(json) => {
                        if sender.send(Message::Text(json.into())).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        eprintln!("ws client lagged {n} messages");
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
            msg = receiver.next() => {
                match msg {
                    None | Some(Ok(Message::Close(_))) => break,
                    _ => {}
                }
            }
        }
    }
}

async fn health() -> &'static str {
    "ok"
}

// ── Main ───────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    let raw_args: Vec<String> = std::env::args().collect();
    #[cfg(any(feature = "cuda", feature = "riscv"))]
    if is_aot_invocation(&raw_args) {
        return run_aot(&raw_args);
    }
    #[cfg(not(any(feature = "cuda", feature = "riscv")))]
    if raw_args.iter().any(|a| a == "--device") {
        anyhow::bail!("--device (AOT compile) requires the 'cuda' or 'riscv' feature");
    }

    let mut args = std::env::args().skip(1);
    // Matches .env.dev's own DATASETS_CACHE_DIR default, so a fresh checkout with no
    // dataset_root arg and no .env override still lands somewhere writable.
    let datasets_cache_dir = std::env::var("DATASETS_CACHE_DIR").unwrap_or_else(|_| {
        format!("{}/.cache/vision-rs/datasets", std::env::var("HOME").unwrap_or_else(|_| ".".to_owned()))
    });
    let mut root = format!("{datasets_cache_dir}/PKLot/PKLot");
    let mut port: u16 = 3001;
    let mut model_spec: Option<String> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--port" => {
                port = args.next().and_then(|p| p.parse().ok()).unwrap_or(3001);
            }
            "--model" => {
                model_spec = args.next();
            }
            other if !other.starts_with("--") => {
                root = other.to_owned();
            }
            other => eprintln!("unknown flag: {other}"),
        }
    }

    ensure_dataset(Path::new(&root))?;

    println!("scanning dataset at {root}");
    let mut lots = scan_lots(Path::new(&root))?;
    println!("\nfound {} lot(s) — ticking every {}s\n", lots.len(), TICK_SECS);

    // ── CUDA inference setup (no-op without --features cuda) ──────────────

    #[cfg(feature = "cuda")]
    let mut infer_fn: Option<Box<dyn FnMut(&Path) -> Result<Vec<(usize, f32, [f32; 4])>>>> = {
        if let Some(ref spec) = model_spec {
            println!("setting up inference for model '{spec}' …");
            Some(build_infer_fn(spec, 640)?)
        } else {
            None
        }
    };

    #[cfg(not(feature = "cuda"))]
    if model_spec.is_some() {
        anyhow::bail!("--model requires compiling with --features cuda");
    }

    // ── Axum HTTP / WebSocket server ───────────────────────────────────────

    let (tx, _) = broadcast::channel::<String>(BROADCAST_CAPACITY);
    let state = AppState { tx: tx.clone() };

    let app = Router::new()
        .route("/api/ws", get(websocket_handler))
        .route("/api/health", get(health))
        .with_state(state);

    let addr: SocketAddr = ([0, 0, 0, 0], port).into();
    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding to {addr}"))?;
    println!("API server on http://{addr}  (ws://{addr}/api/ws)\n");

    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("axum server failed");
    });

    // ── Iteration loop ─────────────────────────────────────────────────────

    let mut ticker = interval(Duration::from_secs(TICK_SECS));

    loop {
        ticker.tick().await;

        for lot in &mut lots {
            let img = &lot.images[lot.cursor];

            let image_path = img.path.clone();
            let xml_path = image_path.with_extension("xml");

            let (image_bytes, spaces) = match tokio::try_join!(
                tokio::fs::read(&image_path),
                tokio::fs::read_to_string(&xml_path),
            ) {
                Ok((bytes, xml)) => {
                    let spaces = parse_xml_spaces(&xml).unwrap_or_else(|e| {
                        eprintln!("xml parse error {}: {e:#}", xml_path.display());
                        vec![]
                    });
                    (bytes, spaces)
                }
                Err(e) => {
                    eprintln!("error reading {}: {e:#}", image_path.display());
                    lot.cursor = (lot.cursor + 1) % lot.images.len();
                    continue;
                }
            };

            let total = spaces.len();
            let gt_occ = spaces.iter().filter(|s| s.occupied).count();

            // ── Inference (cuda feature) ───────────────────────────────────
            let inf_info: Option<(usize, f32)>;
            #[cfg(feature = "cuda")]
            {
                inf_info = if let Some(ref mut infer) = infer_fn {
                    let t0 = std::time::Instant::now();
                    match infer(&image_path) {
                        Ok(dets) => {
                            let lat_ms = t0.elapsed().as_secs_f32() * 1000.0;
                            // Collect vehicle centers in normalised [0,1] coords.
                            let vehicles: Vec<[f32; 2]> = dets.iter()
                                .filter(|(cls, _, _)| VEHICLE_CLASSES.contains(cls))
                                .map(|(_, _, [cx, cy, _, _])| [*cx, *cy])
                                .collect();
                            // Convert normalised centers → pixels using image dimensions.
                            let (img_w, img_h) = image::load_from_memory(&image_bytes)
                                .map(|i| (i.width() as f64, i.height() as f64))
                                .unwrap_or((1280.0, 720.0));
                            // A space is occupied when any vehicle center falls inside it.
                            let inf_occ = spaces.iter().filter(|s| {
                                let [sx, sy, sw, sh] = s.bbox;
                                vehicles.iter().any(|&[cx_n, cy_n]| {
                                    let cx = cx_n as f64 * img_w;
                                    let cy = cy_n as f64 * img_h;
                                    cx >= sx && cx <= sx + sw && cy >= sy && cy <= sy + sh
                                })
                            }).count();
                            Some((inf_occ, lat_ms))
                        }
                        Err(e) => { eprintln!("inference error: {e:#}"); None }
                    }
                } else {
                    None
                };
            }
            #[cfg(not(feature = "cuda"))]
            { inf_info = None; }

            // ── Log ────────────────────────────────────────────────────────
            if let Some((inf_occ, lat_ms)) = inf_info {
                println!(
                    "[{:>8}] {}  gt={}/{} inf={}/{} {:.1}ms",
                    lot.name, img.filename, gt_occ, total, inf_occ, total, lat_ms
                );
            } else {
                println!(
                    "[{:>8}] {}  total={} occupied={} free={}",
                    lot.name, img.filename, total, gt_occ, total - gt_occ
                );
            }

            // ── Build WebSocket payload (ground-truth annotations) ─────────
            let occupied_ids: Vec<usize> = spaces.iter()
                .filter(|s| s.occupied).map(|s| s.id).collect();
            let ws_spaces: Vec<SpaceInfo> = spaces.iter()
                .map(|s| SpaceInfo { id: s.id, bbox: s.bbox, occupied: s.occupied }).collect();
            let snapshot = ParkingLotSnapshot {
                lot: lot.name.clone(),
                timestamp: parse_timestamp(&img.filename),
                image_b64: BASE64.encode(&image_bytes),
                total_spaces: total,
                occupied_ids,
                spaces: ws_spaces,
            };
            let json = serde_json::to_string(&snapshot).expect("serialisation failed");
            let _ = tx.send(json);

            lot.cursor = (lot.cursor + 1) % lot.images.len();
        }
        println!();
    }
}
