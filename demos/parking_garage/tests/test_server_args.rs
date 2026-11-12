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

//! Command-line handling of `parking-garage-server` (#11): `--help`, unknown flags and bad
//! values must stop the program before it touches the dataset, rather than fall through to
//! starting the server (which, with no dataset, downloads ~4.6GB of PKLot).
//!
//! Every run sets `DATASETS_CACHE_DIR` to a temporary directory whose `PKLot/PKLot` already
//! exists, and passes no dataset root, so however the arguments are (mis)parsed the default root
//! exists and nothing is downloaded. An empty dataset makes the server exit on its own ("no lots
//! found"), so a rejection is recognised by the server never getting as far as scanning it.

use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

const TIMEOUT: Duration = Duration::from_secs(10);

/// What the server prints once it has accepted its arguments and moved on to the dataset.
const SCANNING: &str = "scanning dataset";

/// Runs the server with `args`, and returns its output, or panics if it is still running after
/// `TIMEOUT`.
fn run_server(name: &str, args: &[&str]) -> Output {
    let cache: PathBuf =
        std::env::temp_dir().join(format!("parking-garage-args-{name}-{}", std::process::id()));
    std::fs::create_dir_all(cache.join("PKLot/PKLot")).expect("create empty dataset root");

    let mut child = Command::new(env!("CARGO_BIN_EXE_parking-garage-server"))
        .args(args)
        .env("DATASETS_CACHE_DIR", &cache)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn parking-garage-server");
    let start = Instant::now();
    while child.try_wait().expect("poll server").is_none() {
        if start.elapsed() > TIMEOUT {
            let _ = child.kill();
            let _ = child.wait();
            let _ = std::fs::remove_dir_all(&cache);
            panic!(
                "`parking-garage-server {}` was still running after {TIMEOUT:?}",
                args.join(" ")
            );
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let output = child.wait_with_output().expect("collect server output");
    let _ = std::fs::remove_dir_all(&cache);
    output
}

/// Asserts the server refused `args` up front, with an error mentioning `needle`.
fn assert_rejected(name: &str, args: &[&str], needle: &str) {
    let out = run_server(name, args);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let cmd = args.join(" ");
    assert!(!out.status.success(), "`{cmd}` exited successfully");
    assert!(
        !stdout.contains(SCANNING),
        "`{cmd}` was accepted and went on to scan the dataset:\n{stdout}{stderr}"
    );
    assert!(
        stderr.contains(needle),
        "`{cmd}`: stderr should mention {needle:?}:\n{stderr}"
    );
}

#[test]
fn help_prints_usage_and_exits_successfully() {
    for flag in ["--help", "-h"] {
        let out = run_server(flag.trim_start_matches('-'), &[flag]);
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(out.status.success(), "{flag}: {:?}\n{stdout}", out.status);
        assert!(
            !stdout.contains(SCANNING),
            "{flag} went on to scan the dataset"
        );
        assert!(
            stdout.contains("--port") && stdout.contains("--model"),
            "{flag}: {stdout}"
        );
    }
}

#[test]
fn unknown_flag_is_an_error() {
    assert_rejected("unknown", &["--verbose"], "--verbose");
    // A typo'd flag's value must not then be taken as the dataset root.
    assert_rejected("typo", &["--mdoel", "ultralytics/yolo26n"], "--mdoel");
}

#[test]
fn bad_port_values_are_errors() {
    assert_rejected("port-abc", &["--port", "abc"], "abc");
    assert_rejected("port-range", &["--port", "99999"], "99999");
    assert_rejected("port-missing", &["--port"], "--port");
    // A missing value must not swallow the next flag.
    assert_rejected(
        "port-flag",
        &["--port", "--model", "ultralytics/yolo26n"],
        "--port",
    );
}

#[test]
fn model_without_a_value_is_an_error() {
    assert_rejected("model-missing", &["--model"], "--model");
}
