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

//! Command-line handling of `parking-garage-webapp`: `--help`, unknown flags, extra arguments
//! and bad addresses must stop it with a status and a message naming the problem, while a valid
//! listen address still starts it.

use std::io::{BufRead, BufReader};
use std::process::{Command, Output, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

const TIMEOUT: Duration = Duration::from_secs(10);

/// What the webapp prints once it is listening.
const LISTENING: &str = "parking garage webapp on http://";

fn webapp(args: &[&str]) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_parking-garage-webapp"));
    cmd.args(args).stdout(Stdio::piped()).stderr(Stdio::piped());
    cmd
}

/// Runs the webapp with `args` and returns its output, or panics if it is still running after
/// `TIMEOUT` (i.e. it started serving instead of exiting).
fn run_webapp(args: &[&str]) -> Output {
    let mut child = webapp(args).spawn().expect("spawn parking-garage-webapp");
    let start = Instant::now();
    while child.try_wait().expect("poll webapp").is_none() {
        if start.elapsed() > TIMEOUT {
            let _ = child.kill();
            let _ = child.wait();
            panic!(
                "`parking-garage-webapp {}` was still running after {TIMEOUT:?}",
                args.join(" ")
            );
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    child.wait_with_output().expect("collect webapp output")
}

/// Asserts the webapp refused `args`, with an error mentioning `needle`.
fn assert_rejected(args: &[&str], needle: &str) {
    let out = run_webapp(args);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let cmd = args.join(" ");
    assert!(!out.status.success(), "`{cmd}` exited successfully");
    assert!(!stdout.contains(LISTENING), "`{cmd}` started serving");
    assert!(
        stderr.contains(needle),
        "`{cmd}`: stderr should mention {needle:?}:\n{stderr}"
    );
}

#[test]
fn help_prints_usage_and_exits_successfully() {
    for flag in ["--help", "-h"] {
        let out = run_webapp(&[flag]);
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(out.status.success(), "{flag}: {:?}\n{stdout}", out.status);
        assert!(
            stdout.contains("Usage") && stdout.contains("ADDRESS"),
            "{flag}: {stdout}"
        );
    }
}

#[test]
fn unknown_flag_is_an_error() {
    assert_rejected(&["--verbose"], "--verbose");
    assert_rejected(&["127.0.0.1:0", "--verbose"], "--verbose");
}

#[test]
fn extra_argument_is_an_error() {
    assert_rejected(&["127.0.0.1:0", "127.0.0.1:1"], "127.0.0.1:1");
}

#[test]
fn bad_address_is_an_error() {
    assert_rejected(&["localhost"], "localhost");
}

#[test]
fn valid_address_starts_serving() {
    let mut child = webapp(&["127.0.0.1:0"])
        .spawn()
        .expect("spawn parking-garage-webapp");
    let stdout = child.stdout.take().expect("webapp stdout");
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let deadline = Instant::now() + TIMEOUT;
    let mut listening = false;
    while let Some(left) = deadline.checked_duration_since(Instant::now()) {
        match rx.recv_timeout(left) {
            Ok(line) if line.contains(LISTENING) => {
                listening = true;
                break;
            }
            Ok(_) => {}
            Err(_) => break,
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    assert!(listening, "webapp didn't report listening on 127.0.0.1:0");
}
