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
 * Static file server for the parking garage frontend.
 *
 * Ports:
 *   webapp (this binary) — default 3000  — serves the embedded frontend (see INDEX_HTML below)
 *   server               — default 3001  — WebSocket API (ws://localhost:3001/api/ws)
 */

use anyhow::{Context, Result};
use axum::{Router, response::Html};
use dotenv::dotenv;
use std::net::SocketAddr;
use tokio::net::TcpListener;
use tower_http::trace::TraceLayer;

// `ui/dist/index.html` is a single self-contained file (loads Vue/Tailwind from CDN, no
// separate JS/CSS bundle) — embedding it at compile time means this binary has no runtime
// dependency on `ui/dist/` existing next to it, so `cargo teeny package`/`deploy` (which only
// ever handles bin/cache/conf/data) doesn't need to know about it at all.
const INDEX_HTML: &str = include_str!("../../ui/dist/index.html");

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

const USAGE: &str = "\
Usage:
  parking-garage-webapp [ADDRESS]

ADDRESS is the address to serve the frontend on, like 127.0.0.1:3000 (default 0.0.0.0:3000).

Options:
  -h, --help      Print this help";

/// Parses the webapp's arguments into its listen address, or `None` for `--help`, rejecting
/// anything else rather than ignoring it.
fn parse_args(
    args: impl IntoIterator<Item = String>,
) -> std::result::Result<Option<SocketAddr>, String> {
    let mut addr: Option<SocketAddr> = None;
    for arg in args {
        match arg.as_str() {
            "-h" | "--help" => return Ok(None),
            flag if flag.starts_with('-') => return Err(format!("unknown flag: {flag}")),
            value => {
                let parsed = value.parse().map_err(|_| {
                    format!("invalid address '{value}': expected an address like 127.0.0.1:3000")
                })?;
                if let Some(previous) = addr.replace(parsed) {
                    return Err(format!(
                        "unexpected argument '{value}': the address is already '{previous}'"
                    ));
                }
            }
        }
    }
    Ok(Some(
        addr.unwrap_or_else(|| SocketAddr::from(([0, 0, 0, 0], 3000))),
    ))
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    // `cargo teeny package --bin parking-garage-webapp ...` drives every packaged binary
    // through the same host-side AOT step (`cargo run --bin <name> -- --device ... --options
    // ...`, see `parking-garage-server`'s `is_aot_invocation`/`run_aot`). This binary has no
    // GPU kernels to compile, so just no-op instead of misparsing `--device`'s value as the
    // listen address below.
    if std::env::args().any(|a| a == "--device") {
        println!("parking-garage-webapp: nothing to AOT-compile (no GPU kernels)");
        return Ok(());
    }

    tracing_subscriber::fmt::init();

    let addr = match parse_args(std::env::args().skip(1)) {
        Ok(Some(addr)) => addr,
        Ok(None) => {
            println!("{USAGE}");
            return Ok(());
        }
        Err(err) => {
            eprintln!("error: {err}\n\n{USAGE}");
            std::process::exit(2);
        }
    };

    let app = Router::new()
        .fallback(index)
        .layer(TraceLayer::new_for_http());

    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind HTTP listener at {addr}"))?;

    println!("parking garage webapp on http://{addr}");
    println!("  → frontend connects to ws://localhost:3001/api/ws by default");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("HTTP server failed")?;

    Ok(())
}

async fn shutdown_signal() {
    if let Err(error) = tokio::signal::ctrl_c().await {
        tracing::warn!(%error, "failed to listen for shutdown signal");
    }
}
