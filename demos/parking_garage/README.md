# Parking Garage Demo

This crate demonstrates the two-process parking garage shape:

- `parking-garage-server` publishes sample backend events over a Unix domain socket.
- `parking-garage-webapp` serves a small Vue page and exposes an HTTP/WebSocket API for the browser.

Run the server:

```bash
cargo run --bin parking-garage-server
```

Run the browser-facing webapp in another terminal:

```bash
cargo run --bin parking-garage-webapp
```

Then open <http://127.0.0.1:3000>.

The server accepts an optional Unix socket path. If omitted, it uses `/tmp/vision-rs-parking-garage.sock`.

The client accepts an optional HTTP listen address. If omitted, it uses `0.0.0.0:3000`.

## Dataset

The server reads the [PKLot dataset](https://huggingface.co/datasets/teenygrad/pklot)
(Almeida et al., *"PKLot – A robust dataset for parking lot classification"*, Expert Systems
with Applications, 2015; CC BY 4.0 — attribute the paper if you redistribute this data).

It defaults to `$DATASETS_CACHE_DIR/PKLot/PKLot` (falling back to
`$HOME/.cache/vision-rs/datasets/PKLot/PKLot` if `DATASETS_CACHE_DIR` isn't set), or pass an
explicit path as the first positional arg. If that directory doesn't exist yet, the ~4.6GB
archive is downloaded (with checksum verification and resumable retries — this is a large
download over a real network, so drops are expected and handled) from our
[HF mirror](https://huggingface.co/datasets/teenygrad/pklot) — an unmodified copy of the
original UFPR archive, hosted there since `inf.ufpr.br` isn't always reachable — and extracted
automatically on first run.

## RISC-V RVV (SpacemiT K3)

`--features riscv` (without `cuda`) runs inference on RISC-V: the server compiles the YOLO26n
graph kernel-by-kernel through teeny-test's `RuntimeModel`, and a riscv64 build launches the
kernels natively across every core (`TEENY_RISCV_THREADS` overrides). `cargo teeny package`
cross-compiles both binaries and AOT-compiles the kernels into the bundle's `cache/`, so the
board needs no `teenyc`. The server reads the kernels' target (`capability=spacemit-k3`) from
the bundle's `conf/`, and `TEENYC_CAPABILITY` overrides it. From the repo root:

```bash
source .env
cargo teeny package --target spacemit-k3 --package parking-garage \
  --bin parking-garage-server --bin parking-garage-webapp \
  --dest ./demos/parking_garage/dist/parking-garage-k3 \
  --no-default-features --features riscv
cargo teeny deploy --package ./demos/parking_garage/dist/parking-garage-k3 \
  --host jupiter2 --dest /home/<user>/parking-garage-demo --overwrite
```

On the board, run each binary in its own session:

```bash
cd ~/parking-garage-demo
TEENYC_PATH=/nonexistent/teenyc MODELS_CACHE_DIR=$PWD/data/models DATASETS_CACHE_DIR=$PWD/data/datasets \
  ./bin/parking-garage-server --model ultralytics/yolo26n
./bin/parking-garage-webapp
```

`TEENYC_PATH` must be set even though every kernel comes from `cache/`: the kernel compiler looks
for `teenyc` (via `rustup`) before checking the cache, and the board has neither. A dummy path is
enough, and makes a cache miss fail with a clear "kernel not cached" error.

With `--model`, the webapp shows inference time per image (last, mean, std dev, max) over every
image since the server started; the server also logs them per image.

The weights (~10MB) and the PKLot dataset (~4.6GB) are downloaded into `data/` on first run
unless you copy them there first.

## Building and Deploying to NVIDIA Jetson (Jetson Orin Nano)

This demo can be cross-compiled and deployed to a Jetson Orin Nano using the same workflow as the main vision-rs crate.

### Prerequisites

- Host system: Ubuntu 24.04+
- Rust toolchain: Installed via [rustup](https://rustup.rs/)
- [cargo-teeny](https://github.com/teenygrad/cargo-teeny): For cross-compiling and packaging
- [cross](https://github.com/cross-rs/cross): For aarch64 cross-builds
- aarch64 CUDA libraries: Downloadable from NVIDIA's CUDA toolkit for Jetson (see README in project root for instructions)
- Jetson device accessible via SSH (for deployment)

### Cross-Compiling for Jetson

On your x86 host machine, ensure `cargo-teeny` and `cross` are installed and the proper target is added:

```bash
cargo install --git https://github.com/teenygrad/cargo-teeny
cargo install cross --git https://github.com/cross-rs/cross
rustup target add aarch64-unknown-linux-gnu
```

#### Package the Demo for Jetson

`cargo teeny` only supports being invoked from the repo root (where `Cross.toml` lives) —
not from within `demos/parking_garage` — so select this crate with `--package`/`-p` instead
of `cd`-ing into its directory:

`--bin` is repeatable, so both binaries can be packaged into the same bundle in one
invocation. `parking-garage-webapp` has no GPU kernels of its own — it no-ops when the AOT
step invokes it with `--device`/`--options` (see `src/bin/webapp.rs`) — so it's safe to pass
it through the same `--device`/`--features cuda` flags as the server:

```bash
# From the root of the repo
source .env
cargo teeny package \
  --target jetson-orin-nano \
  --package parking-garage \
  --bin parking-garage-server \
  --bin parking-garage-webapp \
  --dest ./demos/parking_garage/dist/parking-garage-orin \
  --device cuda \
  --features cuda \
  --options "capability=sm_87,ptx-version=82,sm-count=8"
```

`sm-count=8` is the Jetson Orin Nano's actual SM count — enables shape-adaptive conv kernel
tile-size selection so deep/small-spatial layers don't under-occupy this GPU's 8 SMs at the
default tile size. Omit it to keep the previous fixed-tile-size behavior.

This creates a self-contained bundle in `./demos/parking_garage/dist/parking-garage-orin/`:

```text
demos/parking_garage/dist/parking-garage-orin/
  bin/parking-garage-server   # Cross-compiled for Jetson (aarch64)
  bin/parking-garage-webapp   # Cross-compiled for Jetson (aarch64)
  cache/                      # Precompiled GPU kernels (server only)
  conf/                       # Build provenance info
  data/                       # Empty, will contain models/datasets if used
```

- Adjust `--options` as needed for your Jetson's CUDA version/GPU capability.

#### Deploy to Jetson

Copy the package to your Jetson device using the deploy command (replace user/host as appropriate):

```bash
cargo teeny deploy \
  --package ./demos/parking_garage/dist/parking-garage-orin \
  --host <user>@<orin-host> \
  --dest /home/<user>/parking-garage-demo
```

#### Running on the Jetson

SSH into your Jetson, `cd` into the deployed directory, and run each binary (in separate
terminals/sessions — the server listens on port 3001, the webapp on port 3000):

```bash
cd /home/<user>/parking-garage-demo
./bin/parking-garage-server
./bin/parking-garage-webapp
```

If your application uses data/models, place them into the `data/` subfolder as needed.

> For further details on Jetson cross-compilation, packaging, and deployment, see the [vision-rs main README](../../README.md#cross-compilation-for-jetson-orin-nano).
