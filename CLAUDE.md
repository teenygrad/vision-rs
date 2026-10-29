# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`vision-rs` is a high-performance computer vision SDK for Rust (crate name `vision_rs`, edition 2024, Apache-2.0). It is in early development — `src/lib.rs` is currently a stub.

## Commands

```bash
cargo build               # Build
cargo test                # Run all tests
cargo test <test_name>    # Run a single test by name
cargo clippy              # Lint
cargo fmt                 # Format
cargo doc --open          # Build and open docs locally
```

## Benchmarking

First time only: `cp .env.dev .env` and adjust the paths for your machine (`.env` is gitignored;
`.env.dev` is the tracked example — see README.md's "Installing the toolchain" section for
where `TEENYC_PATH` comes from).

### Build (required before any bench run)

```bash
source .env
cargo build --release --example yolo26 --features cuda
```

### Run benchmark (clean numbers)

```bash
source .env
./target/release/examples/yolo26 bench \
  --model ultralytics/yolo26n \
  --dataset assets/datasets/coco128.toml \
  --skip-map \
  --warmup 10 \
  --runs 100
```

Remove `--skip-map` to also compute mAP@0.5 (adds ~30s).

### nsys profiling (per-kernel breakdown)

The bench command wraps the batch=1 timed loop with `cudaProfilerStart/Stop`, so
`--capture-range=cudaProfilerApi` records only the timed region (not warmup or
compile). Use `--cuda-graph-trace=node` to get individual kernel stats inside CUDA
graphs, which are otherwise opaque to nsys.

```bash
source .env
nsys profile \
  --capture-range=cudaProfilerApi \
  --cuda-graph-trace=node \
  --output=/tmp/yolo26_bench \
  --force-overwrite=true \
  --stats=true \
  --kill=none \
  ./target/release/examples/yolo26 bench \
    --model ultralytics/yolo26n \
    --dataset assets/datasets/coco128.toml \
    --skip-map \
    --warmup 10 \
    --runs 100
```

`--kill=none` prevents nsys from SIGTERM-ing the process after the capture range
ends (without it the final results row never prints).

The report is saved to `/tmp/yolo26_bench.nsys-rep` and can be opened in the
Nsight Systems GUI. The `--stats=true` flag prints `cuda_gpu_kern_sum`,
`cuda_api_sum`, and `cuda_gpu_mem_time_sum` tables directly to stdout.

### TensorRT baseline (bench.py)

Runs the ultralytics benchmark across PyTorch / ONNX Runtime / TensorRT FP32.
TRT engine export takes ~25s on first run; the `.engine` file is cached at
`$MODELS_CACHE_DIR/ultralytics/yolo26n/yolo26n.engine`.

`bench.py` lives in the `../vision-rs-utils` sibling repo (see "Sibling Projects" below),
not in this repo:

```bash
source .env
python3 ../vision-rs-utils/bench.py
```

### Cross-backend accuracy check (CUDA vs RISC-V under qemu)

`validate --runtime runtime` runs the unoptimised graph kernel-by-kernel through
`teeny_test::model::RuntimeModel` on whichever `teeny-runtime` backend the binary was built
for, so CUDA and RISC-V can be compared like for like (`--runtime cuda-graph`, the default, is
the CUDA-only fast path). RISC-V runs under `qemu-riscv64` (needs `qemu-user` and
`gcc-riscv64-linux-gnu`) at roughly 2.5 min per image; build it into a separate target dir so
it doesn't clobber the CUDA build. `TEENYC_CAPABILITY` selects the RISC-V target.

```bash
source .env
cargo build --release --example yolo26 --features cuda
CARGO_TARGET_DIR=target/riscv cargo build --release --example yolo26 --no-default-features --features qemu

./target/release/examples/yolo26 validate --model ultralytics/yolo26n -b 1 --limit 50 \
  --runtime runtime --dump-outputs /tmp/cuda-50
TEENYC_CAPABILITY=spacemit-k3 ./target/riscv/release/examples/yolo26 validate \
  --model ultralytics/yolo26n -b 1 --limit 50 --runtime runtime --dump-outputs /tmp/riscv-50
```

`--dump-outputs` writes each image's raw `boxes`/`scores` tensors as little-endian f32 for an
element-wise diff; `--trace` prints per-node launch times.

#### On a RISC-V board (SpacemiT K3, ssh alias `jupiter2`)

A `riscv64` build launches kernels natively, in-process, across every core (`TEENY_RISCV_THREADS`
overrides), with no qemu. `scripts/riscv-board-validate.sh` cross-compiles the example for
`riscv64gc-unknown-linux-gnu`, builds the kernel cache here by running that binary on one image
under qemu-user (teenyc is x86-only; the cache is keyed on kernel source + target cpu, so it is
host-independent), then rsyncs binary + cache + weights + the first N val2017 images to the board
and runs there with teenyc disabled. The board needs only glibc >= 2.39.

```bash
source .env
scripts/riscv-board-validate.sh -n 50 -H jupiter2      # outputs in target/riscv64-native/validate
python3 scripts/diff_outputs.py /tmp/cuda-50 target/riscv64-native/validate/board
```

## Architecture

The library is a single crate (`src/lib.rs` as the root). As modules are added, they should be declared here and organized under `src/` as either inline modules or separate files.

## Sibling Projects

### `../teenygrad`

The ML/tensor runtime this project builds on. All its crates are pre-patched in `Cargo.toml` via `[patch.crates-io]`, so adding a dependency is straightforward:

```toml
# In [dependencies] — Cargo resolves to the local path automatically
teeny-core = "0.1"
```

Available crates: `teeny-core`, `teeny-fxgraph`, `teeny-kernels`, `teeny-triton`, `teeny-compiler`, `teeny-macros`, `teeny-cuda`, `teeny-cache`, `teeny-data`, `teeny-http`, `teeny-nlp`, `teeny-onnx`, `teeny-torch`.

When these crates are eventually published to crates.io, remove the `[patch.crates-io]` block from `Cargo.toml` and the version specifiers will resolve normally.

### `../teeny`

A custom fork of the Rust compiler (the `teenyc` binary) used to compile GPU kernels ahead-of-time
and at runtime (see README.md's "Installing the toolchain" section). `vision-rs` does not depend
on it directly as a crate and the crate itself is built with the stable release compiler; only
`TEENYC_PATH` (see `.env.dev`) needs to point at a `teenyc` binary, normally installed via
`cargo teeny install-toolchain` rather than built from a full `../teeny` source checkout. If a GPU
kernel fails to compile with an inexplicable error, the root cause may be in this compiler fork
rather than in `vision-rs` or `teenygrad`.

### `../vision-rs-utils`

Development/debug Python scripts formerly under this repo's `scripts/` directory and root
`bench.py` (github.com/teenygrad/vision-rs-utils, private). `vision-rs` no longer executes
anything from it at runtime — model weights are downloaded pre-converted from Hugging Face
(`teenygrad/ultralytics-yolo26`, see `assets/models/ultralytics/*.toml`). The scripts there are
purely for manual comparison against the Ultralytics reference implementation (`debug_infer.py`,
`debug_train_grads.py`, `bench.py`) and for re-converting `.pt` checkpoints to safetensors by hand
(`ultralytics/convert_model.py`) if the Hugging Face weights ever need to be regenerated.


<!-- BEGIN BEADS INTEGRATION v:2 profile:minimal (br) -->
## Beads Issue Tracker

This project uses **br (beads_rust)** for issue tracking.

**Note:** `br` is non-invasive and never executes git commands. After `br sync --flush-only`, you must manually run `git add .beads/ && git commit`.

### Quick Reference

```bash
br ready              # Find available work
br show <id>          # View issue details
br update <id> --claim  # Claim work
br close <id>         # Complete work
```

### Rules

- Use `br` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Use `br` comments/labels for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   br sync --flush-only
   git add .beads/
   git commit -m "sync beads"
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
