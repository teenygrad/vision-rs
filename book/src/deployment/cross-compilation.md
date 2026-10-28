# Building for Jetson Orin Nano

vision-rs targets the Jetson Orin Nano (`aarch64-unknown-linux-gnu`) using
[`cross`](https://github.com/cross-rs/cross) via the `cargo-teeny` plugin.

## Additional prerequisites

1. **Install `cross`**:
   ```bash
   cargo install cross --git https://github.com/cross-rs/cross
   ```
2. **Add the aarch64 target**:
   ```bash
   rustup target add aarch64-unknown-linux-gnu
   ```
3. **CUDA aarch64 libraries** — the build mounts the host's CUDA aarch64
   target directory into the cross container, so the cross-compiled binary
   links against the CUDA version actually present on the Jetson's JetPack
   (independent of the host's own CUDA 13.3, which is only used for
   AOT-compiling kernels). Default path:
   ```
   /usr/local/cuda-12.6/targets/aarch64-linux
   ```
   If your device's JetPack CUDA version differs, pass `--cuda-path <path>`
   on every `build`/`package` command below.

## Building

```bash
# Build the library in release mode (default)
cargo teeny build --target jetson-orin-nano

# Build all examples / a single example
cargo teeny build --target jetson-orin-nano --examples
cargo teeny build --target jetson-orin-nano --example yolo26

# Type-check only (faster feedback), or lint
cargo teeny check --target jetson-orin-nano
cargo teeny clippy --target jetson-orin-nano

# Debug build
cargo teeny build --target jetson-orin-nano --no-release

# Custom CUDA path (target device's JetPack CUDA version)
cargo teeny build --target jetson-orin-nano --cuda-path /usr/local/cuda-12.8/targets/aarch64-linux
```

Compiled artifacts land in `target/aarch64-unknown-linux-gnu/release/`.

## How it works

`cargo teeny build` wraps `cross build` and automatically:

- Resolves the [teenygrad](https://github.com/teenygrad/teenygrad) workspace
  root from `Cargo.toml`'s `[patch.crates-io]` entries and mounts it into
  the cross container — required because `cross` only auto-mounts
  individual crate directories, not the workspace root providing
  `Cargo.toml` inheritance.
- Mounts the host CUDA aarch64 target directory at the path the cross
  container's Dockerfile expects.
- Uses the custom image defined in `docker/Dockerfile.jetson-orin-nano`,
  which extends the `cross` base image with `clang-12` (required by
  `bindgen` for aarch64 cross-bindings).

## RISC-V RVV

RISC-V boards (`riscv-rvv`, `spacemit-k3`) cross-compile to `riscv64gc-unknown-linux-gnu` with
`cross`'s stock image, with no CUDA mount or custom Dockerfile. Turn off the default `cuda` feature:

```bash
cargo teeny build --target riscv-rvv --example yolo26 --no-default-features --features riscv,training
# run the result locally under qemu-riscv64
cargo teeny run --target riscv-rvv --example yolo26 --no-default-features --features riscv,training -- --help
```

Only ahead-of-time kernel compilation targets RVV. Inference still needs `cuda`, because
teenygrad has no RISC-V model runtime yet.

Once you have a cross-compiled binary, see
[Packaging a Deployable Bundle](./packaging.md) to combine it with
ahead-of-time-compiled kernels into something you can copy straight to the
device.
