# Installation & Toolchain

## Prerequisites

- **OS**: Ubuntu — the only supported development/build platform today.
- **Rust**: the latest stable toolchain via [rustup](https://rustup.rs/).
- **CUDA Toolkit 13.3 or later** on the host — required for the `cuda`
  feature (on by default) and for ahead-of-time kernel compilation.
- Only if cross-compiling/deploying to a Jetson Orin Nano:
  [`cross`](https://github.com/cross-rs/cross), Docker, the
  `aarch64-unknown-linux-gnu` rustup target, `cargo-teeny`, and SSH/`rsync`
  access to the device — see [Cross-Compilation](../deployment/cross-compilation.md).

## Clone and configure

```bash
git clone https://github.com/teenygrad/vision-rs
cd vision-rs
cp .env.dev .env   # then edit the paths inside for your machine
```

## The `teenyc` compiler

vision-rs itself builds against **stable** Rust, but its GPU kernels
(Triton-DSL, compiled to PTX/MLIR) go through a separate, custom compiler
fork: `teenyc`. You need it on your machine, with `TEENYC_PATH` (set in
`.env`) pointing at it, before building or running anything that uses the
`cuda` feature (on by default).

The supported way to get it is via
[`cargo-teeny`](https://github.com/teenygrad/cargo-teeny), which installs a
prebuilt release — you do **not** need to clone or build the `teeny`
compiler fork from source:

```bash
cargo install --git https://github.com/teenygrad/cargo-teeny
cargo teeny install-toolchain
```

This links the compiler as `stable-teenyc-x86_64-unknown-linux-gnu` under
`~/.rustup/toolchains/`. Verify it:

```bash
rustup toolchain list | grep teeny
rustup run stable-teenyc-x86_64-unknown-linux-gnu teenyc --version
```

`.env.dev` already points `TEENYC_PATH` at the right place once installed
this way. `source .env` in every shell before building/running/testing —
the test suite and examples compile kernels via `teenyc` at runtime.

## Build

```bash
source .env
cargo build --release
cargo test
```

See [Custom Kernels](../kernels-and-performance/custom-kernels.md) for how
the kernel compilation pipeline fits together, and
[Building for Jetson Orin Nano](../deployment/cross-compilation.md) for
cross-compilation.
