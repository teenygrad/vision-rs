# The `teenyc` Toolchain

vision-rs's kernels (see [Custom Kernels](./custom-kernels.md)) are
ordinary-looking `#[kernel]`-annotated Rust functions, but they're compiled
to PTX/MLIR by a separate, custom compiler fork — `teenyc` — not by the
stable `rustc` you build the rest of vision-rs with. See
[Installation & Toolchain](../getting-started/installation.md) for how to
install it.

## Two compilation modes

**JIT (just-in-time)**: during normal development — running `cargo test`,
the `yolo26` example, or anything else that touches the `cuda` feature —
kernels are compiled by `teenyc` at runtime, the first time each kernel is
invoked, then cached, so the process needs to be able to shell out to the
compiler on demand. It uses the `teenyc` toolchain `cargo teeny
install-toolchain` installed, or the binary `TEENYC_PATH` names.

**AOT (ahead-of-time)**: `cargo teeny package`/`aot` cross-compiles kernels
*before* deployment, producing a `cache/` directory of precompiled PTX that
ships alongside the binary. This is how vision-rs runs on a Jetson Orin
Nano with no `teenyc` (or even Rust toolchain) installed on the device — see
[Packaging a Deployable Bundle](../deployment/packaging.md). The binary
auto-detects a sibling `cache/` directory at runtime and uses it instead of
trying to JIT-compile, which would simply fail on a device without
`teenyc`.

## Why a separate compiler

`teenyc` exists because the kernel DSL compiles through an MLIR backend
that isn't part of upstream `rustc`. If a kernel fails to compile with an
error that doesn't obviously trace back to anything in vision-rs or
teenygrad's Rust-level code, the root cause may be in this compiler fork
rather than in either of those — see
[teenygrad/teeny](https://github.com/teenygrad/teeny).

## Compute capability

AOT-compiled kernels are compiled for a specific GPU compute capability
(e.g. `sm_87` for the Jetson Orin Nano's Ampere GPU), passed via
`--options capability=sm_87` to `cargo teeny package`. JIT compilation
instead targets whatever capability the local device reports at runtime.
See [Packaging a Deployable Bundle](../deployment/packaging.md) for the
full `package` invocation and what `ptx-version` overrides are for.
