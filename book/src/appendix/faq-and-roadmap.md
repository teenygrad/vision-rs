# FAQ & Roadmap

## Roadmap

vision-rs doesn't have a published, authoritative roadmap yet. What's
implemented today:

- **Model support**: YOLO26 (variants N/S/M/L/XL), inference and training
  (single-head and dual-assignment).
- **Deployment**: cross-compilation and packaging for Jetson Orin Nano.
- **Kernels**: Flash Attention 2, PSA attention wrappers, detect-decode,
  CIoU/classification loss.

`DetectorConfig` is deliberately structured as an enum over model families
(see [The Detection API](../core-concepts/detection-api.md)) so additional
model families can be added without breaking the top-level
`ObjectDetector` API — check the
[repository](https://github.com/teenygrad/vision-rs) for current activity
if you need to know what's actively being worked on.

## FAQ

### Why is `cuda` a default feature rather than optional?

Because vision-rs's models are traced through teenygrad's computational
graph and lowered to GPU kernels — there's currently no CPU execution path
for the shipped models. `--no-default-features` still builds and documents
the parts of the crate that aren't behind `cuda`/`training` (mostly type
definitions), which is what powers this crate's docs.rs page, but it won't
get you a runnable detector without a GPU.

### Why does this depend on a custom Rust compiler fork (`teenyc`)?

Only for compiling GPU kernels (see
[The `teenyc` Toolchain](../kernels-and-performance/teenyc-toolchain.md)) —
the crate itself builds with stable `rustc`. The kernel DSL compiles
through an MLIR backend that isn't part of upstream `rustc`, so a separate
compiler handles just that piece, either just-in-time during development or
ahead-of-time when packaging for deployment.

### Can I install this from crates.io yet?

Yes: `cargo add vision-rs`. Its `teeny-*` dependencies from
[teenygrad](https://github.com/teenygrad/teenygrad) come from crates.io too,
so no teenygrad checkout is needed.

### I'm interested in contributing — how do I get started?

See [Contributing to vision-rs](../contributing/contributing.md).
