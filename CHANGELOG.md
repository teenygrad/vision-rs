# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.6](https://github.com/teenygrad/vision-rs/compare/vision-rs-v0.1.5...vision-rs-v0.1.6) - 2026-11-12

### Added

- *(parking-garage)* --img-size flag for the server's model input size ([#17](https://github.com/teenygrad/vision-rs/pull/17))
- *(parking-garage)* send the model's detections to the page and draw them ([#18](https://github.com/teenygrad/vision-rs/pull/18))
- *(parking-garage)* default the model input size to 1280 ([#19](https://github.com/teenygrad/vision-rs/pull/19))
- *(parking-garage)* click a lot for a detailed, full-size view ([#20](https://github.com/teenygrad/vision-rs/pull/20))

### Fixed

- strict CLI parsing for the parking-garage binaries; teeny-riscv as a direct dependency ([#15](https://github.com/teenygrad/vision-rs/pull/15))
- find teenyc without TEENYC_PATH; correct the env and performance docs ([#16](https://github.com/teenygrad/vision-rs/pull/16))

## [0.1.5](https://github.com/teenygrad/vision-rs/compare/vision-rs-v0.1.4...vision-rs-v0.1.5) - 2026-11-12

### Other

- upgrade to teenygrad 0.3.1 ([#13](https://github.com/teenygrad/vision-rs/pull/13))

## [0.1.4](https://github.com/teenygrad/vision-rs/compare/vision-rs-v0.1.3...vision-rs-v0.1.4) - 2026-11-10

### Added

- add RISC-V RVV feature and make CUDA build-optional ([#5](https://github.com/teenygrad/vision-rs/pull/5))

### Fixed

- restore parking-garage-webapp frontend via build.rs

### Other

- install cargo-teeny from crates.io ([#7](https://github.com/teenygrad/vision-rs/pull/7))

## [0.1.3](https://github.com/teenygrad/vision-rs/compare/vision-rs-v0.1.2...vision-rs-v0.1.3) - 2026-08-16

### Fixed

- upgrade to teenygrad 0.2.0, resolving sm_120 CUDA launch crash ([#2](https://github.com/teenygrad/vision-rs/pull/2))

### Other

- *(release-plz)* allow_dirty for the tracked-but-gitignored dist/ file ([#3](https://github.com/teenygrad/vision-rs/pull/3))
