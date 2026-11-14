// Helpers shared between integration tests. Each test binary that needs them declares
// `mod common;`, so unused items in a given binary are expected.
#![allow(dead_code)]

pub mod stereo_block;
