//! Kernels specific to stereo matching.

pub mod avg_pool2d_padded;
pub mod bilinear_sampler_1d;
pub mod conv_transpose2d_k4s2p1;
pub mod correlation;
pub mod instance_norm3d;
pub mod interpolate_bilinear;
pub mod l2_normalize;
pub mod unfold3x3;
