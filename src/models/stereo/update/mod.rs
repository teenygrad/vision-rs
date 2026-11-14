//! FoundationStereo's iterative refinement head (`core/update.py`), as graph builders.
//!
//! Each builder returns a closure that records its block into a `SymTensor` graph.
//! Weights are named after the PyTorch module's `state_dict` keys, relative to the
//! scope the caller opens, so checkpoint tensors load by name.

pub mod disp_head;
pub mod gru;
pub mod motion_encoder;
pub mod ops;
