/*
 * Copyright 2026 Teenygrad
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Graph builders for FoundationStereo's cost-volume geometry.

use teeny_core::dtype::FloatBytes;
use teeny_core::graph::{CustomData, SymTensor};

use super::kernels::{correlation::CorrelationOp, l2_normalize::L2NormalizeOp};

/// All-pairs cosine correlation volume, as built by FoundationStereo
/// (`core/geometry.py:74`):
///
/// ```python
/// corr = torch.einsum('aijk,aijh->ajkh',
///                     F.normalize(fmap1.float(), dim=1),
///                     F.normalize(fmap2.float(), dim=1))
/// ```
///
/// Both feature maps are L2-normalised along the channel axis, then reduced against
/// each other. Since both operands are unit vectors the result is a cosine similarity
/// in `[-1, 1]`, independent of feature scale.
///
/// `fmap1` is `[B, C, H, W1]`, `fmap2` is `[B, C, H, W2]`; the result is
/// `[B, H, W1, W2]`. That is exactly the layout
/// [`super::kernels::bilinear_sampler_1d`] consumes once reshaped to
/// `[B*H*W1, 1, W2]`, which is how the lookup indexes it.
///
/// `block_c` is the kernel launch block size along the channel axis and must be at
/// least `C`.
///
/// Normalisation is a separate node rather than fused into the correlation: it mirrors
/// FoundationStereo, which normalises whole feature maps, and keeps the normalised maps
/// available for reuse.
pub fn correlation_volume<D: FloatBytes + Send + Sync + 'static>(
    fmap1: &SymTensor,
    fmap2: &SymTensor,
    block_c: i32,
) -> SymTensor {
    let n1 = fmap1.record_custom(CustomData::new(L2NormalizeOp::<D>::new(block_c)), &[], None);
    let n2 = fmap2.record_custom(CustomData::new(L2NormalizeOp::<D>::new(block_c)), &[], None);
    n1.record_custom(
        CustomData::new(CorrelationOp::<D>::new(block_c)),
        &[&n2],
        None,
    )
}
