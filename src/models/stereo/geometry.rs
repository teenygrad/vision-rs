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

use super::kernels::{
    correlation::CorrelationOp, geo_lookup::GeoLookupOp, l2_normalize::L2NormalizeOp,
    pool_pairs::PoolPairsOp,
};
use super::update::ops::cat;

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

/// The geometry pyramid of `Combined_Geo_Encoding_Volume.__init__`
/// (`core/geometry.py`), built once per image pair.
pub struct GeoPyramid {
    /// Filtered cost volume per level, `[B, C, D / 2^i, H, W]`.
    pub geo: Vec<SymTensor>,
    /// All-pairs correlation per level, `[B, H, W, W2 / 2^i]`.
    pub corr: Vec<SymTensor>,
}

/// Builds `levels` pyramid levels from the cost volume `geo` `[B, C, D, H, W]` and the
/// correlation `corr` `[B, H, W, W2]` (as [`correlation_volume`] returns it). Each level
/// averages adjacent pairs of the one before along the disparity axis (cost volume) or
/// the right-image column axis (correlation), as `F.avg_pool2d(x, [1, 2], stride=[1, 2])`
/// does in PyTorch's permuted layout.
pub fn geo_pyramid<D: FloatBytes + Send + Sync + 'static>(
    geo: &SymTensor,
    corr: &SymTensor,
    levels: usize,
) -> GeoPyramid {
    let mut g = vec![geo.clone()];
    let mut c = vec![corr.clone()];
    for _ in 1..levels {
        let gl = g.last().unwrap().record_custom(
            CustomData::new(PoolPairsOp::<D>::new(2, 256)),
            &[],
            None,
        );
        let cl = c.last().unwrap().record_custom(
            CustomData::new(PoolPairsOp::<D>::new(3, 256)),
            &[],
            None,
        );
        g.push(gl);
        c.push(cl);
    }
    GeoPyramid { geo: g, corr: c }
}

/// The per-iteration lookup `Combined_Geo_Encoding_Volume.__call__(disp, coords)`:
/// every level's cost-volume and correlation samples around the current disparity
/// `disp` `[B, 1, H, W]`, concatenated to
/// `[B, levels * (C + 1) * (2 * radius + 1), H, W]` in PyTorch's channel order
/// (`geo_0, corr_0, geo_1, corr_1, …`). `coords` is the column index, so it is not an
/// input. With FoundationStereo's `C = 28`, two levels and radius 4 that is 522
/// channels.
pub fn geo_lookup<D: FloatBytes + Send + Sync + 'static>(
    pyramid: &GeoPyramid,
    disp: &SymTensor,
    radius: usize,
) -> SymTensor {
    let levels: Vec<SymTensor> = pyramid
        .geo
        .iter()
        .zip(&pyramid.corr)
        .enumerate()
        .map(|(i, (g, c))| {
            g.record_custom(
                CustomData::new(GeoLookupOp::<D>::new(i as u32, radius, 32)),
                &[c, disp],
                None,
            )
        })
        .collect();
    let refs: Vec<&SymTensor> = levels.iter().collect();
    cat(&refs)
}
