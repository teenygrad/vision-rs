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

use teeny_core::graph::{Op, SymTensor};

/// Nearest-neighbour 2-D upsample.
///
/// Output shape: `[B, C, H*scale_h, W*scale_w]`.
pub fn upsample(scale_h: usize, scale_w: usize) -> impl Fn(SymTensor) -> SymTensor {
    move |x| {
        let op = Op::UpsampleNearest2d { scale_h, scale_w };
        let shape = vec![
            x.shape[0],
            x.shape[1],
            x.shape[2].map(|h| h * scale_h),
            x.shape[3].map(|w| w * scale_w),
        ];
        let node_id = x
            .graph
            .borrow_mut()
            .add_node(op, vec![x.node_id], x.dtype, shape.clone());
        SymTensor {
            node_id,
            graph: x.graph.clone(),
            dtype: x.dtype,
            shape,
        }
    }
}
