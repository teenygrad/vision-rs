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

use teeny_core::{
    dtype::Float,
    graph::{Op, SymTensor},
    name_scope::name_scope,
    nn::{Layer, pool::MaxPool2d},
};

use super::{
    concat::concat,
    conv::{conv, conv_bn},
};

fn elem_add(a: SymTensor, b: SymTensor) -> SymTensor {
    let shape = a.shape.clone();
    let node_id =
        a.graph
            .borrow_mut()
            .add_node(Op::Add, vec![a.node_id, b.node_id], a.dtype, shape.clone());
    SymTensor {
        node_id,
        graph: a.graph.clone(),
        dtype: a.dtype,
        shape,
    }
}

/// Spatial Pyramid Pooling - Fast (SPPF).
///
/// Matches `ultralytics.nn.modules.block.SPPF(c1, c2, k=5, n=3, shortcut)`:
///   cv1:      Conv(c1, c//2, 1, 1, act=False) — Conv+BN only, no SiLU
///   pool:     MaxPool2d(k=5, stride=1, padding=2) applied 3 times
///   cv2:      Conv(4 * c//2, c2, 1, 1) — Conv+BN+SiLU
///   shortcut: residual add (output + input) when shortcut=true and c_in==c_out
///             (YOLO26 passes shortcut=true; default is false)
pub fn sppf<D: Float>(
    c_in: usize,
    c_out: usize,
    shortcut: bool,
) -> impl Fn(SymTensor) -> SymTensor {
    let c = c_in / 2;
    let add = shortcut && c_in == c_out;
    let cv1 = conv_bn::<D>(c_in, c, 1, 1, 1);
    let cv2 = conv::<D>(4 * c, c_out, 1, 1);
    let pool = || MaxPool2d::<D, SymTensor, SymTensor, 4>::with_padding((5, 5), (1, 1), (2, 2));

    move |x: SymTensor| {
        let y = {
            let _g = name_scope("cv1");
            cv1(x.clone())
        };
        let p1 = pool().call(y.clone());
        let p2 = pool().call(p1.clone());
        let p3 = pool().call(p2.clone());
        let cat = concat()(vec![y, p1, p2, p3]);
        let out = {
            let _g = name_scope("cv2");
            cv2(cat)
        };
        if add { elem_add(out, x) } else { out }
    }
}
