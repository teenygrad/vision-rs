// Graph-level test for the stereo cost-volume geometry.
//
// Traces the correlation-volume builder and asserts the recorded graph: op identity,
// node count, and inferred shapes. This is the same level at which the YOLO block
// tests (e.g. test_sppf_yolo26.rs) validate their graphs.

use teeny_core::graph::{DtypeRepr, Op, SymTensor};
use vision_rs::models::stereo::geometry::correlation_volume;

const B: usize = 2;
const C: usize = 16;
const H: usize = 5;
const W1: usize = 7;
const W2: usize = 9;

#[test]
fn test_correlation_volume_graph() {
    let (f1, graph) = SymTensor::input(DtypeRepr::F32, vec![Some(B), Some(C), Some(H), Some(W1)]);

    // fmap2 shares fmap1's graph so both feed one correlation node.
    let shape2 = vec![Some(B), Some(C), Some(H), Some(W2)];
    let node_id = graph
        .borrow_mut()
        .add_node(Op::Input, vec![], DtypeRepr::F32, shape2.clone());
    let f2 = SymTensor {
        node_id,
        graph: f1.graph.clone(),
        dtype: f1.dtype,
        shape: shape2,
    };

    let out = correlation_volume::<f32>(&f1, &f2, 32);

    // Output shape: [B, H, W1, W2].
    assert_eq!(
        out.shape,
        vec![Some(B), Some(H), Some(W1), Some(W2)],
        "correlation volume shape"
    );

    let graph = out.graph.borrow();
    let names: Vec<String> = graph
        .nodes
        .iter()
        .filter_map(|n| match &n.op {
            Op::Custom { data } => Some(data.0.name().to_string()),
            _ => None,
        })
        .collect();
    assert_eq!(
        names,
        vec![
            "stereo.l2_normalize".to_string(),
            "stereo.l2_normalize".to_string(),
            "stereo.correlation".to_string(),
        ],
        "expected both feature maps normalised, then one correlation"
    );
}
