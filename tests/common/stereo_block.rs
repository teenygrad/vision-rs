// Shared harness for FoundationStereo refinement-head block tests.
//
// A block is traced into a graph over named inputs, compiled with `RuntimeModel` on
// whichever backend teeny-runtime was built for, loaded from PyTorch weights by
// `state_dict` key, run, and compared against the PyTorch output.
//
// Fixture directory layout (`tests/fixtures/<dir>/`):
//   <input>.bin        one per input, little-endian f32, in the order given
//   w__<key>.bin       one per `state_dict` entry
//   expected.bin       the PyTorch output
//
// Every graph weight slot must have a fixture and every weight fixture a slot, so a
// naming mismatch fails loudly instead of leaving a zero-initialised weight.

use std::collections::HashSet;

use teeny_core::graph::{DtypeRepr, SymTensor};
use teeny_core::model::LoweringMode;
use teeny_kernels::graph::TritonLowering;
use teeny_test::model::{RuntimeModel, runtime};
use vision_rs::models::stereo::update::ops::{input_like, torch_key};

fn fixture_dir(dir: &str) -> String {
    format!("{}/tests/fixtures/{dir}", env!("CARGO_MANIFEST_DIR"))
}

pub fn fixture(dir: &str, file: &str) -> Vec<f32> {
    let p = format!("{}/{file}", fixture_dir(dir));
    std::fs::read(&p)
        .unwrap_or_else(|e| panic!("missing fixture {p}: {e}"))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn weight_keys(dir: &str) -> HashSet<String> {
    std::fs::read_dir(fixture_dir(dir))
        .unwrap()
        .filter_map(|e| {
            let name = e.ok()?.file_name().into_string().ok()?;
            Some(name.strip_prefix("w__")?.strip_suffix(".bin")?.to_string())
        })
        .collect()
}

/// Traces `build` over `inputs` (name, `[B, C, H, W]`), runs it against the fixtures in
/// `dir`, and asserts the max absolute difference is below `tol`. Returns that maximum.
pub fn check_block(
    label: &str,
    dir: &str,
    inputs: &[(&str, [usize; 4])],
    build: impl FnOnce(&[SymTensor]),
    tol: f32,
) -> anyhow::Result<f32> {
    dotenv::dotenv().ok();
    let shape = |s: [usize; 4]| s.iter().map(|&d| Some(d)).collect::<Vec<_>>();

    // Inputs are created, and therefore fed, in the order given.
    let (first, graph) = SymTensor::input(DtypeRepr::F32, shape(inputs[0].1));
    let mut syms = vec![first.clone()];
    for (_, s) in &inputs[1..] {
        syms.push(input_like(&first, shape(*s)));
    }
    build(&syms);

    let device = runtime::open()?;
    let target = runtime::default_target(&device)?;
    let batch = inputs[0].1[0];
    let mut model = RuntimeModel::compile(
        &device,
        &graph.borrow(),
        &TritonLowering::new(),
        LoweringMode::Inference,
        &target,
        batch,
        false,
    )?;

    let mut remaining = weight_keys(dir);
    let slots: Vec<(String, usize, usize)> = model.param_info_named().collect();
    for (slot, node, idx) in &slots {
        let key = torch_key(slot);
        assert!(
            remaining.remove(&key),
            "{label}: graph slot {slot} ({key}) has no PyTorch weight"
        );
        model.load_param_f32(*node, *idx, &fixture(dir, &format!("w__{key}.bin")))?;
    }
    assert!(
        remaining.is_empty(),
        "{label}: PyTorch weights not loaded into the graph: {remaining:?}"
    );

    let out_idx = *model
        .terminal_node_indices_sorted_by_size()
        .last()
        .expect("graph has an output");
    let data: Vec<Vec<f32>> = inputs
        .iter()
        .map(|(name, _)| fixture(dir, &format!("{name}.bin")))
        .collect();
    let refs: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();
    let out = model.run(&device, &refs, &[out_idx])?.remove(0);

    let want = fixture(dir, "expected.bin");
    assert_eq!(out.len(), want.len(), "{label}: output length");
    let mut max = 0.0f32;
    let mut worst = 0;
    for (i, (g, w)) in out.iter().zip(&want).enumerate() {
        assert!(g.is_finite(), "{label}: non-finite output at {i}");
        let d = (g - w).abs();
        if d > max {
            max = d;
            worst = i;
        }
    }
    println!(
        "{label}: {} weight slots, max |{} - torch| = {max:e}",
        slots.len(),
        runtime::BACKEND_NAME
    );
    assert!(
        max < tol,
        "{label}: max |diff| {max:e} at {worst} (got {} want {})",
        out[worst],
        want[worst]
    );
    Ok(max)
}
