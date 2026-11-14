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

/// `dir` is relative to `tests/fixtures/`, or an absolute path for fixtures kept
/// outside the repository (e.g. ones built from a real checkpoint).
fn fixture_dir(dir: &str) -> String {
    if dir.starts_with('/') {
        dir.to_string()
    } else {
        format!("{}/tests/fixtures/{dir}", env!("CARGO_MANIFEST_DIR"))
    }
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
/// `dir`, and asserts every named output is within `tol` of PyTorch. `build` returns
/// the outputs to check; output `name` is compared against `expected_<name>.bin`, or
/// `expected.bin` for the empty name. Returns the worst max-abs difference.
pub fn check_block(
    label: &str,
    dir: &str,
    inputs: &[(&str, [usize; 4])],
    build: impl FnOnce(&[SymTensor]) -> Vec<(&'static str, SymTensor)>,
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
    let outputs = build(&syms);

    // RuntimeModel lowers the graph the same way internally; lowering again here gives
    // the graph-node -> DAG-node mapping it does not expose, so outputs that feed other
    // nodes (not just terminals) can be read back.
    let lowering = TritonLowering::new();
    let (_, graph_to_dag) =
        lowering.lower_with_mapping(&graph.borrow(), LoweringMode::Inference)?;

    let device = runtime::open()?;
    let target = runtime::default_target(&device)?;
    let batch = inputs[0].1[0];
    let mut model = RuntimeModel::compile(
        &device,
        &graph.borrow(),
        &lowering,
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

    let out_idx: Vec<usize> = outputs
        .iter()
        .map(|(_, t)| graph_to_dag[t.node_id])
        .collect();
    let data: Vec<Vec<f32>> = inputs
        .iter()
        .map(|(name, _)| fixture(dir, &format!("{name}.bin")))
        .collect();
    let refs: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();
    let got = model.run(&device, &refs, &out_idx)?;

    let mut worst_all = 0.0f32;
    for ((name, _), out) in outputs.iter().zip(&got) {
        let file = if name.is_empty() {
            "expected.bin".to_string()
        } else {
            format!("expected_{name}.bin")
        };
        let want = fixture(dir, &file);
        let tag = if name.is_empty() {
            label.to_string()
        } else {
            format!("{label}/{name}")
        };
        assert_eq!(out.len(), want.len(), "{tag}: output length");
        let mut max = 0.0f32;
        let mut worst = 0;
        for (i, (g, w)) in out.iter().zip(&want).enumerate() {
            assert!(g.is_finite(), "{tag}: non-finite output at {i}");
            let d = (g - w).abs();
            if d > max {
                max = d;
                worst = i;
            }
        }
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        println!(
            "{tag}: max |{} - torch| = {max:e} (max |torch| {scale:.3}, relative {:e})",
            runtime::BACKEND_NAME,
            max / scale.max(f32::MIN_POSITIVE)
        );
        assert!(
            max < tol,
            "{tag}: max |diff| {max:e} at {worst} (got {} want {})",
            out[worst],
            want[worst]
        );
        worst_all = worst_all.max(max);
    }
    println!("{label}: {} weight slots loaded", slots.len());
    Ok(worst_all)
}
