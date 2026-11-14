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

/// A block traced into a graph, compiled with `RuntimeModel` and loaded with PyTorch
/// weights, ready to run any number of times.
pub struct Block<'d> {
    label: String,
    device: &'d runtime::Device<'d>,
    model: RuntimeModel<'d>,
    out_idx: Vec<usize>,
    /// Output names, in the order `run` returns them.
    pub outputs: Vec<&'static str>,
}

impl<'d> Block<'d> {
    /// Traces `build` over `inputs` (name, shape, batch first; created, and therefore
    /// fed, in this order), compiles it on `device` and loads weights from
    /// `weights_dir`'s `w__<state_dict key>.bin` files. When the graph has weight slots,
    /// every slot must have a file and every file a slot. `build` returns the outputs
    /// `run` should read back, by name.
    pub fn new(
        device: &'d runtime::Device<'d>,
        label: &str,
        weights_dir: &str,
        inputs: &[(&str, &[usize])],
        build: impl FnOnce(&[SymTensor]) -> Vec<(&'static str, SymTensor)>,
    ) -> anyhow::Result<Self> {
        dotenv::dotenv().ok();
        let shape = |s: &[usize]| s.iter().map(|&d| Some(d)).collect::<Vec<_>>();
        let (first, graph) = SymTensor::input(DtypeRepr::F32, shape(inputs[0].1));
        let mut syms = vec![first.clone()];
        for (_, s) in &inputs[1..] {
            syms.push(input_like(&first, shape(s)));
        }
        let outputs = build(&syms);

        // RuntimeModel lowers the graph the same way internally; lowering again here
        // gives the graph-node -> DAG-node mapping it does not expose, so outputs that
        // feed other nodes (not just terminals) can be read back.
        let lowering = TritonLowering::new();
        let (_, graph_to_dag) =
            lowering.lower_with_mapping(&graph.borrow(), LoweringMode::Inference)?;
        let target = runtime::default_target(device)?;
        let mut model = RuntimeModel::compile(
            device,
            &graph.borrow(),
            &lowering,
            LoweringMode::Inference,
            &target,
            inputs[0].1[0],
            false,
        )?;

        let slots: Vec<(String, usize, usize)> = model.param_info_named().collect();
        if !slots.is_empty() {
            let mut remaining = weight_keys(weights_dir);
            for (slot, node, idx) in &slots {
                let key = torch_key(slot);
                assert!(
                    remaining.remove(&key),
                    "{label}: graph slot {slot} ({key}) has no PyTorch weight"
                );
                model.load_param_f32(
                    *node,
                    *idx,
                    &fixture(weights_dir, &format!("w__{key}.bin")),
                )?;
            }
            assert!(
                remaining.is_empty(),
                "{label}: PyTorch weights not loaded into the graph: {remaining:?}"
            );
            println!("{label}: {} weight slots loaded", slots.len());
        }

        Ok(Self {
            label: label.to_string(),
            device,
            model,
            out_idx: outputs
                .iter()
                .map(|(_, t)| graph_to_dag[t.node_id])
                .collect(),
            outputs: outputs.iter().map(|(n, _)| *n).collect(),
        })
    }

    /// Runs one forward pass on `inputs` (in creation order), returning the outputs
    /// `build` named.
    pub fn run(&mut self, inputs: &[&[f32]]) -> anyhow::Result<Vec<Vec<f32>>> {
        self.model
            .run(self.device, inputs, &self.out_idx)
            .map_err(|e| e.context(format!("running {}", self.label)))
    }
}

/// Asserts `got` is within `tol` of `want` (max abs difference); prints and returns it.
pub fn compare(tag: &str, got: &[f32], want: &[f32], tol: f32) -> f32 {
    assert_eq!(got.len(), want.len(), "{tag}: output length");
    let mut max = 0.0f32;
    let mut worst = 0;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
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
        got[worst],
        want[worst]
    );
    max
}

/// Traces `build` over `inputs`, runs it once on the fixtures in `dir` (inputs as
/// `<name>.bin`, weights as `w__<key>.bin`), and asserts every named output is within
/// `tol` of PyTorch: output `name` against `expected_<name>.bin`, or `expected.bin` for
/// the empty name. Returns the worst max-abs difference.
pub fn check_block(
    label: &str,
    dir: &str,
    inputs: &[(&str, &[usize])],
    build: impl FnOnce(&[SymTensor]) -> Vec<(&'static str, SymTensor)>,
    tol: f32,
) -> anyhow::Result<f32> {
    let device = runtime::open()?;
    let mut block = Block::new(&device, label, dir, inputs, build)?;
    let data: Vec<Vec<f32>> = inputs
        .iter()
        .map(|(name, _)| fixture(dir, &format!("{name}.bin")))
        .collect();
    let refs: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();
    let got = block.run(&refs)?;

    let mut worst = 0.0f32;
    for (name, out) in block.outputs.clone().iter().zip(&got) {
        let (file, tag) = if name.is_empty() {
            ("expected.bin".to_string(), label.to_string())
        } else {
            (format!("expected_{name}.bin"), format!("{label}/{name}"))
        };
        worst = worst.max(compare(&tag, out, &fixture(dir, &file), tol));
    }
    Ok(worst)
}
