// SelectiveConvGRU (FoundationStereo `core/update.py`) traced into a graph, compiled with
// `RuntimeModel`, loaded by `state_dict` name and checked against PyTorch.
//
// Reduced width (hidden 8, input 8 + 8) on a 6x10 map, so the committed weights stay
// small; the block's structure is identical to the full model's. Fixtures come from the
// real `SelectiveConvGRU` class with seeded random weights scaled up so both GRUs and the
// attention blend affect the output.
//
// Runs on whichever backend teeny-runtime was built for: CUDA, or RISC-V under qemu
// (`--no-default-features --features qemu`). The RISC-V runtime looks for `qemu-riscv64`
// on PATH; with only the static build installed, set
// `TEENYC_QEMU_RISCV64=/usr/bin/qemu-riscv64-static`.

#![cfg(any(feature = "cuda", feature = "qemu"))]

use std::collections::HashSet;

use dotenv::dotenv;
use teeny_core::compiler::Target as _;
use teeny_core::graph::{DtypeRepr, SymTensor};
use teeny_core::model::LoweringMode;
use teeny_kernels::graph::TritonLowering;
use teeny_test::model::{RuntimeModel, runtime};
use vision_rs::models::stereo::update::{gru::selective_conv_gru, ops::input_like};

const HID: usize = 8;
const X1: usize = 8;
const X2: usize = 8;
const B: usize = 1;
const H: usize = 6;
const W: usize = 10;

fn fixture(rel: &str) -> Vec<f32> {
    let p = format!(
        "{}/tests/fixtures/selective_conv_gru/{rel}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&p)
        .unwrap_or_else(|e| panic!("missing fixture {p}: {e}"))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// Weight fixtures are `w__<state_dict key>.bin`.
fn weight_keys() -> HashSet<String> {
    let dir = format!(
        "{}/tests/fixtures/selective_conv_gru",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| {
            let name = e.ok()?.file_name().into_string().ok()?;
            Some(name.strip_prefix("w__")?.strip_suffix(".bin")?.to_string())
        })
        .collect()
}

#[test]
fn test_selective_conv_gru_matches_pytorch() -> anyhow::Result<()> {
    dotenv().ok();

    // Inputs are created in the order they will be fed.
    let shape = |c: usize| vec![Some(B), Some(c), Some(H), Some(W)];
    let (att, graph) = SymTensor::input(DtypeRepr::F32, shape(1));
    let h = input_like(&att, shape(HID));
    let x1 = input_like(&att, shape(X1));
    let x2 = input_like(&att, shape(X2));
    let _y = selective_conv_gru::<f32>(HID, X1 + X2)(&att, &h, &[&x1, &x2]);

    let device = runtime::open()?;
    let target = runtime::default_target(&device)?;
    println!(
        "backend {} ({})",
        runtime::BACKEND_NAME,
        target.target_cpu().unwrap_or_default()
    );
    let mut model = RuntimeModel::compile(
        &device,
        &graph.borrow(),
        &TritonLowering::new(),
        LoweringMode::Inference,
        &target,
        B,
        false,
    )?;

    // Every graph weight slot must have a fixture, and every fixture a slot.
    let mut expected_keys = weight_keys();
    let slots: Vec<(String, usize, usize)> = model.param_info_named().collect();
    for (key, node, slot) in &slots {
        assert!(
            expected_keys.remove(key),
            "graph slot {key} has no PyTorch weight"
        );
        model.load_param_f32(*node, *slot, &fixture(&format!("w__{key}.bin")))?;
    }
    assert!(
        expected_keys.is_empty(),
        "PyTorch weights not loaded into the graph: {expected_keys:?}"
    );

    let out_idx = *model
        .terminal_node_indices_sorted_by_size()
        .last()
        .expect("graph has an output");
    let inputs = [
        fixture("att.bin"),
        fixture("h.bin"),
        fixture("x1.bin"),
        fixture("x2.bin"),
    ];
    let refs: Vec<&[f32]> = inputs.iter().map(Vec::as_slice).collect();
    let out = model.run(&device, &refs, &[out_idx])?.remove(0);

    let want = fixture("expected.bin");
    assert_eq!(out.len(), want.len());
    let mut max = 0.0f32;
    for (i, (g, w)) in out.iter().zip(&want).enumerate() {
        assert!(g.is_finite(), "non-finite at {i}");
        max = max.max((g - w).abs());
    }
    println!(
        "selective_conv_gru: {} weight slots loaded, max |{} - torch| = {max:e}",
        slots.len(),
        runtime::BACKEND_NAME
    );
    assert!(max < 1e-5, "max |diff| {max:e}");
    Ok(())
}
