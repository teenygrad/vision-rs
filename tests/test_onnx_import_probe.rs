// Probe: can teeny-onnx import FoundationStereo's ONNX graph?
//
// Reports success with node count, or the first op it rejects. Run with:
//   FS_ONNX=<path> cargo test --no-default-features --features riscv \
//       --test test_onnx_import_probe -- --nocapture --ignored

#![cfg(all(feature = "riscv", not(feature = "cuda")))]

#[test]
#[ignore = "needs FS_ONNX pointing at a FoundationStereo .onnx export"]
fn probe_onnx_import() {
    let path = std::env::var("FS_ONNX").expect("set FS_ONNX to a FoundationStereo .onnx");
    println!("importing {path}");
    let t0 = std::time::Instant::now();
    match teeny_onnx::Onnx::from_path(&path) {
        Ok(graph) => {
            println!(
                "IMPORT OK in {:.1}s — {} graph nodes",
                t0.elapsed().as_secs_f32(),
                graph.nodes.len()
            );
        }
        Err(e) => {
            println!("IMPORT FAILED after {:.1}s", t0.elapsed().as_secs_f32());
            println!("  {e}");
        }
    }
}
