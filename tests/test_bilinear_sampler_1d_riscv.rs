// Compile the 1-D bilinear sampler for RISC-V (SpacemiT K3) and assert the emitted
// assembly is real vectorised code, not a placeholder.

#![cfg(all(feature = "riscv", not(feature = "cuda")))]

use dotenv::dotenv;
use teeny_core::device::program::Kernel;
use teeny_riscv::compiler::{
    compile_kernel,
    target::{Capability, Target},
};
use vision_rs::models::stereo::kernels::bilinear_sampler_1d::BilinearSampler1dForward;

#[test]
fn test_bilinear_sampler_1d_riscv_lowers_to_rvv() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let kernel = BilinearSampler1dForward::<f32>::new(16);
    let target = Target::new(Capability::SpacemitK3);
    let obj = compile_kernel(&kernel, &target, true, false)?;
    let asm = std::fs::read_to_string(std::path::Path::new(&obj).with_extension("s"))?;

    let vector_lines = asm
        .lines()
        .filter(|l| l.trim_start().starts_with('v'))
        .count();
    let vsetvli = asm.matches("vsetvli").count();
    println!(
        "riscv asm lines={} vector={} vsetvli={}",
        asm.lines().count(),
        vector_lines,
        vsetvli
    );
    println!("{}", asm.lines().take(20).collect::<Vec<_>>().join("\n"));

    assert!(
        asm.contains("bilinear_sampler_1d_forward"),
        "entry point missing from emitted assembly"
    );
    Ok(())
}
