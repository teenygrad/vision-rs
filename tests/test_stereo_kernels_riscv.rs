// Compile each stereo kernel for RISC-V (SpacemiT K3) and assert the emitted assembly
// is real vectorised code, not a placeholder.

#![cfg(all(feature = "riscv", not(feature = "cuda")))]

use dotenv::dotenv;
use teeny_core::device::program::Kernel;
use teeny_riscv::compiler::{
    compile_kernel,
    target::{Capability, Target},
};
use vision_rs::models::stereo::kernels::bilinear_sampler_1d::BilinearSampler1dForward;
use vision_rs::models::stereo::kernels::correlation::CorrelationForward;
use vision_rs::models::stereo::kernels::l2_normalize::L2NormalizeForward;

fn check(name: &str, kernel: &impl Kernel) -> Result<(), Box<dyn std::error::Error>> {
    let target = Target::new(Capability::SpacemitK3);
    let obj = compile_kernel(kernel, &target, true, false)?;
    let asm = std::fs::read_to_string(std::path::Path::new(&obj).with_extension("s"))?;
    let vector = asm
        .lines()
        .filter(|l| l.trim_start().starts_with('v'))
        .count();
    let vsetvli = asm.matches("vsetvli").count();
    println!(
        "{name}: asm lines={} vector={vector} vsetvli={vsetvli}",
        asm.lines().count()
    );
    assert!(
        asm.contains(name),
        "entry point {name} missing from emitted assembly"
    );
    assert!(
        vector > 0,
        "{name} emitted no vector instructions — backend may have produced a placeholder"
    );
    Ok(())
}

#[test]
fn test_stereo_kernels_lower_to_rvv() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    check(
        "bilinear_sampler_1d_forward",
        &BilinearSampler1dForward::<f32>::new(16),
    )?;
    check("l2_normalize_forward", &L2NormalizeForward::<f32>::new(16))?;
    check("correlation_forward", &CorrelationForward::<f32>::new(16))?;
    Ok(())
}
