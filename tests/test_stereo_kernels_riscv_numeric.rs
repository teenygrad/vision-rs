// Numerical validation of the stereo kernels on RISC-V, under qemu-riscv64.
//
// The stage-1 kernels were validated numerically on CUDA and only *compiled* for
// RISC-V, which proves they lower to RVV but not that they compute the right answer
// there. That gap matters here more than usual: several kernels are shaped around
// teenyc ICEs (index arithmetic in float, additive eps instead of a clamp), and those
// workarounds could behave differently on a different backend.
//
// This runs each kernel on RVV against the *same PyTorch fixtures* the CUDA tests use,
// so CUDA and RISC-V are compared to one reference rather than to each other.
//
// teeny_test::riscv::qemu only runs pointwise kernels — signature
// `(x_ptr, y_ptr, n_elements)` — and none of these are pointwise. So this carries its
// own harness: a C shim, generated per kernel signature, that dlopens the kernel .so
// and calls it once per CTA with the launch parameters a Triton launcher would pass.
//
// The shim is generated rather than written once because RISC-V LP64D passes floats in
// a separate register file from integers: an ABI-exact prototype is required, and
// varargs would pass them differently.

#![cfg(all(feature = "riscv", not(feature = "cuda")))]

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::process::Command;

use dotenv::dotenv;
use teeny_core::device::program::Kernel;
use teeny_riscv::compiler::{
    compile_kernel,
    target::{Capability, Target},
};

use vision_rs::models::stereo::kernels::{
    bilinear_sampler_1d::BilinearSampler1dForward,
    conv_transpose2d_k4s2p1::ConvTranspose2dK4s2p1Forward, correlation::CorrelationForward,
    interpolate_bilinear::InterpolateBilinearForward, l2_normalize::L2NormalizeForward,
    unfold3x3::Unfold3x3Forward,
};

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

const CC: &str = "riscv64-linux-gnu-gcc";
const QEMU: &str = "qemu-riscv64-static";

fn scratch() -> PathBuf {
    let d = std::env::var("TEENYC_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/teenyc_cache"));
    std::fs::create_dir_all(&d).expect("create cache dir");
    d
}

/// Sysroot for the Debian/Ubuntu cross toolchain, whose libc lives under
/// `/usr/<multiarch>` rather than anywhere `-print-sysroot` reports.
fn sysroot() -> PathBuf {
    let out = Command::new(CC)
        .arg("-print-multiarch")
        .output()
        .unwrap_or_else(|e| panic!("{CC} not found ({e}); install gcc-riscv64-linux-gnu"));
    let multiarch = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let p = PathBuf::from("/usr").join(&multiarch);
    assert!(p.is_dir(), "no RISC-V sysroot at {}", p.display());
    p
}

/// A kernel's argument shape: `ptrs` buffers, then `ints` i32s, then `floats` f32s.
/// The six launch parameters (pid x/y/z, grid x/y/z) are appended by the shim.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Sig {
    ptrs: usize,
    ints: usize,
    floats: usize,
}

fn shim_source(sig: Sig) -> String {
    let mut params: Vec<String> = (0..sig.ptrs).map(|_| "float *".to_string()).collect();
    params.extend((0..sig.ints).map(|_| "int32_t".to_string()));
    params.extend((0..sig.floats).map(|_| "float".to_string()));
    params.extend(
        [
            "int32_t", "int32_t", "int32_t", "uint32_t", "uint32_t", "uint32_t",
        ]
        .iter()
        .map(|s| s.to_string()),
    );

    let mut call: Vec<String> = (0..sig.ptrs).map(|i| format!("bufs[{i}]")).collect();
    call.extend((0..sig.ints).map(|i| format!("ints[{i}]")));
    call.extend((0..sig.floats).map(|i| format!("flts[{i}]")));
    call.push("pid".to_string());
    call.push("0".to_string());
    call.push("0".to_string());
    call.push("grid".to_string());
    call.push("1".to_string());
    call.push("1".to_string());

    format!(
        r#"
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef void (*kfn)({params});

static float *slurp(const char *path, size_t *count) {{
    FILE *f = fopen(path, "rb");
    if (!f) {{ perror(path); return NULL; }}
    fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
    if (n < 0 || n % 4 != 0) {{ fprintf(stderr, "%s: bad size %ld\n", path, n); fclose(f); return NULL; }}
    *count = (size_t)n / 4;
    float *d = malloc(n > 0 ? (size_t)n : 1);
    if (!d || fread(d, 4, *count, f) != *count) {{ fprintf(stderr, "%s: short read\n", path); fclose(f); return NULL; }}
    fclose(f); return d;
}}

static int spit(const char *path, const float *d, size_t count) {{
    FILE *f = fopen(path, "wb");
    if (!f) {{ perror(path); return 0; }}
    int ok = fwrite(d, 4, count, f) == count;
    fclose(f); return ok;
}}

// argv: so symbol grid  buf0..buf{nptr}  int0..int{nint}  flt0..flt{nflt}
int main(int argc, char **argv) {{
    int expect = 4 + {nptr} + {nint} + {nflt};
    if (argc != expect) {{ fprintf(stderr, "argc %d != %d\n", argc, expect); return 2; }}
    void *h = dlopen(argv[1], RTLD_NOW);
    if (!h) {{ fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }}
    kfn kernel = (kfn)dlsym(h, argv[2]);
    if (!kernel) {{ fprintf(stderr, "dlsym(%s): %s\n", argv[2], dlerror()); return 1; }}
    long grid = atol(argv[3]);

    int a = 4;
    float *bufs[{nptr_or_1}]; size_t counts[{nptr_or_1}]; const char *paths[{nptr_or_1}];
    for (int i = 0; i < {nptr}; i++) {{ paths[i] = argv[a++]; bufs[i] = slurp(paths[i], &counts[i]); if (!bufs[i]) return 1; }}
    int32_t ints[{nint_or_1}];
    for (int i = 0; i < {nint}; i++) ints[i] = (int32_t)atol(argv[a++]);
    float flts[{nflt_or_1}];
    for (int i = 0; i < {nflt}; i++) flts[i] = (float)atof(argv[a++]);

    for (long pid = 0; pid < grid; pid++) kernel({call});

    // Every buffer is written back so the caller can read whichever one is the output.
    for (int i = 0; i < {nptr}; i++) if (!spit(paths[i], bufs[i], counts[i])) return 1;
    return 0;
}}
"#,
        params = params.join(", "),
        call = call.join(", "),
        nptr = sig.ptrs,
        nint = sig.ints,
        nflt = sig.floats,
        nptr_or_1 = sig.ptrs.max(1),
        nint_or_1 = sig.ints.max(1),
        nflt_or_1 = sig.floats.max(1),
    )
}

/// Cross-compiles the shim for `sig`, cached by a hash of its source.
fn shim(sig: Sig) -> PathBuf {
    let src = shim_source(sig);
    let mut h = DefaultHasher::new();
    src.hash(&mut h);
    let bin = scratch().join(format!("stereo_shim_{:x}", h.finish()));
    if bin.is_file() {
        return bin;
    }
    let src_path = scratch().join(format!("stereo_shim_{:x}.c", h.finish()));
    std::fs::write(&src_path, &src).expect("write shim source");
    let out = Command::new(CC)
        .args(["-O2", "-o"])
        .arg(&bin)
        .arg(&src_path)
        .arg("-ldl")
        .output()
        .expect("run cross gcc");
    assert!(
        out.status.success(),
        "shim compile failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    bin
}

/// The dlopen-able artifact for a compiled kernel.
///
/// teenyc emits two files. The one named `.o` is despite its extension already an ELF
/// *shared object* and exports the entry point as a dynamic symbol; the one named `.so`
/// exports only CRT symbols and is useless to `dlsym`. So the `.o` is what gets opened.
fn shared_object(artifact: &str) -> PathBuf {
    let obj = PathBuf::from(artifact).with_extension("o");
    assert!(obj.is_file(), "no kernel object at {}", obj.display());
    obj
}

/// Compiles `kernel` for the K3, then runs it under qemu once per CTA.
/// `bufs` are read in and written back in place, so the output buffer is updated.
fn run(
    name: &str,
    kernel: &impl Kernel,
    grid: usize,
    bufs: &mut [&mut Vec<f32>],
    ints: &[i32],
    floats: &[f32],
) {
    let target = Target::new(Capability::SpacemitK3);
    let artifact = compile_kernel(kernel, &target, true, false)
        .unwrap_or_else(|e| panic!("{name}: riscv compile failed: {e}"));
    let so = shared_object(&artifact);

    let sig = Sig {
        ptrs: bufs.len(),
        ints: ints.len(),
        floats: floats.len(),
    };
    let shim_bin = shim(sig);

    let mut paths = Vec::new();
    for (i, b) in bufs.iter().enumerate() {
        let p = scratch().join(format!("{name}_buf{i}_{}.bin", std::process::id()));
        let bytes: Vec<u8> = b.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&p, bytes).expect("write buffer");
        paths.push(p);
    }

    let mut cmd = Command::new(QEMU);
    cmd.args(["-cpu", "max"])
        .arg("-L")
        .arg(sysroot())
        .arg(&shim_bin)
        .arg(&so)
        .arg(format!("{name}_entry_point"))
        .arg(grid.to_string());
    for p in &paths {
        cmd.arg(p);
    }
    for i in ints {
        cmd.arg(i.to_string());
    }
    for f in floats {
        cmd.arg(format!("{f:.9}"));
    }

    let out = cmd.output().expect("run qemu");
    assert!(
        out.status.success(),
        "{name}: qemu exited {}: {}{}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    for (b, p) in bufs.iter_mut().zip(&paths) {
        let bytes = std::fs::read(p).expect("read back buffer");
        let vals: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        assert_eq!(vals.len(), b.len(), "{name}: buffer length changed");
        b.clear();
        b.extend_from_slice(&vals);
        let _ = std::fs::remove_file(p);
    }
}

fn fixture(rel: &str) -> Vec<f32> {
    let p = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    std::fs::read(&p)
        .unwrap_or_else(|e| panic!("missing fixture {p}: {e}"))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Compares against the same PyTorch reference the CUDA tests use.
fn check(name: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let mut max = 0.0f32;
    let mut worst = 0usize;
    for i in 0..got.len() {
        assert!(got[i].is_finite(), "{name}: non-finite at {i}: {}", got[i]);
        let d = (got[i] - want[i]).abs();
        if d > max {
            max = d;
            worst = i;
        }
    }
    println!("{name}: max |rvv - torch| = {max:e}");
    assert!(
        max < tol,
        "{name}: max |rvv - torch| = {max:e} at {worst} (got {} want {})",
        got[worst],
        want[worst]
    );
}

// ---------------------------------------------------------------------------
// Tests — same fixtures, same tolerances as the CUDA tests
// ---------------------------------------------------------------------------

#[test]
fn rvv_bilinear_sampler_1d() {
    dotenv().ok();
    const N: usize = 6;
    const C: usize = 5;
    const W: usize = 11;
    const K: usize = 9;
    let mut img = fixture("bilinear_sampler_1d/img.bin");
    let mut coords = fixture("bilinear_sampler_1d/coords.bin");
    let want = fixture("bilinear_sampler_1d/expected.bin");
    let mut out = vec![0.0f32; N * C * K];

    run(
        "bilinear_sampler_1d_forward",
        &BilinearSampler1dForward::<f32>::new(16),
        N * C,
        &mut [&mut img, &mut coords, &mut out],
        &[C as i32, W as i32, K as i32],
        &[],
    );
    check("bilinear_sampler_1d", &out, &want, 1e-5);
}

#[test]
fn rvv_l2_normalize() {
    dotenv().ok();
    const B: usize = 2;
    const C: usize = 12;
    const S: usize = 15;
    let mut x = fixture("correlation/fmap1.bin");
    let want = fixture("correlation/n1.bin");
    let mut out = vec![0.0f32; B * C * S];

    run(
        "l2_normalize_forward",
        &L2NormalizeForward::<f32>::new(16),
        B * S,
        &mut [&mut x, &mut out],
        &[C as i32, S as i32],
        &[],
    );
    check("l2_normalize", &out, &want, 1e-6);
}

#[test]
fn rvv_correlation() {
    dotenv().ok();
    const B: usize = 2;
    const C: usize = 12;
    const H: usize = 3;
    const W1: usize = 5;
    const W2: usize = 7;
    let mut n1 = fixture("correlation/n1.bin");
    let mut n2 = fixture("correlation/n2.bin");
    let want = fixture("correlation/expected.bin");
    let mut out = vec![0.0f32; B * H * W1 * W2];

    run(
        "correlation_forward",
        &CorrelationForward::<f32>::new(16),
        B * H * W1,
        &mut [&mut n1, &mut n2, &mut out],
        &[C as i32, H as i32, W1 as i32, W2 as i32],
        &[],
    );
    check("correlation", &out, &want, 1e-5);
}

#[test]
fn rvv_unfold3x3() {
    dotenv().ok();
    const B: usize = 2;
    const C: usize = 3;
    const H: usize = 5;
    const W: usize = 7;
    const BLOCK: usize = 8;
    let mut x = fixture("unfold3x3/x.bin");
    let want = fixture("unfold3x3/expected.bin");
    let mut out = vec![0.0f32; B * C * 9 * H * W];

    let tiles = (H * W).div_ceil(BLOCK);
    run(
        "unfold3x3_forward",
        &Unfold3x3Forward::<f32>::new(BLOCK as i32),
        B * C * 9 * tiles,
        &mut [&mut x, &mut out],
        &[C as i32, H as i32, W as i32],
        &[],
    );
    // A pure gather: CUDA matched the reference exactly, so RVV should too.
    check("unfold3x3", &out, &want, 1e-6);
}

#[test]
fn rvv_interpolate_bilinear() {
    dotenv().ok();
    const B: usize = 2;
    const C: usize = 3;
    const IH: usize = 4;
    const IW: usize = 5;
    const OH: usize = 7;
    const OW: usize = 9;
    const BLOCK: usize = 16;

    for (align_corners, fx) in [
        (true, "interpolate_bilinear/expected_ac_true.bin"),
        (false, "interpolate_bilinear/expected_ac_false.bin"),
    ] {
        let mut x = fixture("interpolate_bilinear/x.bin");
        let want = fixture(fx);
        let mut out = vec![0.0f32; B * C * OH * OW];

        let m = |ins: usize, outs: usize| -> (f32, f32) {
            if align_corners {
                ((ins as f32 - 1.0) / (outs as f32 - 1.0), 0.0)
            } else {
                let s = ins as f32 / outs as f32;
                (s, 0.5 * s - 0.5)
            }
        };
        let (sy, oy) = m(IH, OH);
        let (sx, ox) = m(IW, OW);

        let tiles = (OH * OW).div_ceil(BLOCK);
        run(
            "interpolate_bilinear_forward",
            &InterpolateBilinearForward::<f32>::new(BLOCK as i32),
            B * C * tiles,
            &mut [&mut x, &mut out],
            &[C as i32, IH as i32, IW as i32, OH as i32, OW as i32],
            &[sy, oy, sx, ox],
        );
        check(
            &format!("interpolate_bilinear(align_corners={align_corners})"),
            &out,
            &want,
            1e-5,
        );
    }
}

#[test]
fn rvv_conv_transpose2d_k4s2p1() {
    dotenv().ok();
    const B: usize = 2;
    const CIN: usize = 3;
    const COUT: usize = 2;
    const IH: usize = 4;
    const IW: usize = 5;
    let mut x = fixture("conv_transpose2d_k4s2p1/x.bin");
    let mut w = fixture("conv_transpose2d_k4s2p1/w.bin");
    let mut bias = fixture("conv_transpose2d_k4s2p1/b.bin");
    let want = fixture("conv_transpose2d_k4s2p1/expected.bin");
    let mut out = vec![0.0f32; B * COUT * (IH * 2) * (IW * 2)];

    run(
        "conv_transpose2d_k4s2p1_forward",
        &ConvTranspose2dK4s2p1Forward::<f32>::new(4),
        B * COUT * (IH * 2) * (IW * 2),
        &mut [&mut x, &mut w, &mut bias, &mut out],
        &[CIN as i32, COUT as i32, IH as i32, IW as i32],
        &[],
    );
    check("conv_transpose2d_k4s2p1", &out, &want, 1e-5);
}
