"""Element-wise diff of `yolo26 validate --dump-outputs` directories: python3 scripts/diff_outputs.py REF CAND"""
import sys, pathlib, numpy as np
ref, cand = map(pathlib.Path, sys.argv[1:3])
for rf in sorted(ref.glob("*.f32")):
    cf = cand / rf.name
    if not cf.exists():
        print(f"{rf.name}: missing in {cand}"); continue
    a = np.fromfile(rf, "<f4"); b = np.fromfile(cf, "<f4")
    d = np.abs(a - b)
    rel = d / np.maximum(np.abs(a), 1e-3)
    print(f"{rf.name:32s} n={a.size:7d} max_abs={d.max():.3e} mean_abs={d.mean():.3e} "
          f"max_rel={rel.max():.3e} nan={int(np.isnan(b).sum())} ref_range=[{a.min():.2f},{a.max():.2f}]")
