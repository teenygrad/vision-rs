#!/usr/bin/env python3
"""Build the real-checkpoint fixtures for `update_block_matches_real_checkpoint`.

Runs FoundationStereo on its sample stereo pair with a real checkpoint, captures the
update block's inputs at chosen refinement iterations, and writes them with the block's
weights and an FP64 reference output:

    <out>/update_block_real_it<N>/
        shapes.txt              input name and [B, C, H, W], in feed order
        <input>.bin             little-endian f32
        w__<state_dict key>.bin update_block weights, f32
        expected_<output>.bin   net0 net1 net2 mask delta_disp, from FP64 on the CPU

The reference is FP64 because PyTorch on CUDA defaults to TF32 convolutions, which on
their own are ~1e-2 off at full width; plain FP32 PyTorch is within ~3e-5 of FP64.

Usage, from a FoundationStereo checkout (its Python environment, with a GPU):

    python gen_update_block_real.py --fs . --run pretrained_models/23-51-11 \
        --out ~/.cache/vision-rs/fs_fixtures

then, in vision-rs:

    FS_REAL_FIXTURES=~/.cache/vision-rs/fs_fixtures \
        cargo test --test test_refinement_head update_block_matches_real -- --ignored
"""

import argparse
import os
import sys

import imageio.v2 as imageio
import numpy as np
import torch

INPUTS = ["net0", "net1", "net2", "inp0", "inp1", "inp2", "corr", "disp", "att0", "att1", "att2"]
OUTPUTS = ["net0", "net1", "net2", "mask", "delta_disp"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fs", required=True, help="FoundationStereo checkout")
    ap.add_argument("--run", required=True, help="checkpoint dir with cfg.yaml and model_best_bp2.pth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, nargs="+", default=[0, 8], help="iterations to capture")
    ap.add_argument("--size", type=int, nargs=2, default=[256, 352], help="input H W (multiples of 32)")
    a = ap.parse_args()

    sys.path.insert(0, a.fs)
    from omegaconf import OmegaConf
    from core.foundation_stereo import FoundationStereo
    from core.update import BasicSelectiveMultiUpdateBlock

    cfg = OmegaConf.load(os.path.join(a.run, "cfg.yaml"))
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    model = FoundationStereo(cfg)
    ckpt = torch.load(os.path.join(a.run, "model_best_bp2.pth"), weights_only=False, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.cuda().eval()
    ub = model.update_block

    def load(name):
        im = imageio.imread(os.path.join(a.fs, "assets", name))[..., :3]
        t = torch.as_tensor(im).permute(2, 0, 1)[None].float().cuda()
        return torch.nn.functional.interpolate(t, size=tuple(a.size), mode="bilinear", align_corners=False)

    captured, step = {}, [0]
    orig = ub.forward

    def hook(net, inp, corr, disp, att):
        if step[0] in a.iters:
            f = lambda t: t.detach().float().cpu().clone()
            captured[step[0]] = dict(zip(INPUTS, [*map(f, net), *map(f, inp), f(corr), f(disp), *map(f, att)]))
        step[0] += 1
        return orig(net, inp, corr, disp, att)

    ub.forward = hook
    with torch.no_grad():
        model.forward(load("left.png"), load("right.png"), iters=max(a.iters) + 1, test_mode=True)
    ub.forward = orig

    weights = {k: v.detach().float().cpu() for k, v in ub.state_dict().items()}
    ref = BasicSelectiveMultiUpdateBlock(cfg, cfg.hidden_dims[0], volume_dim=28)
    ref.load_state_dict(weights)
    ref = ref.double().eval()

    for it, x in captured.items():
        d = os.path.join(os.path.expanduser(a.out), f"update_block_real_it{it}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "shapes.txt"), "w") as fh:
            for n in INPUTS:
                x[n].numpy().astype("<f4").tofile(os.path.join(d, n + ".bin"))
                fh.write(f"{n} {' '.join(map(str, x[n].shape))}\n")
        for k, v in weights.items():
            v.numpy().astype("<f4").tofile(os.path.join(d, f"w__{k}.bin"))
        g = lambda n: x[n].double()
        with torch.no_grad():
            net, mask, delta = ref([g("net0"), g("net1"), g("net2")], [g("inp0"), g("inp1"), g("inp2")],
                                   g("corr"), g("disp"), [g("att0"), g("att1"), g("att2")])
        for n, t in zip(OUTPUTS, [*net, mask, delta]):
            t.float().numpy().astype("<f4").tofile(os.path.join(d, f"expected_{n}.bin"))
        print(f"wrote {d}")


if __name__ == "__main__":
    main()
