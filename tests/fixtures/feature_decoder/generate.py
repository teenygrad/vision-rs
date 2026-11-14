#!/usr/bin/env python3
"""Fixtures for `feature_decoder_matches_pytorch` (tests/test_edgenext.rs): the decoder
half of FoundationStereo's `Feature` (deconv32_16, deconv16_8, deconv8_4, conv4) at
reduced width, chans [8, 12, 16, 24] and 8 ViT channels (the checkpoints use
[48, 96, 160, 304] and 128), on a 64x96 input's maps (x4 16x24 ... x32 2x3).

Built from FoundationStereo's own `Conv2x_IN`, `BasicConv` and `ResidualBlock`, with
the same attribute names as `Feature`, so `state_dict` keys match. Seeded random
weights and inputs; FP64 reference on the CPU; little-endian f32 files.

    python generate.py --fs ~/Documents/osn/FoundationStereo
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))


def build(fs, chans, vit):
    sys.path.insert(0, fs)
    from core.submodule import BasicConv, Conv2x_IN
    from core.extractor import ResidualBlock

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.deconv32_16 = Conv2x_IN(chans[3], chans[2], deconv=True, concat=True)
            self.deconv16_8 = Conv2x_IN(chans[2] * 2, chans[1], deconv=True, concat=True)
            self.deconv8_4 = Conv2x_IN(chans[1] * 2, chans[0], deconv=True, concat=True)
            w = chans[0] * 2 + vit
            self.conv4 = nn.Sequential(
                BasicConv(w, w, kernel_size=3, stride=1, padding=1, norm='instance'),
                ResidualBlock(w, w, norm_fn='instance'),
                ResidualBlock(w, w, norm_fn='instance'),
            )

        def forward(self, x4, x8, x16, x32, vit_feat):
            x16 = self.deconv32_16(x32, x16)
            x8 = self.deconv16_8(x16, x8)
            x4 = self.deconv8_4(x8, x4)
            x4 = self.conv4(torch.cat([x4, vit_feat], dim=1))
            return x4, x8, x16

    return Decoder()


def write(d, name, t):
    t.detach().to(torch.float32).contiguous().numpy().tofile(os.path.join(d, name))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fs", required=True)
    a = p.parse_args()
    chans, vit = [8, 12, 16, 24], 8
    m = build(a.fs, chans, vit)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for prm in m.parameters():
            fan_in = prm[0].numel() if prm.dim() > 1 else 1
            prm.copy_(torch.randn(prm.shape, generator=g) / np.sqrt(fan_in))
    m = m.double().eval()

    d = HERE
    for f in os.listdir(d):
        if f.endswith(".bin"):
            os.remove(os.path.join(d, f))
    shapes = {"x4": (chans[0], 16, 24), "x8": (chans[1], 8, 12),
              "x16": (chans[2], 4, 6), "x32": (chans[3], 2, 3), "vit": (vit, 16, 24)}
    ins = {n: torch.randn(1, *s, generator=g, dtype=torch.float64) for n, s in shapes.items()}
    for n, t in ins.items():
        write(d, f"{n}.bin", t)
    for k, v in m.state_dict().items():
        write(d, f"w__{k}.bin", v)
    with torch.no_grad():
        x4, x8, x16 = m(ins["x4"], ins["x8"], ins["x16"], ins["x32"], ins["vit"])
    for n, t in [("x4", x4), ("x8", x8), ("x16", x16)]:
        write(d, f"expected_{n}.bin", t)
        print(n, tuple(t.shape))


if __name__ == "__main__":
    main()
