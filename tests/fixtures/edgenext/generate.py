#!/usr/bin/env python3
"""Fixtures for tests/test_edgenext.rs: pieces of timm's `edgenext_small`, as
FoundationStereo's `Feature` builds it (`timm.create_model('edgenext_small')`).

Weights are seeded random (no download). LayerNorm affines and the ConvBlock layer
scales `gamma` (initialised to 1e-6 in timm) are randomised so every branch matters.
The reference is computed in FP64 on the CPU; inputs, weights and outputs are written
as little-endian f32.

Writes, under this directory:
  stem_stage0/      stem + stages[0] (3 ConvBlocks, k 3), input [1, 3, 32, 48]
  stage1_convs/     stages[1].downsample + its 2 ConvBlocks (k 5), input [1, 48, 8, 12]
  stb_stage{1,2,3}/ each stage's SplitTransposeBlock (stages[i].blocks[-1]): 96 ch,
                    2 splits, positional encoding, 6x10; 160 ch, 3 splits, 5x7;
                    304 ch, 4 splits, 3x5

Run with FoundationStereo's environment (timm 1.0):
    python generate.py
"""
import os
import numpy as np
import timm
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def randomise(m):
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("gamma") or n.endswith("gamma_xca"):
                p.copy_(torch.rand(p.shape, generator=g) * 0.5 + 0.5)
            elif n.endswith("temperature"):
                p.copy_(torch.rand(p.shape, generator=g) * 1.5 + 0.5)
            elif "norm" in n or "downsample.0" in n or "stem.1" in n:
                if n.endswith("weight"):
                    p.copy_(1 + 0.2 * torch.randn(p.shape, generator=g))
                else:
                    p.copy_(0.1 * torch.randn(p.shape, generator=g))
            else:
                fan_in = p[0].numel() if p.dim() > 1 else 1
                p.copy_(torch.randn(p.shape, generator=g) / np.sqrt(fan_in))


def write(d, name, t):
    os.makedirs(d, exist_ok=True)
    t.detach().to(torch.float32).contiguous().numpy().tofile(os.path.join(d, name))


def dump(dir_name, modules, x, fn):
    """modules: {state_dict prefix: module}; keys are written relative to the model."""
    d = os.path.join(HERE, dir_name)
    for f in os.listdir(d) if os.path.isdir(d) else []:
        os.remove(os.path.join(d, f))
    write(d, "x.bin", x)
    for prefix, mod in modules.items():
        for k, v in mod.state_dict().items():
            write(d, f"w__{prefix}.{k}.bin", v)
    with torch.no_grad():
        y = fn(x.double())
    write(d, "expected.bin", y)
    print(dir_name, tuple(y.shape))


def main():
    model = timm.create_model("edgenext_small", pretrained=False)
    randomise(model)
    model = model.double().eval()
    g = torch.Generator().manual_seed(1)

    x = torch.randn(1, 3, 32, 48, generator=g, dtype=torch.float64)
    dump("stem_stage0", {"stem": model.stem, "stages.0": model.stages[0]}, x,
         lambda x: model.stages[0](model.stem(x)))

    s1 = model.stages[1]
    blocks = {f"stages.1.downsample": s1.downsample}
    blocks.update({f"stages.1.blocks.{i}": s1.blocks[i] for i in range(2)})
    x = torch.randn(1, 48, 8, 12, generator=g, dtype=torch.float64)
    dump("stage1_convs", blocks, x,
         lambda x: s1.blocks[1](s1.blocks[0](s1.downsample(x))))

    for i, (h, w) in [(1, (6, 10)), (2, (5, 7)), (3, (3, 5))]:
        blk = model.stages[i].blocks[-1]
        idx = len(model.stages[i].blocks) - 1
        x = torch.randn(1, blk.norm.normalized_shape[0], h, w, generator=g,
                        dtype=torch.float64)
        dump(f"stb_stage{i}", {f"stages.{i}.blocks.{idx}": blk}, x, blk)


if __name__ == "__main__":
    main()
