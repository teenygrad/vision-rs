#!/usr/bin/env python3
"""Real-checkpoint fixtures for `backbone_matches_real_checkpoint` (tests/test_edgenext.rs).

Loads `feature.stem.*` / `feature.stages.*` from a FoundationStereo checkpoint into
timm's `edgenext_small`, runs it in FP64 on the sample left image (resized to
224x320, normalised as `normalize_image` does) and writes, to <out>/edgenext_real/:
    x.bin                   the normalised image [1, 3, 224, 320], f32
    w__<key>.bin            stem/stages weights, keys without the `feature.` prefix
    expected_x{4,8,16,32}.bin  the four stage outputs

The weights (21 MB) are not committed. Usage, from FoundationStereo's environment:
    python gen_real.py --fs ~/Documents/osn/FoundationStereo \\
        --run pretrained_models/23-51-11 --out ~/.cache/vision-rs/fs_fixtures
"""
import argparse
import os

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image


def write(d, name, t):
    t.detach().to(torch.float32).contiguous().numpy().tofile(os.path.join(d, name))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fs", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    ckpt = torch.load(os.path.join(a.fs, a.run, "model_best_bp2.pth"),
                      weights_only=False, map_location="cpu")["model"]
    sd = {k[len("feature."):]: v for k, v in ckpt.items()
          if k.startswith("feature.stem.") or k.startswith("feature.stages.")}
    model = timm.create_model("edgenext_small", pretrained=False)
    missing, _ = model.load_state_dict(sd, strict=False)
    assert not [k for k in missing if k.startswith(("stem.", "stages."))], missing
    model = model.double().eval()

    img = np.array(Image.open(os.path.join(a.fs, "assets/left.png")).convert("RGB"))
    x = torch.from_numpy(img).permute(2, 0, 1)[None].double()
    x = F.interpolate(x, size=(224, 320), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float64).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float64).view(1, 3, 1, 1)
    x = (x / 255.0 - mean) / std
    # The f32 image the port sees, so both sides start from the same input.
    x = x.float().double()

    d = os.path.join(a.out, "edgenext_real")
    os.makedirs(d, exist_ok=True)
    write(d, "x.bin", x)
    for k, v in sd.items():
        write(d, f"w__{k}.bin", v)
    with torch.no_grad():
        y = model.stem(x)
        for i, s in enumerate(model.stages):
            y = s(y)
            write(d, f"expected_x{4 << i}.bin", y)
            print(f"x{4 << i}", tuple(y.shape), float(y.abs().max()))


if __name__ == "__main__":
    main()
