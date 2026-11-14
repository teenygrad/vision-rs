#!/usr/bin/env python3
"""Real-checkpoint fixtures for `feature_decoder_matches_real_checkpoint`
(tests/test_feature_decoder.rs): the whole of FoundationStereo's `Feature` on the
sample left image at 224x320, with a real checkpoint.

The backbone (`feature.stem` / `feature.stages`, timm `edgenext_small`) and the
decoder run in FP64. The frozen DINOv2 / Depth Anything branch (`feature.dino`) runs
in FP32 as `Feature.forward` runs it (bicubic resize to a multiple of 112, ViT-L + DPT
head, bilinear `align_corners=True` resize to 1/4), and its `vit_feat` is a decoder
input, so the test covers the decoder and its handover, not the ViT. Writes to
<out>/feature_real/:
    x4, x8, x16, x32, vit.bin       decoder inputs (backbone maps, vit_feat), f32
    w__<key>.bin                    decoder weights, keys without `feature.`
    expected_x{4,8,16}.bin          `Feature`'s returned maps
    image.bin                       the normalised image [1, 3, 224, 320]

    python gen_real.py --fs ~/Documents/osn/FoundationStereo \\
        --run pretrained_models/23-51-11 --out ~/.cache/vision-rs/fs_fixtures
"""
import argparse
import os
import sys

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from generate import build  # noqa: E402


def write(d, name, t):
    t.detach().to(torch.float32).contiguous().numpy().tofile(os.path.join(d, name))


def sub(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fs", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    sys.path.insert(0, a.fs)
    from core.extractor import DepthAnythingFeature
    from Utils import get_resize_keep_aspect_ratio

    ckpt = torch.load(os.path.join(a.fs, a.run, "model_best_bp2.pth"),
                      weights_only=False, map_location="cpu")["model"]
    feat = sub(ckpt, "feature.")

    img = np.array(Image.open(os.path.join(a.fs, "assets/left.png")).convert("RGB"))
    x = torch.from_numpy(img).permute(2, 0, 1)[None].double()
    x = F.interpolate(x, size=(224, 320), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float64).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float64).view(1, 3, 1, 1)
    x = ((x / 255.0 - mean) / std).float()
    B, C, H, W = x.shape

    # Backbone, FP64.
    bb = timm.create_model("edgenext_small", pretrained=False)
    bb.load_state_dict({k: v for k, v in feat.items()
                        if k.startswith(("stem.", "stages."))}, strict=False)
    bb = bb.double().eval()
    # DINOv2 branch, FP32, as Feature.forward runs it.
    dino = DepthAnythingFeature(encoder="vitl")
    dino.load_state_dict(sub(feat, "dino."))
    dino = dino.eval()
    # Decoder, FP64.
    dec = build(a.fs, [48, 96, 160, 304], 128)
    dec.load_state_dict({k: v for k, v in feat.items()
                         if k.startswith(("deconv", "conv4."))})
    dec = dec.double().eval()

    with torch.no_grad():
        hr, wr = get_resize_keep_aspect_ratio(H, W, divider=int(np.lcm(14, 16)),
                                              max_H=1344, max_W=1344)
        x_in = F.interpolate(x, size=(hr, wr), mode="bicubic", align_corners=False)
        vit = dino(x_in)["out"]
        vit = F.interpolate(vit, size=(H // 4, W // 4), mode="bilinear",
                            align_corners=True)
        print("vit", (hr, wr), tuple(vit.shape))
        xd = x.double()
        y = bb.stem(xd)
        maps = []
        for s in bb.stages:
            y = s(y)
            maps.append(y.float().double())   # the f32 values the port is fed
        vit_d = vit.double()
        x4, x8, x16 = dec(*maps, vit_d)

    d = os.path.join(a.out, "feature_real")
    os.makedirs(d, exist_ok=True)
    write(d, "image.bin", x)
    for n, t in zip(["x4", "x8", "x16", "x32"], maps):
        write(d, f"{n}.bin", t)
    write(d, "vit.bin", vit)
    for k, v in dec.state_dict().items():
        write(d, f"w__{k}.bin", v)
    for n, t in [("x4", x4), ("x8", x8), ("x16", x16)]:
        write(d, f"expected_{n}.bin", t)
        print(n, tuple(t.shape), float(t.abs().max()))


if __name__ == "__main__":
    main()
