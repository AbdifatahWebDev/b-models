#!/usr/bin/env python3
"""Builds Real-ESRGAN Lite (x2plus) for ncnn from the official PyTorch weights.

Usage: tools/lite/build.py RealESRGAN_x2plus.pth realesrgan-x4plus.param OUT_DIR

pnnx runs out of memory on this network, and x2plus is x4plus with one extra first step, so the
ncnn files are written directly:
- the .param is realesrgan-x4plus.param with a pixel unshuffle (Reorg) before the first
  convolution, which takes 12 channels instead of 3;
- the .bin is the 351 convolutions in BasicSR order, fp16 weights + fp32 bias, the layout of
  realesrgan-x4plus.bin (rebuilt from RealESRGAN_x4plus.pth it matches to fp16 rounding).
The result is checked against PyTorch. Inputs need even sides: the app pads tiles to 2.

Weights: Real-ESRGAN (BSD-3-Clause). The RRDBNet below is BasicSR's (Apache-2.0), trimmed.
"""
import argparse
import os
import struct
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FP16_TAG = 0x01306B47


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        return self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = ResidualDenseBlock(nf, gc), ResidualDenseBlock(nf, gc), ResidualDenseBlock(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNetX2(nn.Module):
    def __init__(self, nf=64, nb=23, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(12, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(F.pixel_unshuffle(x, 2))
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def write_bin(state, path):
    names = [k[: -len(".weight")] for k in state if k.endswith(".weight")]
    if len(names) != 351:
        sys.exit(f"expected 351 convolutions, found {len(names)}")
    with open(path, "wb") as f:
        for k in names:
            w = state[k + ".weight"].numpy().astype(np.float16)
            f.write(struct.pack("<I", FP16_TAG))
            f.write(w.tobytes())
            f.write(b"\0" * (-(w.size * 2) % 4))
            f.write(state[k + ".bias"].numpy().astype(np.float32).tobytes())


def write_param(x4_param, path):
    lines = open(x4_param).read().splitlines()
    if lines[0] != "7767517":
        sys.exit("not an ncnn param file")
    layers, blobs = map(int, lines[1].split())
    out = [lines[0], f"{layers + 1} {blobs + 1}"]
    first = False
    for line in lines[2:]:
        if line.startswith("Input "):
            out += [line, "Reorg                    unshuffle                1 1 data data_unshuffle 0=2 1=0"]
            continue
        if line.startswith("Convolution") and " 1 1 data " in line:
            if "6=1728" not in line:
                sys.exit("unexpected first convolution")
            line = line.replace(" 1 1 data ", " 1 1 data_unshuffle ").replace("6=1728", "6=6912")
            first = True
        out.append(line)
    if not first:
        sys.exit("first convolution not found")
    open(path, "w").write("\n".join(out) + "\n")


def verify(model, param, binary):
    import ncnn

    rng = np.random.default_rng(1)
    for h, w in [(64, 64), (100, 132)]:
        img = (rng.random((3, h, w)) * 0.8 + 0.1).astype(np.float32)
        with torch.no_grad():
            want = model(torch.from_numpy(img)[None])[0].numpy()
        net = ncnn.Net()
        net.opt.use_fp16_storage = net.opt.use_fp16_packed = net.opt.use_fp16_arithmetic = False
        net.load_param(param)
        net.load_model(binary)
        ex = net.create_extractor()
        ex.input("data", ncnn.Mat(img))
        _, out = ex.extract("output")
        got = np.array(out)
        if got.shape != want.shape:
            sys.exit(f"shape mismatch on {w}x{h}: ncnn {got.shape}, torch {want.shape}")
        err = float(np.abs(got - want).max())
        print(f"{w}x{h}: ncnn vs torch max {err:.5f}, mean {float(np.abs(got - want).mean()):.6f}")
        if err > 0.01:
            sys.exit("ncnn output differs from PyTorch")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights")
    ap.add_argument("x4_param")
    ap.add_argument("out_dir")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    state = torch.load(a.weights, map_location="cpu", weights_only=True)
    state = state.get("params_ema", state.get("params", state))
    model = RRDBNetX2().eval()
    model.load_state_dict(state, strict=True)
    param = os.path.join(a.out_dir, "realesrgan-x2plus.param")
    binary = os.path.join(a.out_dir, "realesrgan-x2plus.bin")
    write_param(a.x4_param, param)
    write_bin(state, binary)
    verify(model, param, binary)
    print("wrote", param, binary)


if __name__ == "__main__":
    main()
