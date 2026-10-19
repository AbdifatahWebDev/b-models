#!/usr/bin/env python3
"""Converts AdcSR (one-step diffusion 4x upscaler, CVPR 2025) to ncnn for a Brekwol test.

    python tools/adcsr/convert.py --weights net_params_200.pkl --decoder halfDecoder.ckpt \
        --image test.png --out dist/
    python tools/adcsr/convert.py --random --out /tmp/adcsr   # structure test, no weights

Writes adcsr-x4.param/.bin (fixed 128x128 RGB input in [-1, 1] -> 512x512 in [-1, 1]).
With --image, also upscales that photo in PyTorch (tiled like the app) and saves
<name>_adcsr.png, and times it on this CPU. With --check (pip install ncnn) the ncnn model is
compared with PyTorch.

AdcSR: Apache-2.0 (code). Its UNet comes from Stable Diffusion 2.1-base, so the weights carry
the CreativeML Open RAIL++-M license (commercial use allowed, with use-based restrictions).
"""
import argparse
import copy
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TILE = 128

# Stable Diffusion 2.1-base UNet (stabilityai/stable-diffusion-2-1-base/unet/config.json), so the
# 3.5 GB SD checkpoint isn't needed: AdcSR's net_params replace every weight.
SD21_UNET = dict(
    sample_size=64, in_channels=4, out_channels=4, center_input_sample=False, flip_sin_to_cos=True, freq_shift=0,
    down_block_types=["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
    mid_block_type="UNetMidBlock2DCrossAttn",
    up_block_types=["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
    only_cross_attention=False, block_out_channels=[320, 640, 1280, 1280], layers_per_block=2,
    downsample_padding=1, mid_block_scale_factor=1, act_fn="silu", norm_num_groups=32, norm_eps=1e-5,
    cross_attention_dim=1024, attention_head_dim=[5, 10, 20, 20], dual_cross_attention=False,
    use_linear_projection=True, upcast_attention=False,
)


def report(name, result):
    missing, unexpected = result
    print(f"{name}: {len(missing)} missing, {len(unexpected)} unexpected")
    if missing or unexpected:
        print("  missing:", missing[:8])
        print("  unexpected:", unexpected[:8])
        raise SystemExit(f"{name} weights don't match the model")


def build(weights, decoder_ckpt):
    from diffusers import UNet2DConditionModel
    from diffusers.models.autoencoders.vae import Decoder
    from model import Net

    unet = UNet2DConditionModel(**SD21_UNET)
    decoder = Decoder(in_channels=4, out_channels=3, up_block_types=["UpDecoderBlock2D"] * 4,
                      block_out_channels=[64, 128, 256, 256], layers_per_block=2, norm_num_groups=32,
                      act_fn="silu", norm_type="group", mid_block_add_attention=True)
    if decoder_ckpt:
        ckpt = torch.load(decoder_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        dec = {k.split("decoder.", 1)[1]: v for k, v in state.items() if "decoder." in k}
        print(f"decoder checkpoint: {len(state)} tensors, {len(dec)} for the decoder")
        report("decoder", decoder.load_state_dict(dec, strict=False))
    net = Net(unet, copy.deepcopy(decoder))
    if weights:
        state = torch.load(weights, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            state = state.state_dict()
        state = {k.removeprefix("module."): v for k, v in state.items()}
        print(f"net_params: {len(state)} tensors")
        report("net", net.load_state_dict(state, strict=False))
    net = nn.DataParallel(net)
    model = nn.Sequential(net.module, *decoder.up_blocks, decoder.conv_norm_out, decoder.conv_act, decoder.conv_out)
    return Named(model).eval()


class Named(nn.Module):
    """pnnx writes a Python file named after the layers: a top-level nn.Sequential gives names
    like `0_body_0`, which aren't valid Python. Under an attribute they become `net_0_body_0`."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)


def export(model, out_dir, name, fp16=True):
    import pnnx
    example = torch.rand(1, 3, TILE, TILE) * 2 - 1
    with tempfile.TemporaryDirectory() as work, torch.no_grad():
        stem = name.replace("-", "_")
        pnnx.export(model, os.path.join(work, f"{stem}.pt"), (example,), fp16=fp16)
        for ext in ("param", "bin"):
            shutil.copyfile(os.path.join(work, f"{stem}.ncnn.{ext}"), os.path.join(out_dir, f"{name}.{ext}"))
    return example


def check(model, example, out_dir, name):
    import ncnn
    with torch.no_grad():
        ref = model(example)[0].numpy()
    net = ncnn.Net()
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_packed = False
    net.load_param(os.path.join(out_dir, f"{name}.param"))
    net.load_model(os.path.join(out_dir, f"{name}.bin"))
    ex = net.create_extractor()
    ex.input("in0", ncnn.Mat(example[0].numpy()).clone())
    _, out = ex.extract("out0")
    got = np.array(out).reshape(ref.shape)
    err = float(np.abs(got - ref).max())
    print(f"ncnn vs PyTorch: max abs error {err:.4g} (output range {float(np.abs(ref).max()):.4g})")
    return err / (float(np.abs(ref).max()) or 1.0)


def upscale(model, path, out_dir):
    """4x a photo in TILE px tiles with 16 px overlap, then match colour to the input."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    w, h = img.size
    x = torch.from_numpy(np.asarray(img, np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
    pad = 16
    out = torch.zeros(1, 3, h * 4, w * 4)
    weight = torch.zeros(1, 1, h * 4, w * 4)
    step = TILE - 2 * pad
    t0 = time.time()
    tiles = 0
    with torch.no_grad():
        for y0 in range(0, max(1, h - 2 * pad), step):
            for x0 in range(0, max(1, w - 2 * pad), step):
                ys, xs = min(y0, max(0, h - TILE)), min(x0, max(0, w - TILE))
                tile = x[:, :, ys:ys + TILE, xs:xs + TILE]
                th, tw = tile.shape[2:]
                if th < TILE or tw < TILE:
                    tile = nn.functional.pad(tile, (0, TILE - tw, 0, TILE - th), mode="replicate")
                y = model(tile)[:, :, :th * 4, :tw * 4]
                ramp = torch.ones(1, 1, th * 4, tw * 4)
                out[:, :, ys * 4:(ys + th) * 4, xs * 4:(xs + tw) * 4] += y * ramp
                weight[:, :, ys * 4:(ys + th) * 4, xs * 4:(xs + tw) * 4] += ramp
                tiles += 1
    secs = time.time() - t0
    out = out / weight.clamp(min=1e-6)
    out = (out - out.mean([2, 3], keepdim=True)) / out.std([2, 3], keepdim=True) * x.std([2, 3], keepdim=True) \
        + x.mean([2, 3], keepdim=True)
    res = ((out[0].permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    dst = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + "_adcsr.png")
    Image.fromarray(res).save(dst)
    print(f"upscaled {path} ({w}x{h}) in {tiles} tiles, {secs:.1f} s on this CPU -> {dst}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", help="net_params_200.pkl")
    p.add_argument("--decoder", help="halfDecoder.ckpt")
    p.add_argument("--random", action="store_true")
    p.add_argument("--image", action="append", default=[])
    p.add_argument("--check", action="store_true")
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if not a.random and not (a.weights and a.decoder):
        p.error("give --weights and --decoder, or --random")
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    model = build(None if a.random else a.weights, None if a.random else a.decoder)
    print(f"parameters: {sum(t.numel() for t in model.parameters()) / 1e6:.0f} M")
    for path in a.image:
        upscale(model, path, a.out)
    if not a.no_export:
        example = export(model, a.out, "adcsr-x4")
        if a.check:
            # Report only: this is a research test, and the numbers decide fp16 vs fp32.
            rel = check(model, example, a.out, "adcsr-x4")
            print(f"relative error {rel:.4g} ({'OK' if rel <= 0.05 else 'HIGH: try fp32 weights'})")
    for f in sorted(os.listdir(a.out)):
        print(f"{f}  {os.path.getsize(os.path.join(a.out, f)):,} bytes")


if __name__ == "__main__":
    main()
