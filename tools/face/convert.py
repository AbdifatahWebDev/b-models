#!/usr/bin/env python3
"""Converts the face-restoration models to ncnn for Brekwol Upscale.

    python tools/face/convert.py --gfpgan GFPGANv1.4.pth \
        --retinaface detection_mobilenet0.25_Final.pth --out dist/
    python tools/face/convert.py --random --out /tmp/face   # self-test, no weights needed

Writes, into --out:
    gfpgan-1.4.param / .bin        512x512 RGB in [-1, 1] -> 512x512 RGB in [-1, 1]
    retinaface-mnet.param / .bin   640x640 BGR minus (104, 117, 123) -> boxes, scores, landmarks

Needs torch, numpy and pnnx (pip install torch numpy pnnx). With --check (and pip install ncnn)
it also runs both ncnn models and compares them with PyTorch.

GFPGAN architecture: TencentARC/GFPGAN (Apache-2.0). RetinaFace: xinntao/facexlib (MIT).
The GFPGAN v1.4 weights were trained on FFHQ, which is licensed for non-commercial use.
"""
import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gfpganv1_clean_arch import GFPGANv1Clean  # noqa: E402
from retinaface_net import FPN, SSH, MobileNetV1, make_bbox_head, make_class_head, make_landmark_head  # noqa: E402

GFPGAN_SIZE = 512
DETECT_SIZE = 640


class GfpganExport(nn.Module):
    """GFPGAN v1.4 with fixed noise, returning only the restored face."""

    def __init__(self):
        super().__init__()
        self.net = GFPGANv1Clean(
            out_size=512, num_style_feat=512, channel_multiplier=2, decoder_load_path=None,
            fix_decoder=False, num_mlp=8, input_is_latent=True, different_w=True, narrow=1, sft_half=True)

    def forward(self, x):
        return self.net(x, return_latents=False, return_rgb=False, randomize_noise=False)[0]


class RetinaFaceExport(nn.Module):
    """facexlib's RetinaFace (MobileNet 0.25 backbone), test phase."""

    def __init__(self):
        super().__init__()
        self.backbone = MobileNetV1()
        in_channels, out_channels = 32, 64
        self.fpn = FPN([in_channels * 2, in_channels * 4, in_channels * 8], out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)
        self.ClassHead = make_class_head(fpn_num=3, inchannels=out_channels)
        self.BboxHead = make_bbox_head(fpn_num=3, inchannels=out_channels)
        self.LandmarkHead = make_landmark_head(fpn_num=3, inchannels=out_channels)

    def forward(self, x):
        s1 = self.backbone.stage1(x)
        s2 = self.backbone.stage2(s1)
        s3 = self.backbone.stage3(s2)
        fpn = self.fpn([s1, s2, s3])
        features = [self.ssh1(fpn[0]), self.ssh2(fpn[1]), self.ssh3(fpn[2])]
        boxes = torch.cat([self.BboxHead[i](f) for i, f in enumerate(features)], dim=1)
        scores = torch.cat([self.ClassHead[i](f) for i, f in enumerate(features)], dim=1)
        landmarks = torch.cat([self.LandmarkHead[i](f) for i, f in enumerate(features)], dim=1)
        return boxes, F.softmax(scores, dim=-1), landmarks


def load_gfpgan(path):
    model = GfpganExport()
    if path:
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        state = ckpt.get('params_ema', ckpt.get('params', ckpt))
        model.net.load_state_dict(state, strict=True)
    return model.eval()


def load_retinaface(path):
    model = RetinaFaceExport()
    if path:
        state = torch.load(path, map_location='cpu', weights_only=True)
        # Checkpoints were saved from facexlib's RetinaFace: its backbone is "body", wrapped by
        # IntermediateLayerGetter, sometimes with a DataParallel "module." prefix.
        fixed = {}
        for k, v in state.items():
            k = k.removeprefix('module.')
            if k.startswith('body.'):
                k = 'backbone.' + k[len('body.'):]
            fixed[k] = v
        missing, unexpected = model.load_state_dict(fixed, strict=False)
        # The backbone's classifier (fc) is unused by detection and absent from the checkpoint.
        missing = [k for k in missing if not k.startswith('backbone.fc')]
        unexpected = [k for k in unexpected if not k.startswith('backbone.fc')]
        if missing or unexpected:
            raise SystemExit(f'retinaface weights do not match: missing={missing} unexpected={unexpected}')
    return model.eval()


def export(model, example, out_dir, name, fp16=True):
    """Traces [model] with pnnx and copies the ncnn files to out_dir/name.{param,bin}."""
    import pnnx
    with tempfile.TemporaryDirectory() as work, torch.no_grad():
        # pnnx names its outputs after the .pt file and turns dashes into underscores.
        stem = name.replace('-', '_')
        pnnx.export(model, os.path.join(work, f'{stem}.pt'), (example,), fp16=fp16)
        for ext in ('param', 'bin'):
            shutil.copyfile(os.path.join(work, f'{stem}.ncnn.{ext}'), os.path.join(out_dir, f'{name}.{ext}'))


def check(model, example, out_dir, name, outputs):
    """Runs the ncnn model and compares it with PyTorch on [example]."""
    import ncnn
    with torch.no_grad():
        expected = model(example)
    expected = expected if isinstance(expected, tuple) else (expected,)
    net = ncnn.Net()
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_packed = False
    net.load_param(os.path.join(out_dir, f'{name}.param'))
    net.load_model(os.path.join(out_dir, f'{name}.bin'))
    ex = net.create_extractor()
    ex.input('in0', ncnn.Mat(example.squeeze(0).numpy()).clone())
    worst = 0.0
    for i, blob in enumerate(outputs):
        _, out = ex.extract(blob)
        got = np.array(out).reshape(expected[i].squeeze(0).shape)
        ref = expected[i].squeeze(0).numpy()
        err = float(np.abs(got - ref).max())
        scale = float(np.abs(ref).max()) or 1.0
        print(f'  {name} {blob}: max abs error {err:.4g} (output range {scale:.4g})')
        worst = max(worst, err / scale)
    return worst


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gfpgan', help='GFPGANv1.4.pth')
    p.add_argument('--retinaface', help='detection_mobilenet0.25_Final.pth')
    p.add_argument('--random', action='store_true', help='random weights: test the conversion only')
    p.add_argument('--check', action='store_true', help='compare ncnn output with PyTorch (needs pip ncnn)')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    if not a.random and not (a.gfpgan and a.retinaface):
        p.error('give --gfpgan and --retinaface, or --random')
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)

    det = load_retinaface(None if a.random else a.retinaface)
    det_in = torch.rand(1, 3, DETECT_SIZE, DETECT_SIZE) * 255 - 117
    # The detector stays fp32: it is tiny, and fp16 weights moved its boxes by up to 15%.
    export(det, det_in, a.out, 'retinaface-mnet', fp16=False)

    gfp = load_gfpgan(None if a.random else a.gfpgan)
    gfp_in = torch.rand(1, 3, GFPGAN_SIZE, GFPGAN_SIZE) * 2 - 1
    export(gfp, gfp_in, a.out, 'gfpgan-1.4')

    if a.check:
        worst = max(check(det, det_in, a.out, 'retinaface-mnet', ['out0', 'out1', 'out2']),
                    check(gfp, gfp_in, a.out, 'gfpgan-1.4', ['out0']))
        print(f'worst relative error {worst:.4g}')
        # fp16 weights cost a little precision; a broken conversion is off by far more.
        if worst > 0.05:
            raise SystemExit('ncnn output differs from PyTorch')
    for f in sorted(os.listdir(a.out)):
        print(f'{f}  {os.path.getsize(os.path.join(a.out, f)):,} bytes')


if __name__ == '__main__':
    main()
