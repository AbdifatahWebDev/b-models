#!/usr/bin/env python3
"""Converts CodeFormer (face restoration with a fidelity control) to ncnn for Brekwol Upscale.

    python tools/codeformer/convert.py --weights codeformer.pth --out dist/ [--check FACES_DIR]
    python tools/codeformer/convert.py --random --out /tmp/cf      # self-test, no weights

Writes codeformer.param / .bin (and the license) into --out:
    in0  512x512 RGB in [-1, 1] (the same aligned crop GFPGAN takes)
    in1  fidelity w, one value in [0, 1], shape 1x1x1: 0 = cleanest face, 1 = closest to the photo
    out0 512x512 RGB in [-1, 1]

Two changes from sczhou/CodeFormer's forward, both exact at inference:
  * The codebook lookup (top-1 index, scatter, matmul) becomes softmax(logits * SHARPEN) @ codebook,
    which ncnn can run.
  * The fusion weight w is an input tensor and the fusion always runs; w = 0 makes its residual
    zero, which is what the original's `if w > 0` skips.
The transformer runs batch-first (ncnn takes the first axis as the batch); same numbers.

pnnx is killed for memory (over 7 GB) on the whole model, so the encoder + transformer and the
generator are converted separately, each traced with only its own weights, and the two ncnn
graphs joined into one (merge_ncnn).

With --check FACES_DIR (aligned 512 px faces, e.g. CodeFormer's inputs/cropped_faces) and pip ncnn,
it compares PyTorch's original forward with the ncnn model at several w.

CodeFormer: github.com/sczhou/CodeFormer, S-Lab License 1.0 (non-commercial; LICENSE-CodeFormer
must go with the converted files).
"""
import argparse
import copy
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arch.codeformer_arch import CodeFormer  # noqa: E402

SIZE = 512
# Logits are scaled before the softmax so it is one-hot to float precision (1e3 still blended
# near-tied codes: up to 0.03 off on [-1, 1]; 1e5 matches the hard lookup exactly). Needs fp32
# blobs: logits reach about 26, times 1e5 overflows fp16. The app runs face models in fp32.
SHARPEN = 1e5


def batch_first(net):
    """A copy of [net] with batch-first attention; the original must keep working for --check."""
    net = copy.deepcopy(net)
    for layer in net.ft_layers:
        layer.self_attn.batch_first = True
    return net


class Front(nn.Module):
    """Encoder, transformer and codebook lookup: face -> 16x16 code features + skip features."""

    def __init__(self, net):
        super().__init__()
        # Only the parts it uses: pnnx is killed for memory when the traced file holds the
        # whole model.
        self.encoder = net.encoder
        self.position_emb = net.position_emb
        self.feat_emb = net.feat_emb
        self.ft_layers = net.ft_layers
        self.idx_pred_layer = net.idx_pred_layer
        self.codebook = net.quantize.embedding
        self.connect_list = list(net.connect_list)
        # Block index -> feature size, fixed for a 512 px input (shapes are tensors when traced).
        self.enc_at = {net.fuse_encoder_block[s]: s for s in net.connect_list}

    def forward(self, x):
        feats = {}
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in self.enc_at:
                feats[self.enc_at[i]] = x
        pos = self.position_emb.unsqueeze(0)
        query = self.feat_emb(x.flatten(2).permute(0, 2, 1))      # (1, 256 tokens, 512)
        for layer in self.ft_layers:
            v = layer.norm1(query)
            # Query and key are equal but computed twice on purpose: given one blob for both,
            # pnnx writes a two-input ncnn MultiHeadAttention, which ncnn reads as (q, k = v),
            # and the logits came out up to 9 off.
            q = v + pos
            k = pos + v
            query = query + layer.self_attn(q, k, value=v, need_weights=False)[0]
            query = query + layer.linear2(layer.activation(layer.linear1(layer.norm2(query))))
        logits = self.idx_pred_layer(query)                       # (1, 256, 1024)
        soft = F.softmax(logits * SHARPEN, dim=2)
        # (1, 256 tokens, 256 channels). Back makes it a 16x16 map: pnnx dropped a permute and
        # reshape placed here, before the graph's output.
        quant = torch.matmul(soft, self.codebook.weight)
        return (quant,) + tuple(feats[s] for s in self.connect_list)


class Back(nn.Module):
    """Generator with the fidelity-weighted skip fusion."""

    def __init__(self, net):
        super().__init__()
        # Only the parts it uses, so the traced file pnnx loads stays small.
        self.generator = net.generator
        self.fuse = net.fuse_convs_dict
        self.connect_list = list(net.connect_list)
        self.gen_at = {net.fuse_generator_block[s]: s for s in net.connect_list}

    def forward(self, quant, f32, f64, f128, f256, w):
        feats = dict(zip(self.connect_list, (f32, f64, f128, f256)))
        x = quant.permute(0, 2, 1).reshape(1, 256, 16, 16)
        for i, block in enumerate(self.generator.blocks):
            x = block(x)
            if i in self.gen_at:
                x = self.fuse[self.gen_at[i]](feats[self.gen_at[i]], x, w)
        return x


class Export(nn.Module):
    def __init__(self, net):
        super().__init__()
        net = batch_first(net)
        self.front, self.back = Front(net), Back(net)

    def forward(self, x, w):
        return self.back(*self.front(x), w)


def load(path):
    net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                     connect_list=['32', '64', '128', '256'])
    if path:
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        net.load_state_dict(ckpt.get('params_ema', ckpt), strict=True)
    return net.eval()


def faces(folder):
    from PIL import Image
    for name in sorted(os.listdir(folder)):
        img = Image.open(os.path.join(folder, name)).convert('RGB').resize((SIZE, SIZE), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(img, np.float32).transpose(2, 0, 1) / 127.5 - 1.0)
        yield name, x.unsqueeze(0)


def read_param(path):
    with open(path) as f:
        lines = [l.split() for l in f.read().splitlines() if l.strip()]
    assert lines[0] == ['7767517'], path
    layers = []
    for t in lines[2:]:
        nin, nout = int(t[2]), int(t[3])
        layers.append({'type': t[0], 'name': t[1], 'in': t[4:4 + nin], 'out': t[4 + nin:4 + nin + nout],
                       'params': t[4 + nin + nout:]})
    return layers


def merge_ncnn(front_param, back_param, out_param):
    """Joins two ncnn graphs: front's outputs out0..outN feed back's inputs in0..inN, back's last
    input becomes the model's in1, back's out0 the model's out0. The .bin files concatenate in the
    same order (Input layers carry no weights)."""
    front, back = read_param(front_param), read_param(back_param)
    n_links = sum(1 for l in front for b in l['out'] if b.startswith('out'))

    def fname(b):
        return b if b == 'in0' else f'f{b}'

    def bname(b):
        if b.startswith('in'):
            k = int(b[2:])
            return f'fout{k}' if k < n_links else 'in1'
        return b if b == 'out0' else f'b{b}'

    layers = []
    for l in front:
        layers.append({**l, 'name': f'f_{l["name"]}', 'in': [fname(b) for b in l['in']],
                       'out': [fname(b) for b in l['out']]})
    for l in back:
        if l['type'] == 'Input' and int(l['out'][0][2:]) < n_links:
            continue
        layers.append({**l, 'name': f'b_{l["name"]}', 'in': [bname(b) for b in l['in']],
                       'out': [bname(b) for b in l['out']]})
    blobs = {b for l in layers for b in l['out']}
    with open(out_param, 'w') as f:
        f.write(f'7767517\n{len(layers)} {len(blobs)}\n')
        for l in layers:
            f.write(' '.join([l['type'], l['name'], str(len(l['in'])), str(len(l['out']))] + l['in'] + l['out'] + l['params']) + '\n')


def trace_halves(weights, work):
    """Saves front.pt and back.pt and returns the pnnx inputshape for each."""
    model = Export(load(weights)).eval()
    x = torch.rand(1, 3, SIZE, SIZE) * 2 - 1
    w = torch.full((1, 1, 1, 1), 0.7)
    feats = model.front(x)
    torch.jit.trace(model.front, (x,), check_trace=False).save(os.path.join(work, 'front.pt'))
    torch.jit.trace(model.back, feats + (w,), check_trace=False).save(os.path.join(work, 'back.pt'))

    def shapes(ts):
        return 'inputshape=' + ','.join('[' + ','.join(map(str, t.shape)) + ']' for t in ts)
    return {'front': shapes((x,)), 'back': shapes(feats + (w,))}


def export(weights, out_dir):
    import subprocess
    with tempfile.TemporaryDirectory() as work:
        # Traced by a separate Python, so the PyTorch models are gone before pnnx runs: pnnx
        # alone peaks at about 6 GB on the generator half.
        cmd = [sys.executable, os.path.abspath(__file__), '--trace-to', work, '--out', out_dir]
        cmd += ['--weights', weights] if weights else ['--random']
        subprocess.run(cmd, check=True)
        with open(os.path.join(work, 'shapes.json')) as f:
            shapes = json.load(f)
        exe = shutil.which('pnnx') or os.path.join(os.path.dirname(__import__('pnnx').__file__), 'pnnx')
        for part in ('front', 'back'):
            subprocess.run([exe, f'{part}.pt', shapes[part], 'fp16=1'], cwd=work, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        merge_ncnn(os.path.join(work, 'front.ncnn.param'), os.path.join(work, 'back.ncnn.param'),
                   os.path.join(out_dir, 'codeformer.param'))
        with open(os.path.join(out_dir, 'codeformer.bin'), 'wb') as out:
            for part in ('front', 'back'):
                with open(os.path.join(work, f'{part}.ncnn.bin'), 'rb') as f:
                    shutil.copyfileobj(f, out)


def check(weights, out_dir, folder):
    """PyTorch first (the original forward and the export wrapper), then ncnn once the PyTorch
    models are freed: both together don't fit in 8 GB."""
    import gc
    import ncnn
    net = load(weights)
    model = Export(net).eval()
    cases = []
    for name, x in faces(folder):
        for w in (0.0, 0.5, 1.0):
            ref = net(x, w=w, adain=False)[0].squeeze(0).numpy()
            mine = model(x, torch.full((1, 1, 1, 1), w)).squeeze(0).numpy()
            cases.append((name, w, x.squeeze(0).numpy(), ref, float(np.abs(mine - ref).max())))
    del net, model
    gc.collect()

    n = ncnn.Net()
    n.opt.use_fp16_storage = n.opt.use_fp16_packed = False
    n.opt.use_fp16_arithmetic = False
    n.load_param(os.path.join(out_dir, 'codeformer.param'))
    n.load_model(os.path.join(out_dir, 'codeformer.bin'))
    worst = 0.0
    for name, w, x, ref, e_wrap in cases:
        # ncnn.Mat borrows the arrays: keep them alive until extract() has run.
        a = np.ascontiguousarray(x)
        b = np.full((1, 1, 1), w, np.float32)
        ma, mb = ncnn.Mat(a), ncnn.Mat(b)
        ex = n.create_extractor()
        ex.input('in0', ma)
        ex.input('in1', mb)
        _, o = ex.extract('out0')
        got = np.array(o).copy()
        del ex, ma, mb
        d = np.abs(got - ref)
        print(f'  {name} w={w}: export max err {e_wrap:.1e}; ncnn mean err {d.mean():.1e} '
              f'(99.9% under {np.quantile(d, 0.999):.1e}, max {d.max():.1e}) on [-1, 1]', flush=True)
        worst = max(worst, float(d.mean()))
    return worst


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--weights', help='codeformer.pth (v0.1.0 release)')
    p.add_argument('--random', action='store_true', help='random weights: test the conversion only')
    p.add_argument('--check', metavar='FACES_DIR', help='compare with PyTorch on these aligned faces')
    p.add_argument('--out', required=True)
    p.add_argument('--trace-to', help=argparse.SUPPRESS)  # internal: the tracing step of export()
    p.add_argument('--check-only', action='store_true', help='check the files already in --out')
    a = p.parse_args()
    if not a.weights and not a.random:
        p.error('give --weights or --random')
    torch.set_grad_enabled(False)
    weights = None if a.random else a.weights
    if a.trace_to:
        with open(os.path.join(a.trace_to, 'shapes.json'), 'w') as f:
            json.dump(trace_halves(weights, a.trace_to), f)
        return
    os.makedirs(a.out, exist_ok=True)
    if not a.check_only:
        export(weights, a.out)
    shutil.copyfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LICENSE-CodeFormer'),
                    os.path.join(a.out, 'LICENSE-CodeFormer.txt'))
    print('wrote', ', '.join(sorted(os.listdir(a.out))))
    if a.check:
        worst = check(weights, a.out, a.check)
        # fp16 weights: a mean error of a few thousandths on [-1, 1] is under one 8-bit level
        # (0.002-0.009 on the six sample faces; the rare larger pixels are codebook choices that
        # flipped between near-equal codes).
        if worst > 0.01:
            raise SystemExit(f'ncnn output differs from PyTorch (mean error {worst:.3g})')
        print('check passed')


if __name__ == '__main__':
    main()
