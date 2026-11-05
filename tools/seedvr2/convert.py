#!/usr/bin/env python3
"""Converts SeedVR2-3B (one-step diffusion restoration, ByteDance, Apache-2.0) to ncnn pieces for
Brekwol Upscale's experimental Ultra mode.

    python tools/seedvr2/convert.py --dit DIT.safetensors --vae VAE.safetensors --pos-emb pos_emb.pt \\
        --out OUT [--check TILES] [--ref REF_DIR] [--layers-per-piece 2] [--bits 8] [--block 64]
    python tools/seedvr2/convert.py --tiny --out OUT --check TILES   # random small model: tests the
                                                                      # conversion, no weights needed

One tile: x = the 128 px tile in [-1, 1] upscaled 4x by the app (bicubic with antialias, a = -0.5,
as torchvision's resize the model was run and trained with; clamped), 512x512 RGB; noise = 16x64x64
standard normal, made by the app from the tile's position so overlapping tiles share it; out =
512x512 RGB in [-1, 1]. The math is the 3B model's one-step path as run by
github.com/numz/ComfyUI-SeedVR2_VideoUpscaler for a single image:

    lat = VAE encoder mean of x * 0.9152                    (16 x 64 x 64)
    v   = NaDiT([noise, lat, 1], prompt = pos_emb, t = 1000)  (velocity, v_lerp)
    x0  = noise - v                                         (one Euler step from t = 1 to 0)
    out = wavelet colour fix of VAE decoder(x0 / 0.9152) against x, clamped

Rewritten for a fixed 128 px tile, all exact in fp32:
  * The video VAE's causal 3D convolutions on one frame are 2D convolutions with the temporal
    kernel slices summed (the first frame is repeated as the causal padding); its temporal
    up-sampling keeps the first of the two frames, as the model does for a single frame.
  * The DiT's time embedding is fixed (t = 1000) and every block shares it, so each block's AdaLN
    shift/scale/gate is a constant; the prompt projection is a constant too, and the mask channel
    of the input (always 1) is folded into the patch-in bias.
  * Window attention (3x3-ish windows on the 32x32 token grid, shifted in odd layers) is written
    out window by window with the text tokens repeated in each window and their outputs averaged,
    as the model does. The 3D rotary embedding is a constant per window shape; the q/k feature
    order inside each head is permuted (with the q/k norm weights) so the rotation works on
    halves instead of interleaved pairs, which leaves q.k unchanged.
  * Layers 10 and up share one set of weights for image and text tokens; there the two token
    sets go through those layers together, so the weights are stored once.

The network is exported in pieces (encode, groups of DiT blocks, decode) that the app loads one at
a time, so a 3B model runs in little memory. The DiT pieces' Linear layers are quantized by ncnn's
block quantizer (ncnnllm2int, bits/block below); the VAE pieces stay fp16.

--check runs the PyTorch pieces and the ncnn pieces on 128 px tiles of the photos in TILES and
reports the error. With --ref (outputs of tools/seedvr2/reference.py, which runs the community
code's own modules) it also compares the rewrite with the original model.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TILE = 128
SCALE = 4
LAT = TILE * SCALE // 8          # 64: latent side
GRID = LAT // 2                  # 32: token grid side (patch 2)
T_STEP = 1000.0
VAE_SCALING = 0.9152
NORM_EPS = 1e-5
VAE_EPS = 1e-6


# ---------------------------------------------------------------------------------------------
# Weights

class Weights:
    """Lazy fp32 access to a safetensors file (the 3B model doesn't fit in memory twice)."""

    def __init__(self, path=None, tensors=None):
        self.tensors = tensors
        if path:
            from safetensors import safe_open
            self.f = safe_open(path, framework='pt')
            self.names = set(self.f.keys())
        else:
            self.names = set(tensors)

    def __contains__(self, k):
        return k in self.names

    def __getitem__(self, k):
        t = self.tensors[k] if self.tensors is not None else self.f.get_tensor(k)
        return t.float()

    def keys(self):
        return self.names


def dit_config(w):
    d = {}
    d['dim'] = w['vid_in.proj.weight'].shape[0]
    d['in_ch'] = w['vid_in.proj.weight'].shape[1] // 4
    d['layers'] = 1 + max(int(k.split('.')[1]) for k in w.keys() if k.startswith('blocks.'))
    d['mm_layers'] = sum(1 for i in range(d['layers']) if f'blocks.{i}.attn.proj_qkv.vid.weight' in w)
    br = 'vid' if f'blocks.0.attn.norm_q.vid.weight' in w else 'all'
    d['head_dim'] = w[f'blocks.0.attn.norm_q.{br}.weight'].shape[0]
    d['heads'] = w[f'blocks.0.attn.proj_qkv.{br}.weight'].shape[0] // 3 // d['head_dim']
    d['out_ch'] = w['vid_out.proj.weight'].shape[0] // 4
    return d


# ---------------------------------------------------------------------------------------------
# Small ops (written so pnnx turns them into plain ncnn layers)

def rms(x, eps=NORM_EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def linear(w, name, bias=True):
    wt = w[name + '.weight']
    m = nn.Linear(wt.shape[1], wt.shape[0], bias=(name + '.bias') in w and bias)
    with torch.no_grad():
        m.weight.copy_(wt)
        if m.bias is not None:
            m.bias.copy_(w[name + '.bias'])
    return m


def conv(w, name, stride=1, padding=1):
    """A causal 3D (or plain 2D) convolution applied to one frame, as a 2D convolution."""
    wt = w[name + '.weight']
    if wt.ndim == 5:
        wt = wt.sum(2)                       # first frame repeated as causal padding: slices add up
    m = nn.Conv2d(wt.shape[1], wt.shape[0], wt.shape[2:], stride=stride, padding=padding,
                  bias=(name + '.bias') in w)
    with torch.no_grad():
        m.weight.copy_(wt)
        if m.bias is not None:
            m.bias.copy_(w[name + '.bias'])
    return m


def group_norm(w, name):
    g = nn.GroupNorm(32, w[name + '.weight'].shape[0], eps=VAE_EPS)
    with torch.no_grad():
        g.weight.copy_(w[name + '.weight'])
        g.bias.copy_(w[name + '.bias'])
    return g


# ---------------------------------------------------------------------------------------------
# VAE (one frame)

class Resnet(nn.Module):
    def __init__(self, w, p):
        super().__init__()
        self.norm1, self.conv1 = group_norm(w, p + 'norm1'), conv(w, p + 'conv1')
        self.norm2, self.conv2 = group_norm(w, p + 'norm2'), conv(w, p + 'conv2')
        self.short = conv(w, p + 'conv_shortcut', padding=0) if p + 'conv_shortcut.weight' in w else None

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return (self.short(x) if self.short is not None else x) + h


class MidAttention(nn.Module):
    """diffusers Attention (one head, group norm, residual) over the pixels."""

    def __init__(self, w, p):
        super().__init__()
        self.norm = group_norm(w, p + 'group_norm')
        self.q, self.k, self.v = linear(w, p + 'to_q'), linear(w, p + 'to_k'), linear(w, p + 'to_v')
        self.o = linear(w, p + 'to_out.0')

    def forward(self, x):
        b, c, hh, ww = x.shape
        h = self.norm(x).reshape(1, c, hh * ww).transpose(1, 2)
        q, k, v = self.q(h), self.k(h), self.v(h)
        a = torch.softmax(torch.matmul(q, k.transpose(1, 2)) * (c ** -0.5), dim=-1)
        o = self.o(torch.matmul(a, v)).transpose(1, 2).reshape(1, c, hh, ww)
        return x + o


class Mid(nn.Module):
    def __init__(self, w, p):
        super().__init__()
        self.r0, self.attn, self.r1 = Resnet(w, p + 'resnets.0.'), MidAttention(w, p + 'attentions.0.'), Resnet(w, p + 'resnets.1.')

    def forward(self, x):
        return self.r1(self.attn(self.r0(x)))


class Down(nn.Module):
    def __init__(self, w, p):
        super().__init__()
        self.conv = conv(w, p + 'conv', stride=2, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class Up(nn.Module):
    """upscale_conv (1x1, channels x 4 or x 8) + rearrange to 2x space (and 2x time, of which the
    model keeps the first frame), then the 3x3 convolution."""

    def __init__(self, w, p):
        super().__init__()
        uw, ub = w[p + 'upscale_conv.weight'], w[p + 'upscale_conv.bias']
        c = uw.shape[1]
        z = uw.shape[0] // c // 4
        rows = [((x * 2 + y) * z + 0) * c + ch for ch in range(c) for x in range(2) for y in range(2)]
        self.up = nn.Conv2d(c, 4 * c, 1)
        with torch.no_grad():
            self.up.weight.copy_(uw.reshape(uw.shape[0], c)[rows][:, :, None, None])
            self.up.bias.copy_(ub[rows])
        self.conv = conv(w, p + 'conv')

    def forward(self, x):
        return self.conv(F.pixel_shuffle(self.up(x), 2))


class Encoder(nn.Module):
    def __init__(self, w, p='encoder.'):
        super().__init__()
        self.conv_in = conv(w, p + 'conv_in')
        layers = []
        nblocks = 1 + max(int(k.split('.')[2]) for k in w.keys() if k.startswith(p + 'down_blocks.'))
        for i in range(nblocks):
            q = f'{p}down_blocks.{i}.'
            j = 0
            while f'{q}resnets.{j}.norm1.weight' in w:
                layers.append(Resnet(w, f'{q}resnets.{j}.'))
                j += 1
            if f'{q}downsamplers.0.conv.weight' in w:
                layers.append(Down(w, f'{q}downsamplers.0.'))
        self.body = nn.Sequential(*layers)
        self.mid = Mid(w, p + 'mid_block.')
        self.norm_out, self.conv_out = group_norm(w, p + 'conv_norm_out'), conv(w, p + 'conv_out')

    def forward(self, x):
        h = self.mid(self.body(self.conv_in(x)))
        return self.conv_out(F.silu(self.norm_out(h)))


class Decoder(nn.Module):
    def __init__(self, w, p='decoder.'):
        super().__init__()
        self.conv_in = conv(w, p + 'conv_in')
        self.mid = Mid(w, p + 'mid_block.')
        layers = []
        nblocks = 1 + max(int(k.split('.')[2]) for k in w.keys() if k.startswith(p + 'up_blocks.'))
        for i in range(nblocks):
            q = f'{p}up_blocks.{i}.'
            j = 0
            while f'{q}resnets.{j}.norm1.weight' in w:
                layers.append(Resnet(w, f'{q}resnets.{j}.'))
                j += 1
            if f'{q}upsamplers.0.conv.weight' in w:
                layers.append(Up(w, f'{q}upsamplers.0.'))
        self.body = nn.Sequential(*layers)
        self.norm_out, self.conv_out = group_norm(w, p + 'conv_norm_out'), conv(w, p + 'conv_out')

    def forward(self, z):
        h = self.body(self.mid(self.conv_in(z)))
        return self.conv_out(F.silu(self.norm_out(h)))


def upscale(x):
    """What the app feeds the model: torchvision's bicubic resize (antialias, a = -0.5), clamped."""
    return F.interpolate(x, scale_factor=SCALE, mode='bicubic', align_corners=False, antialias=True).clamp(-1, 1)


def wavelet_lowpass(x, levels=5):
    """The low-frequency part of numz's wavelet_decomposition (3x3 blur, dilation 1..16,
    replicate padding)."""
    k = torch.tensor([[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]])
    k = k[None, None].repeat(x.shape[1], 1, 1, 1)
    for i in range(levels):
        r = 2 ** i
        x = F.conv2d(F.pad(x, (r, r, r, r), mode='replicate'), k, groups=x.shape[1], dilation=r)
    return x


# ---------------------------------------------------------------------------------------------
# DiT

def windows(size, num=(4, 3, 3), shifted=False):
    """(h0, h1, w0, w1) of each attention window on a size x size token grid, from the model's
    make_720Pwindows_bysize / make_shifted_720Pwindows_bysize (one frame)."""
    h = w = size
    scale = math.sqrt((45 * 80) / (h * w))
    rh, rw = round(h * scale), round(w * scale)
    wh, ww = math.ceil(rh / num[1]), math.ceil(rw / num[2])
    if not shifted:
        hs = [(i * wh, min((i + 1) * wh, h)) for i in range(math.ceil(h / wh))]
        ws = [(i * ww, min((i + 1) * ww, w)) for i in range(math.ceil(w / ww))]
    else:
        sh = 0.5 if wh < h else 0
        sw = 0.5 if ww < w else 0
        nh = math.ceil((h - sh) / wh) + 1 if sh > 0 else 1
        nw = math.ceil((w - sw) / ww) + 1 if sw > 0 else 1
        hs = [(max(int((i - sh) * wh), 0), min(int((i - sh + 1) * wh), h)) for i in range(nh)]
        ws = [(max(int((i - sw) * ww), 0), min(int((i - sw + 1) * ww), w)) for i in range(nw)]
    hs = [s for s in hs if s[1] > s[0]]
    ws = [s for s in ws if s[1] > s[0]]
    return hs, ws


class Rope:
    """mmrope3d (rotary_embedding_torch, 'lang', theta 10000) for one head size, in the halves
    layout. Text token p sits at (p, p, p); an image token at (txt_len, y, x) within its window."""

    def __init__(self, head_dim, txt_len):
        self.a = head_dim // 3                       # per-axis rotary size (42)
        self.pairs = 3 * (self.a // 2)               # 63
        self.inv = 1.0 / (10000 ** (torch.arange(0, self.a, 2)[: self.a // 2].double() / self.a))
        self.txt_len = txt_len
        self.head_dim = head_dim

    def perm(self):
        """New order of a head's features: rotated pair members first-halves then second-halves."""
        p = self.pairs
        return [2 * j for j in range(p)] + [2 * j + 1 for j in range(p)] + list(range(2 * p, self.head_dim))

    def angles(self, pos):
        """pos: (n, 3) positions -> (n, pairs) angles; pair j is axis j // (a/2), frequency j % (a/2)."""
        per = self.a // 2
        cols = [pos[:, j // per].double() * self.inv[j % per] for j in range(self.pairs)]
        return torch.stack(cols, dim=1)

    def txt(self):
        p = torch.arange(self.txt_len)[:, None].repeat(1, 3)
        a = self.angles(p)
        return a.cos().float(), a.sin().float()

    def window(self, hh, ww):
        y, x = torch.meshgrid(torch.arange(hh), torch.arange(ww), indexing='ij')
        pos = torch.stack([torch.full_like(y, self.txt_len), y, x], dim=-1).reshape(-1, 3)
        a = self.angles(pos)
        return a.cos().float(), a.sin().float()


def rotate(x, cos, sin, pairs):
    """x: (1, n, heads, head_dim) in the halves layout; cos/sin: (1, n, 1, pairs)."""
    x1, x2, x3 = x[..., :pairs], x[..., pairs:2 * pairs], x[..., 2 * pairs:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, x3], dim=-1)


class Branch(nn.Module):
    """One token set's weights in a block (image, text, or the shared set)."""

    def __init__(self, w, p, br, ada, perm, heads, last):
        super().__init__()
        inner = w[f'{p}attn.proj_qkv.{br}.weight'].shape[0] // 3
        hd = inner // heads
        qkv = w[f'{p}attn.proj_qkv.{br}.weight']
        rows = torch.arange(inner).reshape(heads, hd)[:, perm].reshape(-1)
        order = torch.cat([rows, inner + rows, torch.arange(2 * inner, 3 * inner)])
        self.qkv = nn.Linear(qkv.shape[1], qkv.shape[0], bias=False)
        with torch.no_grad():
            self.qkv.weight.copy_(qkv[order])
        assert f'{p}attn.proj_qkv.{br}.bias' not in w
        self.register_buffer('nq', w[f'{p}attn.norm_q.{br}.weight'][perm].clone())
        self.register_buffer('nk', w[f'{p}attn.norm_k.{br}.weight'][perm].clone())
        self.out = linear(w, f'{p}attn.proj_out.{br}')
        if not last:
            self.gate_p = linear(w, f'{p}mlp.{br}.proj_in_gate', bias=False)
            self.in_p = linear(w, f'{p}mlp.{br}.proj_in', bias=False)
            self.out_p = linear(w, f'{p}mlp.{br}.proj_out', bias=False)
        for name, t in ada.items():
            self.register_buffer(name, t.clone())
        self.has_mlp = not last

    def mlp(self, x):
        return self.out_p(F.silu(self.gate_p(x)) * self.in_p(x))


def ada_consts(w, p, br, emb, dim, layers):
    """AdaSingle: emb laid out (dim, layer, 3) plus the block's own shift/scale/gate vectors."""
    e = emb.reshape(dim, len(layers), 3)
    out = {}
    for li, l in enumerate(layers):
        shift = e[:, li, 0] + w[f'{p}ada.{br}.{l}_shift'] if f'{p}ada.{br}.{l}_shift' in w else None
        scale = e[:, li, 1] + w[f'{p}ada.{br}.{l}_scale'] if f'{p}ada.{br}.{l}_scale' in w else None
        gate = e[:, li, 2] + w[f'{p}ada.{br}.{l}_gate'] if f'{p}ada.{br}.{l}_gate' in w else None
        if shift is not None:
            out[f'{l}_shift'], out[f'{l}_scale'] = shift, scale
        if gate is not None:
            out[f'{l}_gate'] = gate
    return out


class Block(nn.Module):
    def __init__(self, w, i, cfg, emb, rope, shifted):
        super().__init__()
        p = f'blocks.{i}.'
        self.heads, self.hd = cfg['heads'], cfg['head_dim']
        self.inner = self.heads * self.hd
        self.last = i == cfg['layers'] - 1
        self.shared = i >= cfg['mm_layers']
        perm = rope.perm()
        dim = cfg['dim']
        if self.shared:
            ada = ada_consts(w, p, 'all', emb, dim, ['attn', 'mlp'])
            self.all = Branch(w, p, 'all', ada, perm, self.heads, False)
        else:
            self.vid = Branch(w, p, 'vid', ada_consts(w, p, 'vid', emb, dim, ['attn', 'mlp']), perm, self.heads, self.last)
            self.txt = Branch(w, p, 'txt', ada_consts(w, p, 'txt', emb, dim, ['attn', 'mlp']), perm, self.heads, self.last)
        hs, ws = windows(GRID, shifted=shifted)
        self.hs, self.ws = hs, ws
        self.pairs = rope.pairs
        tc, ts = rope.txt()
        self.register_buffer('tcos', tc[None, :, None])
        self.register_buffer('tsin', ts[None, :, None])
        for (h0, h1) in hs:
            for (w0, w1) in ws:
                c, s = rope.window(h1 - h0, w1 - w0)
                self.register_buffer(f'c_{h0}_{w0}', c[None, :, None])
                self.register_buffer(f's_{h0}_{w0}', s[None, :, None])

    def heads_of(self, x, n, norm):
        x = x.reshape(1, n, self.heads, self.hd)
        return rms(x) * norm

    # Every tensor keeps a leading 1 so pnnx maps the axes to ncnn without guessing.
    def attention(self, qkv_v, qkv_t, bv, bt):
        I, H, D = self.inner, self.heads, self.hd
        nt = qkv_t.shape[1]
        qt = rotate(self.heads_of(qkv_t[:, :, :I], nt, bt.nq), self.tcos, self.tsin, self.pairs).permute(0, 2, 1, 3)
        kt = rotate(self.heads_of(qkv_t[:, :, I:2 * I], nt, bt.nk), self.tcos, self.tsin, self.pairs).permute(0, 2, 1, 3)
        vt = qkv_t[:, :, 2 * I:].reshape(1, nt, H, D).permute(0, 2, 1, 3)
        g = qkv_v.reshape(1, GRID, GRID, 3 * I)
        rows, txt_sum, count = [], None, 0
        for (h0, h1) in self.hs:
            cols = []
            for (w0, w1) in self.ws:
                n = (h1 - h0) * (w1 - w0)
                x = g[:, h0:h1, w0:w1].reshape(1, n, 3 * I)
                c, s = getattr(self, f'c_{h0}_{w0}'), getattr(self, f's_{h0}_{w0}')
                q = rotate(self.heads_of(x[:, :, :I], n, bv.nq), c, s, self.pairs).permute(0, 2, 1, 3)
                k = rotate(self.heads_of(x[:, :, I:2 * I], n, bv.nk), c, s, self.pairs).permute(0, 2, 1, 3)
                v = x[:, :, 2 * I:].reshape(1, n, H, D).permute(0, 2, 1, 3)
                qq, kk, vv = torch.cat([q, qt], 2), torch.cat([k, kt], 2), torch.cat([v, vt], 2)
                a = torch.softmax(torch.matmul(qq, kk.transpose(2, 3)) * (D ** -0.5), dim=-1)
                o = torch.matmul(a, vv)                                   # (1, H, n + nt, D)
                cols.append(o[:, :, :n].permute(0, 2, 1, 3).reshape(1, h1 - h0, w1 - w0, I))
                ot = o[:, :, n:]
                txt_sum = ot if txt_sum is None else txt_sum + ot
                count += 1
            rows.append(torch.cat(cols, dim=2))
        vid = torch.cat(rows, dim=1).reshape(1, GRID * GRID, I)
        txt = (txt_sum * (1.0 / count)).permute(0, 2, 1, 3).reshape(1, nt, I)
        return vid, txt

    def forward(self, h, c):
        n = h.shape[1]
        if self.shared:
            b = self.all
            x = torch.cat([h, c], dim=1)
            qkv = b.qkv(rms(x) * b.attn_scale + b.attn_shift)
            av, at = self.attention(qkv[:, :n], qkv[:, n:], b, b)
            x = x + b.out(torch.cat([av, at], dim=1)) * b.attn_gate
            if self.last:
                h = x[:, :n]
                return h + b.mlp(rms(h) * b.mlp_scale + b.mlp_shift) * b.mlp_gate, None
            x = x + b.mlp(rms(x) * b.mlp_scale + b.mlp_shift) * b.mlp_gate
            return x[:, :n], x[:, n:]
        bv, bt = self.vid, self.txt
        qv = bv.qkv(rms(h) * bv.attn_scale + bv.attn_shift)
        qt = bt.qkv(rms(c) * bt.attn_scale + bt.attn_shift)
        av, at = self.attention(qv, qt, bv, bt)
        h = h + bv.out(av) * bv.attn_gate
        c = c + bt.out(at) * bt.attn_gate
        h = h + bv.mlp(rms(h) * bv.mlp_scale + bv.mlp_shift) * bv.mlp_gate
        if self.last:
            return h, None
        c = c + bt.mlp(rms(c) * bt.mlp_scale + bt.mlp_shift) * bt.mlp_gate
        return h, c


def time_embedding(w, dim):
    from diffusers.models.embeddings import get_timestep_embedding
    e = get_timestep_embedding(torch.tensor([T_STEP]), 256, flip_sin_to_cos=False, downscale_freq_shift=0)
    e = F.linear(e, w['emb_in.proj_in.weight'], w['emb_in.proj_in.bias'])
    e = F.linear(F.silu(e), w['emb_in.proj_hid.weight'], w['emb_in.proj_hid.bias'])
    e = F.linear(F.silu(e), w['emb_in.proj_out.weight'], w['emb_in.proj_out.bias'])
    return e[0]


# ---------------------------------------------------------------------------------------------
# Pieces

class Encode(nn.Module):
    """(upscaled tile, noise) -> image tokens."""

    def __init__(self, vae, dit, cfg):
        super().__init__()
        self.encoder = Encoder(vae)
        pw = dit['vid_in.proj.weight']                              # (dim, t h w c) = (dim, 1*2*2*33)
        pw = pw.reshape(cfg['dim'], 2, 2, cfg['in_ch']).permute(0, 3, 1, 2)
        z = cfg['out_ch']
        assert cfg['in_ch'] == 2 * z + 1
        self.patch = nn.Conv2d(2 * z, cfg['dim'], 2, stride=2)
        with torch.no_grad():
            self.patch.weight.copy_(pw[:, :2 * z])
            self.patch.bias.copy_(dit['vid_in.proj.bias'] + pw[:, 2 * z].sum((1, 2)))   # mask channel = 1
        self.z = z

    def latent(self, x):
        return self.encoder(x)[:, : self.z] * VAE_SCALING

    def forward(self, x, noise):
        lat = self.latent(x)
        t = self.patch(torch.cat([noise, lat], dim=1))                # (1, dim, 32, 32)
        return t.reshape(1, t.shape[1], GRID * GRID).transpose(1, 2)


class Blocks(nn.Module):
    """A run of DiT blocks; the first piece starts from the constant projected prompt."""

    def __init__(self, blocks, c0=None):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.first = c0 is not None
        if self.first:
            self.register_buffer('c0', c0.clone())

    def forward(self, h, c=None):
        if self.first:
            c = self.c0
        for b in self.blocks:
            h, c = b(h, c)
        return (h, c) if c is not None else h


class Decode(nn.Module):
    """(image tokens, noise, upscaled tile) -> 512 px RGB in [-1, 1]."""

    def __init__(self, vae, dit, cfg, emb):
        super().__init__()
        dim, z = cfg['dim'], cfg['out_ch']
        # The model's output AdaSingle (one layer) reads the per-run cache entry the blocks' "attn"
        # layer filled (same key, emb_repeat_0_vid), so it gets the blocks' (dim, 2, 3) layout's
        # first slice, not a (2 dim, 1, 3) split of its own. That is what the model was trained with.
        e = emb.reshape(dim, 2, 3)
        self.register_buffer('norm_w', dit['vid_out_norm.weight'].clone())
        self.register_buffer('scale', (e[:, 0, 1] + dit['vid_out_ada.out_scale']).clone())
        self.register_buffer('shift', (e[:, 0, 0] + dit['vid_out_ada.out_shift']).clone())
        pw, pb = dit['vid_out.proj.weight'], dit['vid_out.proj.bias']   # rows (h w c)
        rows = [(hh * 2 + ww) * z + c for c in range(z) for hh in range(2) for ww in range(2)]
        self.proj = nn.Linear(dim, 4 * z)
        with torch.no_grad():
            self.proj.weight.copy_(pw[rows])
            self.proj.bias.copy_(pb[rows])
        self.decoder = Decoder(vae)

    def velocity(self, h):
        v = self.proj(rms(h) * self.norm_w * self.scale + self.shift)    # (1, 1024, 64)
        return F.pixel_shuffle(v.transpose(1, 2).reshape(1, v.shape[2], GRID, GRID), 2)

    def forward(self, h, noise, x):
        v = self.velocity(h)
        out = self.decoder((noise - v) * (1.0 / VAE_SCALING))
        return (out - wavelet_lowpass(out) + wavelet_lowpass(x)).clamp(-1, 1)


class Model:
    """Builds the pieces one at a time from the weights."""

    def __init__(self, dit, vae, pos_emb, per_piece):
        self.dit, self.vae = dit, vae
        self.cfg = dit_config(dit)
        with torch.no_grad():
            self.emb = time_embedding(dit, self.cfg['dim'])
            self.c0 = F.linear(pos_emb.float(), dit['txt_in.weight'], dit['txt_in.bias'])[None]
        self.rope = Rope(self.cfg['head_dim'], pos_emb.shape[0])
        L = self.cfg['layers']
        self.groups = [list(range(i, min(i + per_piece, L))) for i in range(0, L, per_piece)]
        print('model:', self.cfg, f'{len(self.groups)} DiT pieces', flush=True)

    def names(self):
        return ['encode'] + [f'dit{k:02d}' for k in range(len(self.groups))] + ['decode']

    def build(self, name):
        with torch.no_grad():
            if name == 'encode':
                return Encode(self.vae, self.dit, self.cfg).eval()
            if name == 'decode':
                return Decode(self.vae, self.dit, self.cfg, self.emb).eval()
            k = int(name[3:])
            blocks = [Block(self.dit, i, self.cfg, self.emb, self.rope, shifted=(i % 2 == 1)) for i in self.groups[k]]
            return Blocks(blocks, self.c0 if k == 0 else None).eval()


def piece_inputs(name, state, x, noise):
    """Inputs of piece [name] from the running state (dict of the previous outputs)."""
    if name == 'encode':
        return (x, noise)
    if name == 'decode':
        return (state['h'], noise, x)
    return (state['h'],) if state.get('c') is None else (state['h'], state['c'])


def piece_outputs(name, out, state):
    if name == 'encode':
        state['h'] = out
    elif name == 'decode':
        state['out'] = out
    else:
        if isinstance(out, tuple):
            state['h'], state['c'] = out
        else:
            state['h'], state['c'] = out, None


# ---------------------------------------------------------------------------------------------
# Tiny random model (same layout and names as the real files, small sizes)

def tiny_weights():
    g = torch.Generator().manual_seed(0)
    rnd = lambda *s, k=0.3: torch.randn(*s, generator=g) * k
    dim, heads, hd, L, mm, txt_in, z = 64, 2, 32, 4, 2, 48, 16
    hid = 256
    w = {
        'vid_in.proj.weight': rnd(dim, (2 * z + 1) * 4, k=0.1), 'vid_in.proj.bias': rnd(dim),
        'txt_in.weight': rnd(dim, txt_in, k=0.1), 'txt_in.bias': rnd(dim),
        'emb_in.proj_in.weight': rnd(dim, 256, k=0.1), 'emb_in.proj_in.bias': rnd(dim),
        'emb_in.proj_hid.weight': rnd(dim, dim, k=0.1), 'emb_in.proj_hid.bias': rnd(dim),
        'emb_in.proj_out.weight': rnd(6 * dim, dim, k=0.1), 'emb_in.proj_out.bias': rnd(6 * dim),
        'vid_out_norm.weight': 1 + rnd(dim), 'vid_out_ada.out_shift': rnd(dim), 'vid_out_ada.out_scale': 1 + rnd(dim),
        'vid_out.proj.weight': rnd(4 * z, dim, k=0.1), 'vid_out.proj.bias': rnd(4 * z),
    }
    for i in range(L):
        p = f'blocks.{i}.'
        brs = ['vid', 'txt'] if i < mm else ['all']
        for br in brs:
            w[f'{p}attn.proj_qkv.{br}.weight'] = rnd(3 * heads * hd, dim, k=0.15)
            w[f'{p}attn.proj_out.{br}.weight'] = rnd(dim, heads * hd, k=0.15)
            w[f'{p}attn.proj_out.{br}.bias'] = rnd(dim)
            w[f'{p}attn.norm_q.{br}.weight'] = 1 + rnd(hd)
            w[f'{p}attn.norm_k.{br}.weight'] = 1 + rnd(hd)
            if not (i == L - 1 and br == 'txt'):
                w[f'{p}mlp.{br}.proj_in_gate.weight'] = rnd(hid, dim, k=0.1)
                w[f'{p}mlp.{br}.proj_in.weight'] = rnd(hid, dim, k=0.1)
                w[f'{p}mlp.{br}.proj_out.weight'] = rnd(dim, hid, k=0.1)
            for l in (['attn', 'mlp'] if not (i == L - 1 and br == 'txt') else ['attn']):
                w[f'{p}ada.{br}.{l}_shift'] = rnd(dim)
                w[f'{p}ada.{br}.{l}_scale'] = 1 + rnd(dim)
                w[f'{p}ada.{br}.{l}_gate'] = rnd(dim)
    # VAE, the real layout (5D causal convs) with small channels
    v = {}
    ch = [32, 32, 64, 64]

    def c3(name, cout, cin, kt=3, k=3, bias=True):
        v[name + '.weight'] = rnd(cout, cin, kt, k, k, k=1.0 / math.sqrt(cin * kt * k * k))
        if bias:
            v[name + '.bias'] = rnd(cout, k=0.05)

    def gn(name, c):
        v[name + '.weight'] = 1 + rnd(c, k=0.1)
        v[name + '.bias'] = rnd(c, k=0.1)

    def res(p, cin, cout):
        gn(p + 'norm1', cin); c3(p + 'conv1', cout, cin)
        gn(p + 'norm2', cout); c3(p + 'conv2', cout, cout)
        if cin != cout:
            c3(p + 'conv_shortcut', cout, cin, kt=1, k=1)

    def mid(p, c):
        res(p + 'resnets.0.', c, c); res(p + 'resnets.1.', c, c)
        gn(p + 'attentions.0.group_norm', c)
        for n in ('to_q', 'to_k', 'to_v', 'to_out.0'):
            v[f'{p}attentions.0.{n}.weight'] = rnd(c, c, k=1.0 / math.sqrt(c))
            v[f'{p}attentions.0.{n}.bias'] = rnd(c, k=0.05)

    c3('encoder.conv_in', ch[0], 3)
    cin = ch[0]
    for i, c in enumerate(ch):
        res(f'encoder.down_blocks.{i}.resnets.0.', cin, c)
        res(f'encoder.down_blocks.{i}.resnets.1.', c, c)
        if i < 3:
            c3(f'encoder.down_blocks.{i}.downsamplers.0.conv', c, c, kt=3 if i >= 1 else 1)
        cin = c
    mid('encoder.mid_block.', ch[-1])
    gn('encoder.conv_norm_out', ch[-1]); c3('encoder.conv_out', 2 * z, ch[-1])
    c3('decoder.conv_in', ch[-1], z)
    mid('decoder.mid_block.', ch[-1])
    rch = list(reversed(ch))
    prev = rch[0]
    for i, c in enumerate(rch):
        for j in range(3):
            res(f'decoder.up_blocks.{i}.resnets.{j}.', prev if j == 0 else c, c)
        if i < 3:
            zr = 2 if i < 2 else 1
            v[f'decoder.up_blocks.{i}.upsamplers.0.upscale_conv.weight'] = rnd(c * 4 * zr, c, 1, 1, 1, k=1.0 / math.sqrt(c))
            v[f'decoder.up_blocks.{i}.upsamplers.0.upscale_conv.bias'] = rnd(c * 4 * zr, k=0.05)
            c3(f'decoder.up_blocks.{i}.upsamplers.0.conv', c, c)
        prev = c
    gn('decoder.conv_norm_out', ch[0]); c3('decoder.conv_out', 3, ch[0])
    pos = torch.randn(58, txt_in, generator=g)
    return Weights(tensors=w), Weights(tensors=v), pos


# ---------------------------------------------------------------------------------------------
# Tiles and noise

def tiles(folder, n=8):
    """(name, upscaled 512 px tile) for the check."""
    return [(name, upscale(x)) for name, x in tiles_raw(folder, n)]


def tiles_raw(folder, n=8):
    """(name, 128 px tile in [-1, 1]): the middle of each photo in [folder]."""
    from PIL import Image
    out = []
    for name in sorted(os.listdir(folder)):
        if os.path.splitext(name)[1].lower() not in ('.png', '.jpg', '.jpeg'):
            continue
        im = Image.open(os.path.join(folder, name)).convert('RGB')
        if min(im.size) < TILE:
            continue
        x0, y0 = (im.width - TILE) // 2, (im.height - TILE) // 2
        t = np.asarray(im.crop((x0, y0, x0 + TILE, y0 + TILE)), np.float32).transpose(2, 0, 1) / 127.5 - 1
        out.append((os.path.splitext(name)[0], torch.from_numpy(t)[None]))
        if len(out) == n:
            break
    return out


def tile_noise(seed=42):
    """The check's noise (the app makes its own from the tile position)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 16, LAT, LAT, generator=g)


# ---------------------------------------------------------------------------------------------
# PyTorch run (piece by piece) and export

def largest(module, ins):
    worst = [0.0, '']

    def hook(name):
        def f(_, __, out):
            for t in (out if isinstance(out, tuple) else (out,)):
                if isinstance(t, torch.Tensor) and t.is_floating_point():
                    v = float(t.abs().max())
                    if v > worst[0]:
                        worst[:] = [v, name]
        return f
    hs = [m.register_forward_hook(hook(n)) for n, m in module.named_modules()]
    with torch.no_grad():
        out = module(*ins)
    for h in hs:
        h.remove()
    return out, worst


def pnnx_exe():
    """The pnnx of the installed pnnx package (an older pnnx elsewhere on PATH drops ops)."""
    import pnnx
    return os.path.join(os.path.dirname(pnnx.__file__), 'pnnx')


def export_piece(name, module, ins, work, fp16):
    with torch.no_grad():
        torch.jit.trace(module, ins, check_trace=False).save(os.path.join(work, f'{name}.pt'))
    shapes = 'inputshape=' + ','.join('[' + ','.join(map(str, t.shape)) + ']' for t in ins)
    r = subprocess.run([pnnx_exe(), f'{name}.pt', shapes, f'fp16={1 if fp16 else 0}'], cwd=work,
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(os.path.join(work, f'{name}.ncnn.param')):
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f'pnnx failed on {name} (exit {r.returncode})')
    for f in os.listdir(work):
        if f.startswith(name + '.') and not f.startswith(name + '.ncnn.'):
            os.remove(os.path.join(work, f))
        elif f.startswith(name + '_') and f.endswith('.py'):
            os.remove(os.path.join(work, f))
    return os.path.join(work, f'{name}.ncnn.param'), os.path.join(work, f'{name}.ncnn.bin')


def quantize(param, binf, out_param, out_bin, tool, bits, block):
    r = subprocess.run([tool, param, binf, out_param, out_bin, 'method=minmax', f'bits={bits}', f'block={block}'],
                       capture_output=True, text=True)
    log = r.stdout + r.stderr
    if r.returncode != 0 or not os.path.exists(out_bin):
        print(log[-3000:])
        raise SystemExit(f'{os.path.basename(tool)} failed on {param}')
    quantized = sum(1 for l in log.splitlines() if 'quantize' in l.lower() and 'skip' not in l.lower())
    skipped = [l for l in log.splitlines() if 'skip' in l.lower()]
    return quantized, skipped


def layer_counts(param):
    counts = {}
    with open(param) as f:
        for line in f.read().splitlines()[2:]:
            t = line.split()
            if t:
                counts[t[0]] = counts.get(t[0], 0) + 1
    return counts


def ncnn_piece(param, binf, ins, n_out):
    import ncnn
    net = ncnn.Net()
    net.opt.use_fp16_storage = net.opt.use_fp16_packed = net.opt.use_fp16_arithmetic = False
    net.opt.num_threads = os.cpu_count() or 4
    net.load_param(param)
    net.load_model(binf)
    arrs = [np.ascontiguousarray(t[0].numpy() if t.shape[0] == 1 and t.ndim in (3, 4) else t.numpy()) for t in ins]
    mats = [ncnn.Mat(a) for a in arrs]
    ex = net.create_extractor()
    for i, m in enumerate(mats):
        ex.input(f'in{i}', m)
    outs = []
    for i in range(n_out):
        r, o = ex.extract(f'out{i}')
        if r != 0:
            raise SystemExit(f'ncnn extract out{i} failed ({r}) in {param}')
        outs.append(torch.from_numpy(np.array(o).copy())[None])
    del ex, mats, arrs, net
    return outs


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def run(model, tile_list, noise, out_dir, args):
    """Piece by piece: build, run every tile in PyTorch, export, run every tile in ncnn, free."""
    work = args.keep_pieces or tempfile.mkdtemp()
    os.makedirs(work, exist_ok=True)
    tool = args.quantizer
    st_pt = {n: {} for n, _ in tile_list}
    st_nc = {n: {} for n, _ in tile_list}
    report, manifest = [], {'tile': TILE, 'scale': SCALE, 'noise': [16, LAT, LAT], 'pieces': []}
    for name in model.names():
        module = model.build(name)
        worst, ins0 = [0.0, ''], None
        for tn, x in tile_list:
            ins = piece_inputs(name, st_pt[tn], x, noise)
            ins0 = ins0 or ins
            out, big = largest(module, ins)
            if big[0] > worst[0]:
                worst = big
            if name == 'encode':
                st_pt[tn]['lat'] = module.latent(x)
            elif name == 'decode':
                st_pt[tn]['v'] = module.velocity(ins[0])
            piece_outputs(name, out, st_pt[tn])
        is_dit = name.startswith('dit')
        entry = {'name': name, 'inputs': len(ins0), 'outputs': 2 if (is_dit and st_pt[tile_list[0][0]].get('c') is not None) else 1}
        line = f'{name}: largest activation {worst[0]:.0f} ({worst[1]})'
        if not args.no_export:
            param, binf = export_piece(name, module, ins0, work, fp16=not is_dit)
            del module
            if is_dit and tool:
                qp, qb = os.path.join(work, f'{name}.q.param'), os.path.join(work, f'{name}.q.bin')
                nq, skipped = quantize(param, binf, qp, qb, tool, args.bits, args.block)
                os.remove(param), os.remove(binf)
                param, binf = qp, qb
                line += f'; {nq} Gemm quantized ({args.bits} bit, block {args.block})'
                if skipped:
                    line += f'; skipped: {len(skipped)}'
                    print('\n'.join(skipped[:10]))
            dst_p, dst_b = os.path.join(out_dir, f'ultra-{name}.param'), os.path.join(out_dir, f'ultra-{name}.bin')
            shutil.move(param, dst_p)
            shutil.move(binf, dst_b)
            entry.update({'param': os.path.basename(dst_p), 'bin': os.path.basename(dst_b),
                          'param_bytes': os.path.getsize(dst_p), 'bin_bytes': os.path.getsize(dst_b),
                          'param_sha256': sha256(dst_p), 'bin_sha256': sha256(dst_b),
                          'layers': layer_counts(dst_p)})
            line += f'; {os.path.getsize(dst_b) / 2**20:.0f} MB'
            if args.check:
                errs = []
                for tn, x in tile_list:
                    ins = piece_inputs(name, st_nc[tn], x, noise)
                    outs = ncnn_piece(dst_p, dst_b, ins, entry['outputs'])
                    out = tuple(outs) if len(outs) > 1 else outs[0]
                    piece_outputs(name, out, st_nc[tn])
                    ref = st_pt[tn]['out'] if name == 'decode' else st_pt[tn]['h']
                    got = st_nc[tn]['out'] if name == 'decode' else st_nc[tn]['h']
                    errs.append(float((got.reshape(ref.shape) - ref).abs().mean() / (ref.abs().mean() + 1e-12)))
                line += f'; ncnn vs PyTorch (carried through all pieces so far) relative mean error {max(errs):.2e}'
        else:
            del module
        manifest['pieces'].append(entry)
        print('  ' + line, flush=True)
        report.append(line)
        import gc
        gc.collect()
    if args.keep_pieces is None:
        shutil.rmtree(work, ignore_errors=True)
    return st_pt, st_nc, report, manifest


def save_sheet(tile_list, st_pt, st_nc, ref_dir, out_dir):
    from PIL import Image
    def img(a, side=TILE * SCALE):
        a = np.asarray(a).reshape(3, side, side).transpose(1, 2, 0)
        return Image.fromarray(np.clip((a + 1) * 127.5 + 0.5, 0, 255).astype(np.uint8))
    rows, lines, worst = [], [], 0.0
    for tn, x in tile_list:
        cells = [img(x[0].numpy())]
        if ref_dir and os.path.exists(os.path.join(ref_dir, tn + '.npz')):
            r = np.load(os.path.join(ref_dir, tn + '.npz'))
            cells.append(img(r['out']))
            d = np.abs(st_pt[tn]['out'][0].numpy() - r['out'])
            rel = lambda a, b: float(np.abs(a - b).mean() / (np.abs(b).mean() + 1e-12))
            dl = rel(st_pt[tn]['lat'][0].numpy(), r['lat'])
            dv = rel(st_pt[tn]['v'][0].numpy(), r['v'])
            lines.append(f'{tn}: rewrite vs original model: latent rel {dl:.2e}, velocity rel {dv:.2e}, '
                         f'output mean {d.mean():.2e} max {d.max():.2e} on [-1, 1]')
            worst = max(worst, float(d.mean()))
        cells.append(img(st_pt[tn]['out'][0].numpy()))
        if 'out' in st_nc[tn]:
            g = st_nc[tn]['out'].reshape(1, 3, TILE * SCALE, TILE * SCALE)
            cells.append(img(g[0].numpy()))
            d = (g - st_pt[tn]['out']).abs()
            lines.append(f'{tn}: ncnn vs PyTorch output mean {float(d.mean()):.2e} max {float(d.max()):.2e} on [-1, 1]')
            os.makedirs(os.path.join(out_dir, 'ncnn-tiles'), exist_ok=True)
            img(g[0].numpy()).save(os.path.join(out_dir, 'ncnn-tiles', tn + '.png'))
        row = Image.new('RGB', (TILE * SCALE * len(cells), TILE * SCALE))
        for k, c in enumerate(cells):
            row.paste(c, (k * TILE * SCALE, 0))
        rows.append(row)
    if rows:
        sheet = Image.new('RGB', (max(r.width for r in rows), sum(r.height for r in rows)))
        for k, r in enumerate(rows):
            sheet.paste(r, (0, k * r.height))
        sheet.save(os.path.join(out_dir, 'check-ultra.jpg'), quality=88)
    return lines, worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dit')
    ap.add_argument('--vae')
    ap.add_argument('--pos-emb')
    ap.add_argument('--tiny', action='store_true')
    ap.add_argument('--out', required=True)
    ap.add_argument('--check')
    ap.add_argument('--ref')
    ap.add_argument('--tiles', type=int, default=8)
    ap.add_argument('--layers-per-piece', type=int, default=2)
    ap.add_argument('--bits', type=int, default=8)
    ap.add_argument('--block', type=int, default=64)
    ap.add_argument('--quantizer', help='path of ncnnllm2int (no quantization without it)')
    ap.add_argument('--no-export', action='store_true', help='PyTorch only (with --ref: the rewrite check)')
    ap.add_argument('--keep-pieces')
    ap.add_argument('--save-tiny', help='with --tiny: write the random model here (for reference.py --tiny)')
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    os.makedirs(a.out, exist_ok=True)
    if a.tiny:
        dit, vae, pos = tiny_weights()
        if a.save_tiny:
            from safetensors.torch import save_file
            os.makedirs(a.save_tiny, exist_ok=True)
            save_file({k: v.contiguous() for k, v in dit.tensors.items()}, os.path.join(a.save_tiny, 'dit.safetensors'))
            save_file({k: v.contiguous() for k, v in vae.tensors.items()}, os.path.join(a.save_tiny, 'vae.safetensors'))
            torch.save(pos, os.path.join(a.save_tiny, 'pos_emb.pt'))
    else:
        dit, vae = Weights(a.dit), Weights(a.vae)
        pos = torch.load(a.pos_emb, map_location='cpu', weights_only=True).float()
    model = Model(dit, vae, pos, a.layers_per_piece)
    tile_list = tiles(a.check, a.tiles) if a.check else [('random', upscale(torch.rand(1, 3, TILE, TILE) * 2 - 1))]
    noise = tile_noise()
    st_pt, st_nc, report, manifest = run(model, tile_list, noise, a.out, a)
    lines, worst = save_sheet(tile_list, st_pt, st_nc, a.ref, a.out)
    for l in lines:
        print('  ' + l)
    report += lines
    if not a.no_export:
        manifest.update({'bits': a.bits, 'block': a.block, 'layers_per_piece': a.layers_per_piece, 'config': model.cfg,
                         'total_bytes': sum(p['param_bytes'] + p['bin_bytes'] for p in manifest['pieces'])})
        with open(os.path.join(a.out, 'ultra.json'), 'w') as f:
            json.dump(manifest, f, indent=1)
        print(f"total {manifest['total_bytes'] / 2**30:.2f} GB in {len(manifest['pieces'])} pieces")
    with open(os.path.join(a.out, 'check-ultra.txt'), 'w') as f:
        f.write('\n'.join(report) + '\n')
    if a.ref and worst > 0.02:
        raise SystemExit(f'the rewrite differs from the original model (mean error {worst:.3g})')


if __name__ == '__main__':
    main()
