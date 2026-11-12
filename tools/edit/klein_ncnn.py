#!/usr/bin/env python3
"""Runs the edit model's ncnn pieces the way the phone will (see klein_export.py, PLAN.md).

Pieces.text(ids, real) -> the transformer's text input; Pieces.generate(text, noise, references)
-> the picture.
"""
import json
import math
import os
import struct

import numpy as np


def ids_text(n):
    """Token ids (t, h, w, l) of the text tokens (Flux2KleinPipeline._prepare_text_ids)."""
    ids = np.zeros((n, 4), np.int64)
    ids[:, 3] = np.arange(n)
    return ids


def ids_image(height, width, t):
    """Ids of a height × width latent: t = 0 for the picture being made, 10, 20, ... for references."""
    ids = np.zeros((height * width, 4), np.int64)
    ids[:, 0] = t
    ids[:, 1] = np.repeat(np.arange(height), width)
    ids[:, 2] = np.tile(np.arange(width), height)
    return ids


def rope_image(ids, axes, theta):
    """cos, sin [N, sum(axes)] as Flux2PosEmbed: per axis, frequencies theta^(-2i/d), each repeated
    twice (interleaved pairs)."""
    cos, sin = [], []
    for k, d in enumerate(axes):
        freq = 1.0 / (theta ** (np.arange(0, d, 2, dtype=np.float64) / d))
        f = np.outer(ids[:, k].astype(np.float64), freq)
        cos.append(np.repeat(np.cos(f), 2, axis=1))
        sin.append(np.repeat(np.sin(f), 2, axis=1))
    return np.concatenate(cos, 1).astype(np.float32), np.concatenate(sin, 1).astype(np.float32)


def empirical_mu(tokens, steps):
    """diffusers' compute_empirical_mu: the sigma shift for [tokens] image tokens and [steps] steps."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if tokens > 4300:
        return a2 * tokens + b2
    m200, m10 = a2 * tokens + b2, a1 * tokens + b1
    a = (m200 - m10) / 190.0
    return a * steps + (m200 - 200.0 * a)


def sigmas(tokens, steps):
    """The sampler's sigmas: linspace(1, 1/steps) shifted exponentially by mu, then 0
    (FlowMatchEulerDiscreteScheduler with dynamic shifting, as Flux2KleinPipeline sets it up)."""
    mu = empirical_mu(tokens, steps)
    s = np.linspace(1.0, 1.0 / steps, steps)
    s = math.exp(mu) / (math.exp(mu) + (1.0 / s - 1.0))
    return np.append(s, 0.0).astype(np.float32)


def patchify(z):
    """[C, h, w] -> [4C, h/2, w/2] (Flux2KleinPipeline._patchify_latents)."""
    c, h, w = z.shape
    return z.reshape(c, h // 2, 2, w // 2, 2).transpose(0, 2, 4, 1, 3).reshape(c * 4, h // 2, w // 2)


def unpatchify(z):
    """[4C, h, w] -> [C, 2h, 2w]."""
    c, h, w = z.shape
    return z.reshape(c // 4, 2, 2, h, w).transpose(0, 3, 1, 4, 2).reshape(c // 4, h * 2, w * 2)


class Pieces:
    def __init__(self, dist, threads=None):
        import ncnn
        self.ncnn = ncnn
        self.dist = dist
        self.threads = threads or int(os.environ.get('OMP_NUM_THREADS', '4'))
        self.meta = {}
        for name in ('meta-text.json', 'meta-image.json', 'meta-vae.json', 'edit.json'):
            p = os.path.join(dist, name)
            if os.path.exists(p):
                self.meta.update(json.load(open(p)))

    def net(self, name):
        n = self.ncnn.Net()
        n.opt.use_vulkan_compute = False
        n.opt.num_threads = self.threads
        # fp32 between layers, as on the phone (fp16 storage overflows there).
        n.opt.use_fp16_storage = n.opt.use_fp16_packed = n.opt.use_fp16_arithmetic = False
        n.load_param(os.path.join(self.dist, name + '.param'))
        n.load_model(os.path.join(self.dist, name + '.bin'))
        return n

    def run(self, net, inputs):
        ex = net.create_extractor()
        for name, a in zip(net.input_names(), inputs):
            # clone(): ncnn.Mat(array) shares numpy's memory, which is neither kept alive nor aligned
            # for ncnn's SIMD (in-place layers crashed on it at some sizes).
            ex.input(name, self.ncnn.Mat(np.ascontiguousarray(a, dtype=np.float32)).clone())
        r, m = ex.extract(sorted(net.output_names())[0])
        assert r == 0
        return np.array(m)

    def embed(self, ids):
        with open(os.path.join(self.dist, 'qwen-embed.bin'), 'rb') as f:
            magic, _, rows, cols = struct.unpack('<4sIII', f.read(16))
            assert magic == b'GEMB'
            out = np.empty((len(ids), cols), np.float32)
            for i, t in enumerate(ids):
                f.seek(16 + 4 * t)
                scale = struct.unpack('<f', f.read(4))[0]
                f.seek(16 + 4 * rows + t * cols)
                out[i] = np.frombuffer(f.read(cols), np.int8).astype(np.float32) * scale
        return out

    def text(self, ids, real):
        """[ids] padded to text_tokens with [real] real ones first -> [tokens, 3 * hidden]."""
        m = self.meta
        n, d = m['text_tokens'], m['head_dim']
        assert len(ids) == n
        inv = 1.0 / (np.float32(m['rope_theta']) ** (np.arange(0, d, 2, dtype=np.float32) / d))
        f = np.outer(np.arange(n, dtype=np.float32), inv)
        emb = np.concatenate((f, f), axis=-1)
        cos, sin = np.cos(emb), np.sin(emb)
        mask = np.full((n, n), m['mask_neg'], np.float32)
        for i in range(n):
            mask[i, :min(i + 1, real)] = 0
        x = self.embed(ids)
        kept = []
        for k, name in enumerate(m['text_pieces']):
            x = self.run(self.net(name), [x, cos, sin, mask]).reshape(n, -1)
            if k in m['text_outputs']:
                kept.append(x.copy())
        return np.concatenate(kept, axis=1)

    def transformer(self, text, tokens, image_ids, t):
        """One call of the transformer: text [nt, J], image tokens [N, C] (the picture's first,
        then the references') with their ids [N, 4], timestep t in [0, 1] -> [N, C]."""
        m = self.meta
        D = m['inner_dim']
        nt = text.shape[0]
        cos, sin = rope_image(np.concatenate((ids_text(nt), image_ids)), m['rope_axes'], m['rope_theta_image'])
        tm = self.run(self.net('klein-time'), [np.array([t], np.float32)]).reshape(-1)
        temb, mi, mt, ms = tm[:D], tm[D:7 * D], tm[7 * D:13 * D], tm[13 * D:16 * D]
        x = None
        for k, name in enumerate(m['double_pieces']):
            head = [text, tokens] if k == 0 else [x]
            x = self.run(self.net(name), head + [mi, mt, cos, sin])
        last = len(m['single_pieces']) - 1
        for k, name in enumerate(m['single_pieces']):
            x = self.run(self.net(name), [x, ms, cos, sin] + ([temb] if k == last else []))
        return x.reshape(tokens.shape[0], -1)

    def encode(self, rgb):
        """A reference picture, RGB in [-1, 1] [3, H, W] (sides multiples of 16) -> its tokens
        [h·w, 4C] and grid (h, w)."""
        m = self.meta
        z = patchify(self.run(self.net('klein-enc'), [rgb]))
        z = (z - np.array(m['bn_mean'], np.float32)[:, None, None]) / np.array(m['bn_std'], np.float32)[:, None, None]
        return z.reshape(z.shape[0], -1).T.copy(), z.shape[1:]

    def generate(self, text, noise, references=(), steps=4, progress=None, trace=None):
        """text [nt, J]; noise [4C, h, w] standard normal (the pipeline's latents); references:
        RGB arrays in [-1, 1] -> RGB in [-1, 1] [3, 16h, 16w]. trace(step, tokens in, model out, t)
        sees every model call (for the checks)."""
        m = self.meta
        c4, h, w = noise.shape
        x = noise.reshape(c4, -1).T.astype(np.float32).copy()
        ids = [ids_image(h, w, 0)]
        extra = []
        for k, rgb in enumerate(references):
            tokens, (rh, rw) = self.encode(rgb)
            extra.append(tokens)
            ids.append(ids_image(rh, rw, 10 * (k + 1)))
        ids = np.concatenate(ids)
        sg = sigmas(x.shape[0], steps)
        for i in range(steps):
            tokens = np.concatenate([x] + extra)
            full = self.transformer(text, tokens, ids, float(sg[i]))
            if trace:
                trace(i, tokens, full, float(sg[i]))
            x = x + (sg[i + 1] - sg[i]) * full[:x.shape[0]]
            if progress:
                progress(i + 1, steps)
        z = x.T.reshape(c4, h, w)
        z = z * np.array(m['bn_std'], np.float32)[:, None, None] + np.array(m['bn_mean'], np.float32)[:, None, None]
        return self.run(self.net('klein-dec'), [unpatchify(z)])
