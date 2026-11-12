#!/usr/bin/env python3
"""Runs the edit model's ncnn pieces the way the phone will (see klein_export.py, PLAN.md).

So far: Pieces.text(ids, real) -> the transformer's text input [tokens, 3 * hidden].
"""
import json
import os
import struct

import numpy as np


class Pieces:
    def __init__(self, dist, threads=None):
        import ncnn
        self.ncnn = ncnn
        self.dist = dist
        self.threads = threads or int(os.environ.get('OMP_NUM_THREADS', '4'))
        self.meta = {}
        for name in ('meta-text.json', 'meta-image.json', 'edit.json'):
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
        keep = [np.ascontiguousarray(a, dtype=np.float32) for a in inputs]  # ncnn.Mat shares this memory
        for name, a in zip(net.input_names(), keep):
            ex.input(name, self.ncnn.Mat(a))
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
