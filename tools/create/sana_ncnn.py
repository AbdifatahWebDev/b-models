#!/usr/bin/env python3
"""SANA-Sprint on the ncnn pieces from sana_export.py: the reference the app's engine follows
(engine/create.cpp), and the check of the conversion against PyTorch.

    python sana_ncnn.py DIST "a red fox in the snow" out.png [--seed 42]
"""
import json
import math
import os
import struct
import sys

import numpy as np


class Pieces:
    def __init__(self, dist):
        import ncnn
        self.ncnn = ncnn
        self.dist = dist
        self.meta = json.load(open(os.path.join(dist, 'create.json')))
        raw = open(os.path.join(dist, 'gemma-embed.bin'), 'rb').read()
        assert raw[:4] == b'GEMB'
        _, rows, cols = struct.unpack('<III', raw[4:16])
        self.scales = np.frombuffer(raw, '<f4', rows, 16)
        self.rows = np.frombuffer(raw, np.int8, rows * cols, 16 + 4 * rows).reshape(rows, cols)
        self.h0 = np.fromfile(os.path.join(dist, 'text-h0.bin'), '<f4')

    def net(self, name, fp16=True):
        n = self.ncnn.Net()
        n.opt.use_fp16_storage = fp16
        n.opt.use_fp16_packed = fp16
        n.opt.use_fp16_arithmetic = False
        n.load_param(os.path.join(self.dist, name + '.param'))
        n.load_model(os.path.join(self.dist, name + '.bin'))
        return n

    def run(self, net, inputs):
        ex = net.create_extractor()
        for name, a in zip(net.input_names(), inputs):
            ex.input(name, self.ncnn.Mat(np.ascontiguousarray(a, dtype=np.float32)))
        r, m = ex.extract(sorted(net.output_names())[0])
        assert r == 0
        return np.array(m)

    # ---- 1. text
    def text(self, ids):
        """The transformer's text (L x hidden) for token ids that start with the cached prefix."""
        m = self.meta
        p = len(m['prefix_ids'])
        assert ids[:p] == m['prefix_ids'], 'prompt ids do not start with the cached instruction'
        new = ids[p:]
        n = len(new)
        N = m['text_tokens']
        assert n <= N
        # The pieces take a fixed N tokens: the rest is padding after the prompt, which the
        # causal mask keeps out of every real token's view.
        x = np.zeros((N, m['hidden']), np.float32)
        x[:n] = self.rows[new].astype(np.float32) * self.scales[new, None] * np.float32(m['normalizer'])
        d = m['head_dim']
        inv = 1.0 / (m['rope_theta'] ** (np.arange(0, d, 2, dtype=np.float32) / d))
        f = np.outer(np.arange(p, p + N, dtype=np.float32), inv)
        emb = np.concatenate([f, f], axis=-1)
        cos, sin = np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)
        mask = np.full((N, p + N), m['mask_neg'], np.float32)
        for i in range(N):
            mask[i, :p + i + 1] = 0
        for piece in m['text_pieces']:
            x = self.run(self.net(piece, fp16=False), [x, cos, sin, mask]).reshape(N, -1)
        x = x[:n]
        # Token 0 and every token from the instruction's last one on (encode_prompt's selection).
        first = m['instruction_tokens'] - 1 - p
        return np.concatenate([self.h0[None], x[first:]], 0)

    # ---- 2 + 3. image
    def image(self, text, noise):
        """noise: three arrays of the latent's shape (start, step 1, step 2), standard normal."""
        m = self.meta
        sd = m['sigma_data']
        ts = m['timesteps']
        dit = self.net('sana-dit')
        latents = noise[0] * sd
        denoised = None
        for i in range(len(ts) - 1):
            s, t = ts[i], ts[i + 1]
            x = latents / sd
            c = math.sin(s) / (math.cos(s) + math.sin(s))
            r = math.sqrt(c * c + (1 - c) ** 2)
            xin = x * r
            out = self.run(dit, [xin, np.array([c], np.float32), np.array([m['guidance']], np.float32), text])
            out = out.reshape(xin.shape)
            f = ((1 - 2 * c) * xin + (1 - 2 * c + 2 * c * c) * out) / r * sd
            denoised = math.cos(s) * latents - math.sin(s) * f
            latents = math.cos(t) * denoised + math.sin(t) * noise[i + 1] * sd
        z = denoised / sd / m['scaling_factor']
        img = self.run(self.net('sana-vae'), [z])
        return np.clip((img + 1) * 127.5 + 0.5, 0, 255).astype(np.uint8).transpose(1, 2, 0)


def tokenize(vocab_model, meta, prompt):
    from sentencepiece import SentencePieceProcessor
    sp = SentencePieceProcessor(model_file=vocab_model)
    # SanaSprintPipeline lowercases and trims the prompt (_text_preprocessing, clean_caption=False).
    ids = [2] + sp.encode(meta['instruction'] + prompt.lower().strip())
    return ids[:meta['instruction_tokens'] + meta['max_prompt_tokens']]  # longer prompts are cut


def check(pipe, dist, vocab_model):
    """Compares the pieces with PyTorch: tokens, text embeddings, one transformer pass and images."""
    import torch
    pc = Pieces(dist)
    m = pc.meta
    prompts = ['A red fox sitting in fresh snow in a forest, sharp focus on the eyes',
               'A cozy coffee shop interior with a chalkboard menu that says "BREKWOL CAFE"',
               'Portrait of a young African woman with braided hair smiling, natural window light']
    worst = 1.0
    for prompt in prompts:
        ids = tokenize(vocab_model, m, prompt)
        hf = pipe.tokenizer(m['instruction'] + prompt.lower().strip(), add_special_tokens=True).input_ids
        assert ids == hf[:len(ids)], f'tokenizer differs from HF on: {prompt}'
        with torch.no_grad():
            pe, pm = pipe.encode_prompt(prompt, device='cpu', complex_human_instruction=m['instruction'].split('\n'))
        ref = pe[0][pm[0].bool()].numpy()
        got = pc.text(ids)
        assert ref.shape == got.shape, (ref.shape, got.shape)
        cos = (ref * got).sum(1) / np.linalg.norm(ref, axis=1) / np.linalg.norm(got, axis=1)
        print(f'text: {len(got)} tokens, cosine min {cos.min():.5f} mean {cos.mean():.5f}  ({prompt[:40]})')
        worst = min(worst, float(cos.min()))
    assert worst > 0.99, f'text embeddings differ from PyTorch (cosine {worst:.4f})'

    # Full image: PyTorch's pipeline with a seeded generator against the pieces with the same noise.
    prompt = prompts[0]
    ids = tokenize(vocab_model, m, prompt)
    shape = (1, m['latent_channels'], m['latent_side'], m['latent_side'])
    gen = torch.Generator('cpu').manual_seed(42)
    noise = [torch.randn(shape, generator=gen).numpy()[0] for _ in range(3)]
    with torch.no_grad():
        ref = pipe(prompt=prompt, num_inference_steps=2, generator=torch.Generator('cpu').manual_seed(42),
                   use_resolution_binning=False,
                   width=shape[2] * m['vae_scale'], height=shape[3] * m['vae_scale'],
                   output_type='np').images[0]
    ref = np.clip(ref * 255 + 0.5, 0, 255).astype(np.uint8)
    got = pc.image(pc.text(ids), noise)
    mse = np.mean((ref.astype(np.float64) - got.astype(np.float64)) ** 2)
    psnr = 10 * math.log10(255 ** 2 / max(mse, 1e-9))
    print(f'image: PSNR {psnr:.1f} dB against PyTorch')
    from PIL import Image
    Image.fromarray(np.concatenate([ref, got], 1)).save(os.path.join(dist, 'check-create.png'))
    assert psnr > 25, 'image differs from PyTorch'


def main():
    dist, prompt, out = sys.argv[1], sys.argv[2], sys.argv[3]
    seed = int(sys.argv[sys.argv.index('--seed') + 1]) if '--seed' in sys.argv else 42
    vocab = sys.argv[sys.argv.index('--vocab') + 1]
    pc = Pieces(dist)
    m = pc.meta
    rng = np.random.default_rng(seed)
    shape = (m['latent_channels'], m['latent_side'], m['latent_side'])
    noise = [rng.standard_normal(shape).astype(np.float32) for _ in range(3)]
    from PIL import Image
    Image.fromarray(pc.image(pc.text(tokenize(vocab, m, prompt)), noise)).save(out)


if __name__ == '__main__':
    main()
