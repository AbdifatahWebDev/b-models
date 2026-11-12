#!/usr/bin/env python3
"""The edit model's pieces against FLUX.2 klein in PyTorch, from the same noise.

    python klein_check.py reference REPO WORK INPUTS    # PyTorch: prompts, noise, pictures
    python klein_check.py pieces DIST WORK OUT          # the pieces on the same; sheets and numbers

Two processes, so the 16 GB of PyTorch weights are gone before the pieces run. Cases: a picture
from words and a two-reference edit (dress the person in image 1 in the dress from image 2), both
512 px, 4 steps. INPUTS has person_2.png and dress.png (the create-test release's inputs).
"""
import gc
import json
import math
import os
import resource
import sys
import time

import numpy as np
from PIL import Image

CASES = [
    ('fox', 'A red fox sitting in fresh snow in a forest, sharp focus on the eyes', []),
    ('dress', 'Dress the woman in image 1 in the yellow floral dress from image 2. Keep her face, pose and the background.',
     ['person_2', 'dress']),
]
AREA = 512 * 512
STEPS = 4
SEED = 42
TOKENS = 512


def fit(img):
    """Scale to about AREA pixels, sides a multiple of 16 (as the phone will)."""
    s = math.sqrt(AREA / (img.width * img.height))
    w, h = max(16, round(img.width * s / 16) * 16), max(16, round(img.height * s / 16) * 16)
    return img.convert('RGB').resize((w, h), Image.LANCZOS)


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def reference(repo, work, inputs):
    import torch
    from diffusers import Flux2KleinPipeline
    from transformers import AutoModel, AutoTokenizer
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import klein_export
    os.makedirs(work, exist_ok=True)
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(repo, subfolder='tokenizer')
    enc = AutoModel.from_pretrained(repo, subfolder='text_encoder', torch_dtype=torch.bfloat16).eval()
    embeds = {}
    for key, prompt, _ in CASES:
        emb, ids, real = klein_export.reference(enc, tok, prompt, TOKENS)
        embeds[key] = emb
        np.save(os.path.join(work, f'{key}-emb.npy'), emb)
        json.dump({'ids': ids, 'real': real}, open(os.path.join(work, f'{key}-ids.json'), 'w'))
        print(f'{key}: {real} prompt tokens', flush=True)
    del enc
    gc.collect()

    pipe = Flux2KleinPipeline.from_pretrained(repo, text_encoder=None, tokenizer=None, torch_dtype=torch.float32)
    pipe.set_progress_bar_config(disable=True)
    times = {}
    for key, _, srcs in CASES:
        imgs = [fit(Image.open(os.path.join(inputs, s + '.png'))) for s in srcs]
        for s, im in zip(srcs, imgs):
            im.save(os.path.join(work, f'{key}-in-{s}.png'))
        noise = torch.randn(1, pipe.transformer.config.in_channels, 32, 32, generator=torch.Generator().manual_seed(SEED))
        np.save(os.path.join(work, f'{key}-noise.npy'), noise[0].numpy())
        t = time.time()
        img = pipe(image=imgs or None, prompt_embeds=torch.from_numpy(embeds[key])[None], latents=noise.clone(),
                   height=512, width=512, num_inference_steps=STEPS, guidance_scale=1.0).images[0]
        times[key] = round(time.time() - t, 1)
        img.save(os.path.join(work, f'{key}-reference.png'))
        print(f'{key}: PyTorch {times[key]} s, peak {peak_gb():.1f} GB', flush=True)
    json.dump(times, open(os.path.join(work, 'times-reference.json'), 'w'))


def pieces(dist, work, out):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import klein_ncnn
    os.makedirs(out, exist_ok=True)
    p = klein_ncnn.Pieces(dist)
    ref_times = json.load(open(os.path.join(work, 'times-reference.json')))
    ok = True
    for key, prompt, srcs in CASES:
        meta = json.load(open(os.path.join(work, f'{key}-ids.json')))
        want_text = np.load(os.path.join(work, f'{key}-emb.npy'))
        t = time.time()
        text = p.text(meta['ids'], meta['real'])
        t_text = time.time() - t
        r = meta['real']
        cos = [float(np.dot(text[i], want_text[i]) / (np.linalg.norm(text[i]) * np.linalg.norm(want_text[i]) + 1e-12)) for i in range(len(text))]
        refs = [np.asarray(Image.open(os.path.join(work, f'{key}-in-{s}.png')).convert('RGB'), np.float32).transpose(2, 0, 1) / 127.5 - 1
                for s in srcs]
        noise = np.load(os.path.join(work, f'{key}-noise.npy'))
        t = time.time()
        img = p.generate(text, noise, refs, STEPS, progress=lambda i, n: print(f'  {key}: step {i}/{n}, {time.time() - t:.0f} s', flush=True))
        t_img = time.time() - t
        got = Image.fromarray(((np.clip(img, -1, 1).transpose(1, 2, 0) + 1) * 127.5).round().astype(np.uint8))
        got.save(os.path.join(out, f'{key}-pieces.png'))
        want = Image.open(os.path.join(work, f'{key}-reference.png')).convert('RGB')
        want.save(os.path.join(out, f'{key}-reference.png'))
        a, b = np.asarray(got, np.float32), np.asarray(want, np.float32)
        psnr = 10 * np.log10(255.0 ** 2 / max(float(np.mean((a - b) ** 2)), 1e-9)) if a.shape == b.shape else 0.0
        print(f'{key}: text cosine real {min(cos[:r]):.4f} / padding mean {np.mean(cos[r:] or [1]):.4f}; picture PSNR {psnr:.1f} dB; '
              f'pieces text {t_text:.0f} s + picture {t_img:.0f} s, PyTorch picture {ref_times[key]} s; peak {peak_gb():.1f} GB', flush=True)
        ok = ok and psnr > 15
        # Sheet: inputs, PyTorch, pieces.
        tiles = [Image.open(os.path.join(work, f'{key}-in-{s}.png')).convert('RGB') for s in srcs] + [want, got]
        sheet = Image.new('RGB', (sum(i.width for i in tiles), max(i.height for i in tiles)), (20, 20, 24))
        x = 0
        for im in tiles:
            sheet.paste(im, (x, 0))
            x += im.width
        sheet.save(os.path.join(out, f'{key}.jpg'), quality=92)
    if not ok:
        raise SystemExit('a picture from the pieces is far from PyTorch')


if __name__ == '__main__':
    {'reference': reference, 'pieces': pieces}[sys.argv[1]](*sys.argv[2:])
