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
        # Every model call's input and output, so the pieces can be compared step by step.
        calls = {}
        inner = pipe.transformer.forward

        def traced(*a, _inner=inner, _calls=calls, **k):
            out = _inner(*a, **k)
            i = len(_calls) // 3
            _calls[f'x{i}'] = k['hidden_states'][0].float().numpy()
            _calls[f'v{i}'] = out[0][0].float().numpy()
            _calls[f't{i}'] = np.float32(k['timestep'].float().reshape(-1)[0].item())
            return out
        pipe.transformer.forward = traced
        t = time.time()
        img = pipe(image=imgs or None, prompt_embeds=torch.from_numpy(embeds[key])[None], latents=noise.clone(),
                   height=512, width=512, num_inference_steps=STEPS, guidance_scale=1.0).images[0]
        times[key] = round(time.time() - t, 1)
        pipe.transformer.forward = inner
        np.savez(os.path.join(work, f'{key}-trace.npz'), **calls)
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
        tp = os.path.join(work, f'{key}-trace.npz')
        want_calls = np.load(tp) if os.path.exists(tp) else None

        def cosine(a, b):
            return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

        def trace(i, tokens, out, sigma, n=noise.shape[1] * noise.shape[2]):
            bad = int((~np.isfinite(tokens)).sum() + (~np.isfinite(out)).sum())
            line = f'  {key}: call {i}, t {sigma:.4f}, in range {np.abs(tokens).max():.1f}, out range {np.abs(out).max():.1f}, not finite {bad}'
            if want_calls is not None and f'x{i}' in want_calls:
                wx, wv = want_calls[f'x{i}'], want_calls[f'v{i}']
                if wx.shape == tokens.shape and wv.shape == out.shape:
                    line += (f'; against PyTorch (t {float(want_calls[f"t{i}"]):.4f}): in picture {cosine(tokens[:n], wx[:n]):.4f}'
                             + (f', in references {cosine(tokens[n:], wx[n:]):.4f} (range {np.abs(wx[n:]).max():.1f})' if len(tokens) > n else '')
                             + f', out picture {cosine(out[:n], wv[:n]):.4f}')
                else:
                    line += f'; shapes differ from PyTorch: in {tokens.shape} / {wx.shape}, out {out.shape} / {wv.shape}'
            print(line, flush=True)

        t = time.time()
        img = p.generate(text, noise, refs, STEPS, trace=trace,
                         progress=lambda i, n: print(f'  {key}: step {i}/{n}, {time.time() - t:.0f} s', flush=True))
        print(f'  {key}: picture range {np.nanmin(img):.2f} .. {np.nanmax(img):.2f}, not finite {int((~np.isfinite(img)).sum())}', flush=True)
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
