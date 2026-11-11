#!/usr/bin/env python3
"""FLUX.2 klein 4B (distilled, 4 steps) on the CPU in fp32 at 512 px, for the Create and edit tests.

    python klein.py REPO OUT_DIR INPUTS_DIR SANA_DIR

Text-to-image: cases.KLEIN_PROMPTS at 512x512 -> OUT_DIR/<key>.png. Edits: cases.EDITS, whose input
images (from INPUTS_DIR, or SANA_DIR for "sana:<key>") are scaled to about 512x512 first, as the
phone would -> OUT_DIR/edit_<key>.png. All prompts are encoded in one pass of the text encoder
(Qwen3-4B), which is then freed before the transformer loads. Times go to OUT_DIR/times.json.
"""
import argparse
import gc
import json
import math
import os
import time

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

from cases import EDITS, KLEIN_PROMPTS, PROMPTS, SEED

AREA = 512 * 512


def fit(img):
    """Scale to about AREA pixels, sides a multiple of 16."""
    s = math.sqrt(AREA / (img.width * img.height))
    w, h = max(16, round(img.width * s / 16) * 16), max(16, round(img.height * s / 16) * 16)
    return img.convert("RGB").resize((w, h), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("out")
    ap.add_argument("inputs")
    ap.add_argument("sana")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    os.makedirs(a.out, exist_ok=True)
    prompts = dict((k, p) for k, _, p in PROMPTS)

    jobs = [(k, prompts[k], []) for k in KLEIN_PROMPTS]
    for key, _, srcs, instruction in EDITS:
        imgs = []
        for s in srcs:
            path = os.path.join(a.sana, s[5:] + ".png") if s.startswith("sana:") else os.path.join(a.inputs, s + ".png")
            imgs.append(fit(Image.open(path)))
        jobs.append(("edit_" + key, instruction, imgs))
    times = {}

    t = time.time()
    enc = Flux2KleinPipeline.from_pretrained(a.repo, transformer=None, vae=None, dtype=torch.float32)
    times["load_text_encoder_s"] = round(time.time() - t, 1)
    t = time.time()
    with torch.inference_mode():
        embeds, _ = enc.encode_prompt(prompt=[p for _, p, _ in jobs], device="cpu")
    times["encode_all_s"] = round(time.time() - t, 1)
    print(f"encoded {len(jobs)} prompts in {times['encode_all_s']} s", flush=True)
    del enc
    gc.collect()

    t = time.time()
    pipe = Flux2KleinPipeline.from_pretrained(a.repo, text_encoder=None, tokenizer=None, dtype=torch.float32)
    times["load_transformer_s"] = round(time.time() - t, 1)
    for i, (key, _, imgs) in enumerate(jobs):
        size = {} if imgs else {"width": 512, "height": 512}
        t = time.time()
        with torch.inference_mode():
            img = pipe(image=imgs or None, prompt_embeds=embeds[i:i + 1], num_inference_steps=4, guidance_scale=1.0,
                       generator=torch.Generator("cpu").manual_seed(SEED), **size).images[0]
        img.save(os.path.join(a.out, key + ".png"))
        times[key] = round(time.time() - t, 1)
        print(f"{key}: {times[key]} s", flush=True)
        with open(os.path.join(a.out, "times.json"), "w") as f:
            json.dump(times, f, indent=1)


if __name__ == "__main__":
    main()
