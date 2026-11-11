#!/usr/bin/env python3
"""SANA-Sprint text-to-image on the CPU in fp32, for the Create quality test.

    python sana.py REPO OUT_DIR [--inputs]

Writes OUT_DIR/<prompt key>.png for every prompt in cases.PROMPTS (1024x1024, 2 steps, the
model's defaults), and with --inputs also the swap and try-on inputs (cases.INPUTS) to
OUT_DIR/inputs/. Prompts are encoded first with the text encoder alone, which is then freed, so
the text encoder (Gemma-2-2B) and the transformer are never in memory together; that is also how
the phone would run it. Times go to OUT_DIR/times.json.
"""
import argparse
import gc
import inspect
import json
import os
import time

import torch
from diffusers import SanaSprintPipeline

from cases import INPUTS, PROMPTS, SEED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("out")
    ap.add_argument("--inputs", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    os.makedirs(a.out, exist_ok=True)

    jobs = [(k, p, 1024, 1024, os.path.join(a.out, f"{k}.png")) for k, _, p in PROMPTS]
    if a.inputs:
        os.makedirs(os.path.join(a.out, "inputs"), exist_ok=True)
        jobs += [(k, p, w, h, os.path.join(a.out, "inputs", f"{k}.png")) for k, w, h, p in INPUTS]
    times = {}

    # 1. Text encoder only.
    instruction = inspect.signature(SanaSprintPipeline.__call__).parameters["complex_human_instruction"].default
    t = time.time()
    enc = SanaSprintPipeline.from_pretrained(a.repo, transformer=None, vae=None, dtype=torch.float32)
    times["load_text_encoder_s"] = round(time.time() - t, 1)
    embeds = []
    with torch.inference_mode():
        for key, prompt, *_ in jobs:
            t = time.time()
            pe, pm = enc.encode_prompt(prompt, num_images_per_prompt=1, device="cpu",
                                       complex_human_instruction=instruction)
            embeds.append((pe, pm))
            times[f"encode_{key}_s"] = round(time.time() - t, 1)
    del enc
    gc.collect()

    # 2. Transformer and autoencoder.
    t = time.time()
    pipe = SanaSprintPipeline.from_pretrained(a.repo, text_encoder=None, tokenizer=None, dtype=torch.float32)
    times["load_transformer_s"] = round(time.time() - t, 1)
    for (key, _, w, h, path), (pe, pm) in zip(jobs, embeds):
        t = time.time()
        with torch.inference_mode():
            img = pipe(prompt_embeds=pe, prompt_attention_mask=pm, width=w, height=h, num_inference_steps=2,
                       generator=torch.Generator("cpu").manual_seed(SEED)).images[0]
        img.save(path)
        times[key] = round(time.time() - t, 1)
        print(f"{key}: {times[key]} s", flush=True)
    with open(os.path.join(a.out, "times.json"), "w") as f:
        json.dump(times, f, indent=1)


if __name__ == "__main__":
    main()
