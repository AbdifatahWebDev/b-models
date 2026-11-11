#!/usr/bin/env python3
"""FASHN VTON v1.5 try-on on the CPU (fp32, its defaults: 30 steps, guidance 1.5, no mask), for
the Swap quality test.

    python tryon.py WEIGHTS_DIR OUT_DIR INPUTS_DIR FASHN_REPO

Runs cases.TRYONS -> OUT_DIR/<key>.png; "fashn_model"/"fashn_garment" are the example pair in
FASHN_REPO/examples/data. Times go to OUT_DIR/times.json.
"""
import argparse
import json
import os
import time

import torch
from fashn_vton import TryOnPipeline
from PIL import Image

from cases import SEED, TRYONS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("out")
    ap.add_argument("inputs")
    ap.add_argument("fashn")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    os.makedirs(a.out, exist_ok=True)
    examples = {"fashn_model": "model.webp", "fashn_garment": "garment.webp"}

    def load(name):
        if name in examples:
            return Image.open(os.path.join(a.fashn, "examples", "data", examples[name])).convert("RGB")
        return Image.open(os.path.join(a.inputs, name + ".png")).convert("RGB")

    times = {}
    t = time.time()
    pipe = TryOnPipeline(weights_dir=a.weights, device="cpu")
    times["load_s"] = round(time.time() - t, 1)
    os.makedirs(os.path.join(a.out, "inputs"), exist_ok=True)
    for name in examples:
        load(name).save(os.path.join(a.out, "inputs", name + ".png"))
    for key, person, garment, category, photo_type in TRYONS:
        t = time.time()
        result = pipe(person_image=load(person), garment_image=load(garment), category=category,
                      garment_photo_type=photo_type, seed=SEED)
        result.images[0].save(os.path.join(a.out, key + ".png"))
        times[key] = round(time.time() - t, 1)
        print(f"{key}: {times[key]} s", flush=True)
        with open(os.path.join(a.out, "times.json"), "w") as f:
            json.dump(times, f, indent=1)


if __name__ == "__main__":
    main()
