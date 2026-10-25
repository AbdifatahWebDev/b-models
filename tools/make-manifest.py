#!/usr/bin/env python3
"""Builds models.json from a folder of model files.

Usage: tools/make-manifest.py <models-dir> [-o models.json]

The app checks every file's size and SHA-256 against this list after downloading it.
Keep MODELS in step with ModelCatalog.kt in brekwol-upscaler.
"""
import argparse
import hashlib
import json
import os
import sys

# Model id -> files the app downloads for it.
MODELS = {
    "realesrgan-x4plus": ["realesrgan-x4plus.param", "realesrgan-x4plus.bin"],
    "realesrgan-x2plus": ["realesrgan-x2plus.param", "realesrgan-x2plus.bin"],
    "realesrgan-anime": ["realesrgan-x4plus-anime.param", "realesrgan-x4plus-anime.bin"],
    "swinir": ["swinir-x4.param", "swinir-x4.bin"],
    "scunet": ["scunet-color.param", "scunet-color.bin"],
    # Face restore: GFPGAN plus the RetinaFace detector that finds the faces.
    "gfpgan": ["gfpgan-1.4.param", "gfpgan-1.4.bin", "retinaface-mnet.param", "retinaface-mnet.bin"],
    # Face restore with a likeness input (tools/codeformer); non-commercial licence.
    "codeformer": ["codeformer.param", "codeformer.bin", "retinaface-mnet.param", "retinaface-mnet.bin"],
    # Generative: AdcSR one-step diffusion (tools/adcsr), fixed 128 px tiles.
    "adcsr": ["adcsr-x4.param", "adcsr-x4.bin"],
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models_dir")
    ap.add_argument("-o", "--output", default="-")
    args = ap.parse_args()

    models = []
    for model_id, files in MODELS.items():
        paths = [os.path.join(args.models_dir, f) for f in files]
        if not all(os.path.isfile(p) for p in paths):
            continue
        models.append({
            "id": model_id,
            "files": [{"name": f, "size": os.path.getsize(p), "sha256": sha256(p)} for f, p in zip(files, paths)],
        })
    if not models:
        sys.exit(f"no models found in {args.models_dir}")

    text = json.dumps({"version": 1, "models": models}, indent=2) + "\n"
    if args.output == "-":
        sys.stdout.write(text)
    else:
        with open(args.output, "w", newline="\n") as f:
            f.write(text)
        print(f"wrote {args.output}: {', '.join(m['id'] for m in models)}", file=sys.stderr)


if __name__ == "__main__":
    main()
