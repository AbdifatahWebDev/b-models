#!/usr/bin/env python3
"""Face swaps with FaceFusion's swappers on the CPU, for the Swap quality test.

    python swap.py FACEFUSION_DIR OUT_DIR INPUTS_DIR

For every swapper in cases.SWAPPERS and every pair in cases.SWAPS runs FaceFusion's own
headless-run (swapper at 512 px "pixel boost", then GFPGAN 1.4 as the app's face restore would),
-> OUT_DIR/<swapper>/<pair>.png. hyperswap_1a_256 also runs without GFPGAN, to see what the
swapper alone gives. Times go to OUT_DIR/times.json.
"""
import argparse
import json
import os
import subprocess
import sys
import time

from cases import SWAPPERS, SWAPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("facefusion")
    ap.add_argument("out")
    ap.add_argument("inputs")
    a = ap.parse_args()
    out, inputs = os.path.abspath(a.out), os.path.abspath(a.inputs)
    runs = [(m, m, ["face_enhancer"]) for m in SWAPPERS] + [("hyperswap_1a_256_raw", "hyperswap_1a_256", [])]
    times, failed = {}, 0
    for name, model, extra in runs:
        os.makedirs(os.path.join(out, name), exist_ok=True)
        for key, src, dst in SWAPS:
            target = os.path.join(out, name, key + ".png")
            cmd = [sys.executable, "facefusion.py", "headless-run",
                   "-s", os.path.join(inputs, src + ".png"), "-t", os.path.join(inputs, dst + ".png"), "-o", target,
                   "--processors", "face_swapper", *extra,
                   "--face-swapper-model", model, "--face-swapper-pixel-boost", "512x512",
                   "--face-enhancer-model", "gfpgan_1.4", "--face-selector-mode", "one",
                   "--execution-providers", "cpu", "--output-image-quality", "95", "--log-level", "info"]
            t = time.time()
            r = subprocess.run(cmd, cwd=a.facefusion)
            times[f"{name}/{key}"] = round(time.time() - t, 1)
            ok = r.returncode == 0 and os.path.exists(target)
            failed += not ok
            print(f"{name}/{key}: {'ok' if ok else 'FAILED'} in {times[f'{name}/{key}']} s (includes model loading)",
                  flush=True)
    with open(os.path.join(out, "times.json"), "w") as f:
        json.dump(times, f, indent=1)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
