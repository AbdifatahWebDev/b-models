#!/usr/bin/env python3
"""Compares the ncnn model's tiles with TSD-SR's own script on the same 128 px tiles.

    tools/tsdsr/vs_official.py OFFICIAL_DIR NCNN_DIR

The official script samples the VAE posterior (random) where the conversion takes its mean, so
small differences are expected; a broken conversion shows as a large one (and in the images).
"""
import os
import sys

import numpy as np
from PIL import Image

off, mine = sys.argv[1:3]
diffs = []
for f in sorted(os.listdir(mine)):
    p = os.path.join(off, f)
    if not os.path.exists(p):
        print(f'{f}: no official output')
        continue
    a = np.asarray(Image.open(p).convert('RGB'), np.float32)
    b = np.asarray(Image.open(os.path.join(mine, f)).convert('RGB'), np.float32)
    d = np.abs(a - b)
    psnr = 10 * np.log10(255 ** 2 / max(float((d ** 2).mean()), 1e-9))
    diffs.append(d.mean())
    print(f'{f}: mean difference {d.mean():.2f} of 255, PSNR {psnr:.1f} dB')
if not diffs:
    sys.exit('nothing compared')
print(f'average mean difference {np.mean(diffs):.2f} of 255')
