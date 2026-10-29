#!/usr/bin/env python3
"""Sheets for the "Max" test: is a bigger generative model, or cleaning first, worth it?

    tools/max/compare_max.py INPUTS MODELS_DIR TSDSR_OUT OUT_DIR

Columns per photo: the input (bicubic), AdcSR as the app runs it, SCUNet then AdcSR (noise and
JPEG blocks removed first, so AdcSR doesn't turn them into invented detail), and TSD-SR (CVPR
2025, one step on Stable Diffusion 3 Medium, about 2B parameters; run by its own code into
TSDSR_OUT). The top row is the whole photo, the bottom row a 2x crop of its busiest area.
"""
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'compare'))
import compare as C  # noqa: E402

SCUNET_SIZE = 256  # scunet-color is exported for a fixed 256 px input, like in the app


def scunet(n, img):
    """1x clean-up of a photo up to 256 px: padded to the fixed input, cropped back."""
    w, h = img.size
    src = np.asarray(img, np.float32) / 255.0
    pad = np.pad(src, ((0, SCUNET_SIZE - h), (0, SCUNET_SIZE - w), (0, 0)), mode='edge')
    y = C.run(n, pad.transpose(2, 0, 1), n.input_names()[0], n.output_names()[0])
    y = y.transpose(1, 2, 0)[:h, :w]
    return Image.fromarray((np.clip(y, 0, 1) * 255 + 0.5).astype(np.uint8))


def row(cells):
    r = Image.new('RGB', (sum(i.width for i in cells), max(i.height for i in cells)), (20, 20, 24))
    x = 0
    for i in cells:
        r.paste(i, (x, 0))
        x += i.width
    return r


def main():
    inputs, models, tsd_dir, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    adc = C.net(models, 'adcsr-x4', fp32=True, lean=True)
    scu = C.net(models, 'scunet-color', fp32=True)
    times = []
    for f in sorted(os.listdir(inputs)):
        stem = os.path.splitext(f)[0]
        lq = Image.open(os.path.join(inputs, f)).convert('RGB')
        W, H = lq.size[0] * 4, lq.size[1] * 4
        t = time.time()
        a = C.adcsr(adc, lq)
        t_adc = time.time() - t
        t = time.time()
        clean = scunet(scu, lq)
        t_scu = time.time() - t
        ca = C.adcsr(adc, clean)
        t_ca = time.time() - t
        times.append(f'{f}: AdcSR {t_adc:.0f} s, SCUNet {t_scu:.0f} s + AdcSR = {t_ca:.0f} s')
        cols = [('input (bicubic)', lq.resize((W, H), Image.BICUBIC)),
                (f'AdcSR (app now, {t_adc:.0f} s here)', a),
                (f'SCUNet clean, then AdcSR ({t_ca:.0f} s)', ca)]
        p = os.path.join(tsd_dir, stem + '.png')
        if os.path.exists(p):
            cols.append(('TSD-SR (SD3 Medium, 2B)', Image.open(p).convert('RGB').resize((W, H), Image.BICUBIC)))
        full = [C.label(im, t) for t, im in cols]
        crop = min(256, W, H)
        cx, cy = C.busiest_crop(cols[1][1], crop)
        crops = [C.label(im.crop((cx, cy, cx + crop, cy + crop)).resize((crop * 2, crop * 2), Image.NEAREST), t + ' (crop 2x)')
                 for t, im in cols]
        r1, r2 = row(full), row(crops)
        sheet = Image.new('RGB', (max(r1.width, r2.width), r1.height + r2.height), (20, 20, 24))
        sheet.paste(r1, (0, 0))
        sheet.paste(r2, (0, r1.height))
        if sheet.width > 3600:
            sheet = sheet.resize((3600, round(sheet.height * 3600 / sheet.width)), Image.LANCZOS)
        sheet.save(os.path.join(out, f'max-{stem}.jpg'), quality=90)
        print('wrote', stem, '|', times[-1], flush=True)
    with open(os.path.join(out, 'app-times.txt'), 'w') as fh:
        fh.write('\n'.join(times) + '\n')


if __name__ == '__main__':
    main()
