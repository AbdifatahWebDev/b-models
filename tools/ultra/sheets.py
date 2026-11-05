#!/usr/bin/env python3
"""Side-by-side sheets for the Ultra test.

    tools/ultra/sheets.py TILES OUT_DIR "Label=DIR" ["Label=DIR" ...]

One sheet per tile: the input (bicubic x4) and each model's output from DIR/<stem>.png (a column
is left out when its file is missing, so one failed model doesn't lose the others). The top row
is the whole 512 px result, the bottom row a 3x zoom of the busiest 160 px area.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageOps

BG = (20, 20, 24)
CROP = 160
ZOOM = 3


def busiest_crop(img, size):
    g = np.asarray(img.convert('L'), np.float32)
    best, at = -1.0, (0, 0)
    step = max(8, size // 4)
    for y in range(0, max(1, g.shape[0] - size + 1), step):
        for x in range(0, max(1, g.shape[1] - size + 1), step):
            v = float(np.abs(np.diff(g[y:y + size, x:x + size], axis=1)).mean())
            if v > best:
                best, at = v, (x, y)
    return at


def label(img, text):
    img = ImageOps.expand(img, border=(0, 28, 0, 0), fill=BG)
    ImageDraw.Draw(img).text((8, 6), text, fill=(240, 240, 240))
    return img


def row(cells):
    r = Image.new('RGB', (sum(i.width for i in cells), max(i.height for i in cells)), BG)
    x = 0
    for i in cells:
        r.paste(i, (x, 0))
        x += i.width
    return r


def main():
    tiles, out = sys.argv[1], sys.argv[2]
    models = [a.split('=', 1) for a in sys.argv[3:]]
    os.makedirs(out, exist_ok=True)
    for f in sorted(os.listdir(tiles)):
        stem = os.path.splitext(f)[0]
        lq = Image.open(os.path.join(tiles, f)).convert('RGB')
        W, H = lq.width * 4, lq.height * 4
        cols = [('input (bicubic x4)', lq.resize((W, H), Image.BICUBIC))]
        for name, d in models:
            p = os.path.join(d, stem + '.png')
            if os.path.exists(p):
                im = Image.open(p).convert('RGB')
                if im.size != (W, H):
                    print(f'  {name} {stem}: {im.size}, resized to {(W, H)}')
                    im = im.resize((W, H), Image.LANCZOS)
                cols.append((name, im))
            else:
                print(f'  {name} {stem}: missing')
        cx, cy = busiest_crop(cols[-1][1], CROP)
        full = row([label(im, t) for t, im in cols])
        zoom = row([label(im.crop((cx, cy, cx + CROP, cy + CROP)).resize((CROP * ZOOM,) * 2, Image.NEAREST),
                          f'{t} (zoom {ZOOM}x)') for t, im in cols])
        sheet = Image.new('RGB', (max(full.width, zoom.width), full.height + zoom.height), BG)
        sheet.paste(full, (0, 0))
        sheet.paste(zoom, (0, full.height))
        sheet.save(os.path.join(out, f'ultra-{stem}.jpg'), quality=92)
        print('wrote', stem, [t for t, _ in cols], flush=True)


if __name__ == '__main__':
    main()
