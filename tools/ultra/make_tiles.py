#!/usr/bin/env python3
"""Test tiles for the Ultra test: 128 px crops, the size of one Max tile on the phone.

    tools/ultra/make_tiles.py OUT_DIR SOURCE_DIR...

Photos larger than 240 px are first shrunk to 240 px on the long side, like a small phone photo;
smaller ones (already real low-resolution photos) are cropped as they are. The crop is the upper
middle, where the faces are in portraits. Grey and transparent images are skipped.
"""
import glob
import os
import sys

from PIL import Image

SIZE = 128
SHRINK_TO = 240


def main():
    out, sources = sys.argv[1], sys.argv[2:]
    os.makedirs(out, exist_ok=True)
    for src in sources:
        for p in sorted(glob.glob(os.path.join(src, '*'))):
            if os.path.splitext(p)[1].lower() not in ('.png', '.jpg', '.jpeg'):
                continue
            im = Image.open(p)
            if im.mode != 'RGB':
                continue
            f = SHRINK_TO / max(im.size)
            if f < 1:
                im = im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS)
            w, h = im.size
            if min(w, h) < SIZE:
                continue
            x0, y0 = (w - SIZE) // 2, max(0, min(h - SIZE, h // 4 - SIZE // 2))
            stem = os.path.splitext(os.path.basename(p))[0]
            im.crop((x0, y0, x0 + SIZE, y0 + SIZE)).save(os.path.join(out, stem + '.png'))
            print(stem, 'from', p, (w, h))


if __name__ == '__main__':
    main()
