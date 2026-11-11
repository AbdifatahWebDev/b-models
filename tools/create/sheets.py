#!/usr/bin/env python3
"""Side-by-side sheets for the Create / Swap quality test.

    python sheets.py ART_DIR RESULTS_DIR

ART_DIR holds the jobs' artifacts: sana-0.6b/, sana-1.6b/ (with inputs/), klein/, swap/, tryon/.
Writes create.jpg, swap.jpg, tryon.jpg and edits.jpg to RESULTS_DIR. Each cell is labelled with
the model and the CPU time it took on the runner (4 cores, fp32); a missing result (a job that
failed or did not finish) shows as a grey cell.
"""
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

from cases import EDITS, KLEIN_PROMPTS, PROMPTS, SWAPPERS, SWAPS, TRYONS

FONT = ImageFont.load_default(size=20)
SMALL = ImageFont.load_default(size=16)
BG, FG, DIM = (24, 24, 28), (240, 240, 240), (150, 150, 160)


def times(d):
    try:
        with open(os.path.join(d, "times.json")) as f:
            return json.load(f)
    except OSError:
        return {}


def cell(path, w, h, caption):
    """The image fitted into w x h (letterboxed), with a caption bar under it."""
    c = Image.new("RGB", (w, h + 28), BG)
    if path and os.path.exists(path):
        img = Image.open(path).convert("RGB")
        s = min(w / img.width, h / img.height)
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
        c.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
    elif path is None:
        return c
    else:
        ImageDraw.Draw(c).rectangle([0, 0, w - 1, h - 1], fill=(60, 60, 66))
        ImageDraw.Draw(c).text((w // 2, h // 2), "no result", fill=DIM, font=FONT, anchor="mm")
    ImageDraw.Draw(c).text((6, h + 4), caption.rstrip(" ·"), fill=FG, font=SMALL)
    return c


def sheet(title, header, rows, w, h, out):
    """rows: [(row label, [(path, caption), ...])], header: column names."""
    cols = max(len(r[1]) for r in rows)
    lab = 32
    sh = Image.new("RGB", (cols * (w + 8) + 8, 48 + 30 + len(rows) * (h + 28 + lab + 8)), BG)
    d = ImageDraw.Draw(sh)
    d.text((8, 10), title, fill=FG, font=ImageFont.load_default(size=26))
    for i, name in enumerate(header):
        d.text((8 + i * (w + 8), 50), name, fill=(255, 196, 120), font=FONT)
    y = 80
    for label, cells in rows:
        d.text((8, y + 6), label, fill=DIM, font=FONT)
        y += lab
        for i, (p, cap) in enumerate(cells):
            sh.paste(cell(p, w, h, cap), (8 + i * (w + 8), y))
        y += h + 28 + 8
    sh.save(out, quality=90)
    print(out, sh.size)


def sec(t, k):
    return f"{t[k]:.0f} s" if k in t else ""


def main():
    art, res = sys.argv[1], sys.argv[2]
    os.makedirs(res, exist_ok=True)
    s06, s16, kl, sw, tr = (os.path.join(art, n) for n in ("sana-0.6b", "sana-1.6b", "klein", "swap", "tryon"))
    t06, t16, tkl, tsw, ttr = (times(d) for d in (s06, s16, kl, sw, tr))
    inputs = os.path.join(s16, "inputs")

    rows = []
    for key, label, _ in PROMPTS:
        cells = [(os.path.join(s06, key + ".png"), f"SANA-Sprint 0.6B · 1024 · {sec(t06, key)}"),
                 (os.path.join(s16, key + ".png"), f"SANA-Sprint 1.6B · 1024 · {sec(t16, key)}")]
        if key in KLEIN_PROMPTS:
            cells.append((os.path.join(kl, key + ".png"), f"FLUX.2 klein 4B · 512 · {sec(tkl, key)}"))
        rows.append((label, cells))
    sheet("Create: text to image (CPU fp32, 4 cores; times per image)",
          ["SANA-Sprint 0.6B", "SANA-Sprint 1.6B", "FLUX.2 klein 4B (512 px)"], rows, 512, 512,
          os.path.join(res, "create.jpg"))

    rows = []
    for key, src, dst in SWAPS:
        cells = [(os.path.join(inputs, src + ".png"), "face from"), (os.path.join(inputs, dst + ".png"), "put on")]
        for m in SWAPPERS:
            cells.append((os.path.join(sw, m, key + ".png"), f"{m} + GFPGAN · {sec(tsw, f'{m}/{key}')}"))
        cells.append((os.path.join(sw, "hyperswap_1a_256_raw", key + ".png"), "hyperswap_1a alone"))
        if any(e[0] == key for e in EDITS):
            cells.append((os.path.join(kl, f"edit_{key}.png"), f"FLUX.2 klein · {sec(tkl, 'edit_' + key)}"))
        rows.append((f"{src} -> {dst}", cells))
    sheet("Face swap (FaceFusion swappers, 512 px boost; times include loading)",
          ["Face from", "Put on", *[f"{m}+GFPGAN" for m in SWAPPERS], "hyperswap alone", "FLUX.2 klein"],
          rows, 360, 360, os.path.join(res, "swap.jpg"))

    rows = []
    for key, person, garment, category, _ in TRYONS:
        src = os.path.join(tr, "inputs") if person.startswith("fashn") else inputs
        cells = [(os.path.join(src, person + ".png"), "person"),
                 (os.path.join(src, garment + ".png"), f"garment ({category})"),
                 (os.path.join(tr, key + ".png"), f"FASHN VTON v1.5 · {sec(ttr, key)}")]
        if any(e[0] == key for e in EDITS):
            cells.append((os.path.join(kl, f"edit_{key}.png"), f"FLUX.2 klein · {sec(tkl, 'edit_' + key)}"))
        rows.append((key.replace("_", " "), cells))
    sheet("Dress: try-on (CPU fp32, 4 cores)", ["Person", "Garment", "FASHN VTON v1.5", "FLUX.2 klein (512 px)"],
          rows, 384, 512, os.path.join(res, "tryon.jpg"))

    rows = []
    for key, label, srcs, instruction in EDITS:
        cells = []
        for s in srcs:
            p = os.path.join(s16, s[5:] + ".png") if s.startswith("sana:") else os.path.join(inputs, s + ".png")
            cells.append((p, "input"))
        while len(cells) < 2:
            cells.append((None, ""))
        cells.append((os.path.join(kl, f"edit_{key}.png"), f"FLUX.2 klein · {sec(tkl, 'edit_' + key)}"))
        rows.append((f"{label}: {instruction[:90]}{"..." if len(instruction) > 90 else ""}", cells))
    sheet("Edit with FLUX.2 klein 4B (512 px, 4 steps)", ["Image 1", "Image 2", "Result"], rows, 448, 448,
          os.path.join(res, "edits.jpg"))


if __name__ == "__main__":
    main()
