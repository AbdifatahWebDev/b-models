#!/usr/bin/env python3
"""Side-by-side comparison of upscalers on small low-quality photos (4x).

Usage: tools/compare/compare.py INPUT_DIR MODELS_DIR VOSR_OUT_DIR OUT_DIR

Columns: bicubic, Real-ESRGAN x4plus and AdcSR (the ncnn files the app ships, run the way the
app runs them: AdcSR in fixed 128 px tiles with 16 px of context), and VOSR 0.5B one-step
(already run by its own script into VOSR_OUT_DIR). Writes one PNG per photo, the full images
on top and a 1:1 crop of the busiest area below.
"""
import os
import sys
import time

import ncnn
import numpy as np
from PIL import Image, ImageDraw, ImageOps


def net(models, base, fp32=True, lean=False):
    n = ncnn.Net()
    # fp16 storage only: x86 runners have no fp16 arithmetic (it gave Real-ESRGAN a blue screen).
    n.opt.use_fp16_storage = n.opt.use_fp16_packed = not fp32
    n.opt.use_fp16_arithmetic = False
    if lean:
        n.opt.use_winograd_convolution = False
        n.opt.use_sgemm_convolution = False
    n.opt.num_threads = os.cpu_count() or 4
    n.load_param(os.path.join(models, base + ".param"))
    n.load_model(os.path.join(models, base + ".bin"))
    return n


def run(n, chw, inp, out):
    # ncnn.Mat borrows the array's memory: keep the array alive until extract() has run (a
    # temporary here was freed first and the models read garbage).
    arr = np.ascontiguousarray(chw, dtype=np.float32)
    mat = ncnn.Mat(arr)
    ex = n.create_extractor()
    ex.input(inp, mat)
    _, o = ex.extract(out)
    result = np.array(o).copy()
    del ex, mat, arr
    return result


def realesrgan(n, img):
    x = np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0
    y = run(n, x, "data", "output")
    return Image.fromarray((np.clip(y, 0, 1) * 255 + 0.5).astype(np.uint8).transpose(1, 2, 0))


def adcsr(n, img, fixed=128, ctx=16):
    """Like the app's engine: sliding 128 px windows, cores of 96 px, feathered seams, then the
    per-channel colour match to the input."""
    src = np.asarray(img, np.float32)
    h, w, _ = src.shape
    s = 4
    pad_h, pad_w = max(0, fixed - h), max(0, fixed - w)
    padded = np.pad(src, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    ph, pw, _ = padded.shape
    acc = np.zeros((ph * s, pw * s, 3), np.float64)
    wsum = np.zeros((ph * s, pw * s, 1), np.float64)
    core = fixed - 2 * ctx
    ramp = np.minimum(np.arange(fixed * s) + 0.5, fixed * s - np.arange(fixed * s) - 0.5)
    ramp = np.clip(ramp / (ctx * s), 0.05, 1.0)
    weight = np.outer(ramp, ramp)[..., None]
    ys = list(range(0, max(1, ph - fixed) + 1, core)) or [0]
    xs = list(range(0, max(1, pw - fixed) + 1, core)) or [0]
    if ys[-1] + fixed < ph:
        ys.append(ph - fixed)
    if xs[-1] + fixed < pw:
        xs.append(pw - fixed)
    for y0 in ys:
        for x0 in xs:
            tile = padded[y0:y0 + fixed, x0:x0 + fixed]
            x = tile.transpose(2, 0, 1) / 127.5 - 1.0
            y = run(n, x, "in0", "out0").transpose(1, 2, 0) * 127.5 + 127.5
            acc[y0 * s:(y0 + fixed) * s, x0 * s:(x0 + fixed) * s] += y * weight
            wsum[y0 * s:(y0 + fixed) * s, x0 * s:(x0 + fixed) * s] += weight
    out = (acc / np.maximum(wsum, 1e-6))[: h * s, : w * s]
    for c in range(3):
        o, i = out[..., c], src[..., c]
        out[..., c] = (o - o.mean()) / max(o.std(), 1e-6) * i.std() + i.mean()
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))


def busiest_crop(img, size):
    g = np.asarray(img.convert("L"), np.float32)
    best, at = -1.0, (0, 0)
    step = max(8, size // 4)
    for y in range(0, max(1, g.shape[0] - size + 1), step):
        for x in range(0, max(1, g.shape[1] - size + 1), step):
            v = float(np.abs(np.diff(g[y:y + size, x:x + size], axis=1)).mean())
            if v > best:
                best, at = v, (x, y)
    return at


def label(img, text):
    img = ImageOps.expand(img, border=(0, 28, 0, 0), fill=(20, 20, 24))
    ImageDraw.Draw(img).text((8, 6), text, fill=(240, 240, 240))
    return img


def main():
    inp, models, vosr_dir, out_dir = sys.argv[1:5]
    os.makedirs(out_dir, exist_ok=True)
    esr = net(models, "realesrgan-x4plus", fp32=True)
    adc = net(models, "adcsr-x4", fp32=True, lean=True)
    times = []
    for name in sorted(os.listdir(inp)):
        stem = os.path.splitext(name)[0]
        lq = Image.open(os.path.join(inp, name)).convert("RGB")
        W, H = lq.size[0] * 4, lq.size[1] * 4
        t0 = time.time()
        a = adcsr(adc, lq)
        t_adc = time.time() - t0
        cols = [
            ("bicubic", lq.resize((W, H), Image.BICUBIC)),
            ("Real-ESRGAN x4plus (app)", realesrgan(esr, lq)),
            (f"AdcSR (app, {t_adc:.0f} s here)", a),
        ]
        vp = os.path.join(vosr_dir, stem + ".png")
        if os.path.exists(vp):
            cols.append(("VOSR 0.5B one-step", Image.open(vp).convert("RGB").resize((W, H), Image.BICUBIC)))
        times.append(f"{name}: {lq.size[0]}x{lq.size[1]}, AdcSR {t_adc:.1f} s")
        full = [label(im, t) for t, im in cols]
        crop = min(256, W, H)
        cx, cy = busiest_crop(cols[2][1], crop)
        crops = [label(im.crop((cx, cy, cx + crop, cy + crop)).resize((crop * 2, crop * 2), Image.NEAREST), t + " (crop 2x)") for t, im in cols]
        row1 = Image.new("RGB", (sum(i.width for i in full), max(i.height for i in full)), (20, 20, 24))
        x = 0
        for i in full:
            row1.paste(i, (x, 0))
            x += i.width
        row2 = Image.new("RGB", (sum(i.width for i in crops), max(i.height for i in crops)), (20, 20, 24))
        x = 0
        for i in crops:
            row2.paste(i, (x, 0))
            x += i.width
        sheet = Image.new("RGB", (max(row1.width, row2.width), row1.height + row2.height), (20, 20, 24))
        sheet.paste(row1, (0, 0))
        sheet.paste(row2, (0, row1.height))
        if sheet.width > 3200:
            sheet = sheet.resize((3200, round(sheet.height * 3200 / sheet.width)), Image.LANCZOS)
        sheet.save(os.path.join(out_dir, f"compare-{stem}.png"))
        print("wrote", stem)
    with open(os.path.join(out_dir, "compare-times.txt"), "w") as f:
        f.write("\n".join(times) + "\n")


if __name__ == "__main__":
    main()
