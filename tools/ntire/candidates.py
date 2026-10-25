#!/usr/bin/env python3
"""Runs NTIRE 2026 Mobile Real-World SR entries next to the app's models, for a look and a time.

    tools/ntire/candidates.py fetch  ZOO_DIR                   # weights from the challenge's Drive folder
    tools/ntire/candidates.py run    CHALLENGE_REPO ZOO_DIR INPUTS OUT_DIR
    tools/ntire/candidates.py sheets INPUTS MODELS_DIR CANDIDATES_DIR OUT_DIR  # next to x4plus, AdcSR

Entries (github.com/jiatongli2024/NTIRE2026_Mobile_RealWorld_ImageSR, report arXiv 2604.17306):
  * Antman (2nd): Real-ESRGAN x4plus's RRDBNet, fine-tuned. Same network as the app's default.
  * SamsungAICamera (3rd): one-step diffusion compressed for phones: SD 2.1 UNet at half width,
    no text or time inputs, then a TAESD decoder. About 12x faster than OSEDiff.
The diffusion entries that scored higher (TODSR, IMAG2006) run the full SD 2.1 UNet: slower than
AdcSR on a phone, so they aren't tried here.
"""
import os
import sys
import time

DRIVE = 'https://drive.google.com/drive/folders/1WvVqMqS8XxAsBRaYpPgHYTYe62agsP01'
TEAMS = {'antman': ('team06', 'antman'), 'samsung': ('team11', 'samsung')}

# Stable Diffusion 2.1-base UNet config (as in tools/adcsr/convert.py): Samsung's checkpoint
# replaces every weight, so only the layout is needed.
SD21_UNET = dict(
    sample_size=64, in_channels=4, out_channels=4, center_input_sample=False, flip_sin_to_cos=True, freq_shift=0,
    down_block_types=["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
    mid_block_type="UNetMidBlock2DCrossAttn",
    up_block_types=["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
    only_cross_attention=False, block_out_channels=[320, 640, 1280, 1280], layers_per_block=2,
    downsample_padding=1, mid_block_scale_factor=1, act_fn="silu", norm_num_groups=32, norm_eps=1e-5,
    cross_attention_dim=1024, attention_head_dim=[5, 10, 20, 20], dual_cross_attention=False,
    use_linear_projection=True, upcast_attention=False,
)


def list_drive(folder_id, path, keep, out):
    """Walks a public Drive folder through its embedded view (what gdown reads), descending only
    into folders [keep] accepts. gdown's own walk stops at the first folder that isn't shared
    (one in the challenge folder isn't), so the walk is done here, one folder at a time."""
    import re
    import bs4
    import requests
    res = requests.get('https://drive.google.com/embeddedfolderview', params={'id': folder_id}, timeout=60)
    if res.status_code != 200:
        print(f'  skipped {path or "/"}: status {res.status_code}')
        return
    for a in bs4.BeautifulSoup(res.text, 'html.parser').find_all('a'):
        href, name = a.get('href', ''), a.get_text(strip=True)
        rel = f'{path}/{name}' if path else name
        f = re.match(r'https://drive\.google\.com/file/d/([-\w]{25,})/view', href)
        if f:
            out.append((f.group(1), rel))
            continue
        d = re.search(r'/drive/folders/([-\w]{25,})', href)
        if d:
            print('  folder', rel)
            if keep(rel):
                list_drive(d.group(1), rel, keep, out)


def fetch(zoo):
    """Lists the challenge's Drive folder and downloads only the files of the teams in TEAMS (it
    also holds several Stable Diffusion copies, gigabytes each)."""
    import gdown
    folder = DRIVE.rsplit('/', 1)[-1]

    def wanted(rel):
        rel = rel.lower()
        return any(any(k in rel for k in keys) for keys in TEAMS.values())

    files = []
    print('Drive folder:')
    list_drive(folder, '', wanted, files)
    for fid, rel in files:
        print('  file', rel)
    for fid, rel in files:
        low = rel.lower()
        if not wanted(rel) or '/text_encoder/' in low or '/vae/' in low or 'tokenizer' in low:
            continue
        # The UNet weights of a bundled SD copy: only the layout is needed (SD21_UNET).
        if '/unet/' in low and not low.endswith('.json'):
            continue
        dest = os.path.join(zoo, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print('downloading', rel, flush=True)
        gdown.download(id=fid, output=dest, quiet=True)
        print('  ', os.path.getsize(dest) // 1024 // 1024, 'MB', flush=True)


def find(zoo, team, suffixes):
    keys = TEAMS[team]
    hits = []
    for root, _, names in os.walk(zoo):
        if not any(k in root.lower() for k in keys):
            continue
        hits += [os.path.join(root, n) for n in names if n.endswith(suffixes)]
    return sorted(hits, key=os.path.getsize, reverse=True)


def load_image(path):
    import numpy as np
    import torch
    from PIL import Image
    im = np.asarray(Image.open(path).convert('RGB'), np.float32) / 255.0
    return torch.from_numpy(im.transpose(2, 0, 1)).unsqueeze(0)


def save_image(t, path):
    import numpy as np
    from PIL import Image
    a = (t.squeeze(0).clamp(0, 1).numpy().transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(a).save(path)


def antman(repo, zoo):
    import torch
    sys.path.insert(0, repo)
    from models.team06_Antman.model import RRDBNet
    ckpt = find(zoo, 'antman', ('.pth', '.pt'))[0]
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    for key in ('params_ema', 'params', 'state_dict'):
        if isinstance(state, dict) and isinstance(state.get(key), dict):
            state = state[key]
            break
    net = RRDBNet()
    net.load_state_dict(state, strict=True)
    print('antman:', ckpt)
    return net.eval()


def samsung(repo, zoo):
    import torch
    from diffusers import UNet2DConditionModel
    sys.path.insert(0, repo)
    from models.team11_SamsungAKCamera.model import SRNet
    ckpt = find(zoo, 'samsung', ('.pth', '.pt'))[0]
    model = SRNet(unet=None)
    model._initialize_modules(UNet2DConditionModel(**SD21_UNET), ckpt)
    print('samsung:', ckpt, sum(p.numel() for p in model.parameters()) // 1_000_000, 'M parameters')
    return model.eval()


def run(repo, zoo, inputs, out):
    import torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(os.cpu_count() or 4)
    os.makedirs(out, exist_ok=True)
    times = []
    for name, build, multiple in (('antman', antman, 1), ('samsung', samsung, 16)):
        try:
            model = build(repo, zoo)
        except Exception as e:  # one entry failing shouldn't stop the other
            print(f'{name}: could not load: {e!r}')
            times.append(f'{name}: failed to load ({e!r})')
            continue
        os.makedirs(os.path.join(out, name), exist_ok=True)
        for f in sorted(os.listdir(inputs)):
            x = load_image(os.path.join(inputs, f))
            h, w = x.shape[-2:]
            x = x[..., : h // multiple * multiple, : w // multiple * multiple]
            t = time.time()
            y = model(x)
            dt = time.time() - t
            save_image(y, os.path.join(out, name, os.path.splitext(f)[0] + '.png'))
            times.append(f'{name} {f} {x.shape[-1]}x{x.shape[-2]}: {dt:.1f} s')
            print(times[-1], flush=True)
        del model
    with open(os.path.join(out, 'candidate-times.txt'), 'w') as fh:
        fh.write('\n'.join(times) + '\n')


def sheets(inputs, models, cands, out):
    """One sheet per photo: input, the app's Real-ESRGAN and AdcSR, then the candidates."""
    from PIL import Image
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'compare'))
    import compare as C
    esr = C.net(models, 'realesrgan-x4plus', fp32=True)
    adc = C.net(models, 'adcsr-x4', fp32=True, lean=True)
    os.makedirs(out, exist_ok=True)
    times = []
    for f in sorted(os.listdir(inputs)):
        stem = os.path.splitext(f)[0]
        lq = Image.open(os.path.join(inputs, f)).convert('RGB')
        W, H = lq.size[0] * 4, lq.size[1] * 4
        t = time.time()
        a = C.adcsr(adc, lq)
        t_adc = time.time() - t
        times.append(f'adcsr {f}: {t_adc:.1f} s')
        cols = [('input (bicubic)', lq.resize((W, H), Image.BICUBIC)),
                ('Real-ESRGAN x4plus (app)', C.realesrgan(esr, lq)),
                (f'AdcSR (app, {t_adc:.0f} s here)', a)]
        for name, title in (('antman', 'Antman: Real-ESRGAN fine-tuned'), ('samsung', 'Samsung: mobile diffusion')):
            p = os.path.join(cands, name, stem + '.png')
            if os.path.exists(p):
                cols.append((title, Image.open(p).convert('RGB').resize((W, H), Image.BICUBIC)))
        full = [C.label(im, t) for t, im in cols]
        crop = min(256, W, H)
        cx, cy = C.busiest_crop(cols[2][1], crop)
        crops = [C.label(im.crop((cx, cy, cx + crop, cy + crop)).resize((crop * 2, crop * 2), Image.NEAREST), t + ' (crop 2x)')
                 for t, im in cols]
        rows = []
        for cells in (full, crops):
            row = Image.new('RGB', (sum(i.width for i in cells), max(i.height for i in cells)), (20, 20, 24))
            x = 0
            for i in cells:
                row.paste(i, (x, 0))
                x += i.width
            rows.append(row)
        sheet = Image.new('RGB', (max(r.width for r in rows), sum(r.height for r in rows)), (20, 20, 24))
        sheet.paste(rows[0], (0, 0))
        sheet.paste(rows[1], (0, rows[0].height))
        if sheet.width > 3600:
            sheet = sheet.resize((3600, round(sheet.height * 3600 / sheet.width)), Image.LANCZOS)
        sheet.save(os.path.join(out, f'ntire-{stem}.jpg'), quality=90)
        print('wrote', stem, flush=True)
    with open(os.path.join(out, 'adcsr-times.txt'), 'w') as fh:
        fh.write('\n'.join(times) + '\n')


if __name__ == '__main__':
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == 'fetch':
        fetch(*args)
    elif cmd == 'run':
        run(*args)
    elif cmd == 'sheets':
        sheets(*args)
    else:
        sys.exit(__doc__)
