#!/usr/bin/env python3
"""Is DC-AE-Lite a drop-in, faster decoder for SANA-Sprint's latents?

    python lite_vae_test.py SANA_REPO LITE_REPO OUT_DIR IMAGE...

Each image (1024 px square) is encoded with SANA-Sprint's own DC-AE, then decoded twice: with that
DC-AE and with DC-AE-Lite's decoder. Prints both decoders' shapes, sizes and times, and the PSNR
of each against the picture and against each other; writes OUT_DIR/<name>.jpg (picture, DC-AE,
Lite). On the CPU in fp32, as the phone runs it.
"""
import os
import sys
import time

import numpy as np
import torch
from diffusers import AutoencoderDC
from PIL import Image


def load(repo):
    for sub in ('vae', None):
        try:
            return AutoencoderDC.from_pretrained(repo, subfolder=sub, torch_dtype=torch.float32).eval()
        except Exception as e:  # noqa: BLE001 - the repo keeps it in a subfolder or at the root
            err = e
    raise err


def psnr(a, b):
    return 10 * np.log10(4 / max(float(np.mean((a - b) ** 2)), 1e-12))


def main():
    sana_repo, lite_repo, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    torch.set_grad_enabled(False)
    full, lite = load(sana_repo), load(lite_repo)
    for name, v in (('DC-AE (SANA-Sprint)', full), ('DC-AE-Lite', lite)):
        c = v.config
        n = sum(p.numel() for p in v.decoder.parameters())
        print(f'{name}: decoder {n / 1e6:.0f} M parameters, scaling_factor {c.scaling_factor}, latent {c.latent_channels}')
        print(f'  blocks {c.decoder_block_types} channels {c.decoder_block_out_channels} layers {c.decoder_layers_per_block}')
        print(f'  norms {c.decoder_norm_types} acts {c.decoder_act_fns} upsample {c.upsample_block_type}')
    for path in sys.argv[4:]:
        key = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path).convert('RGB').resize((1024, 1024), Image.LANCZOS)
        x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)[None].float() / 127.5 - 1
        z = full.encode(x, return_dict=False)[0]
        z = z.latent if hasattr(z, 'latent') else z
        t = time.time(); a = full.decode(z, return_dict=False)[0]; ta = time.time() - t
        t = time.time(); b = lite.decode(z, return_dict=False)[0]; tb = time.time() - t
        a, b = a.clamp(-1, 1).numpy(), b.clamp(-1, 1).numpy()
        print(f'{key}: DC-AE {ta:.1f} s, Lite {tb:.1f} s ({ta / tb:.2f}x); PSNR to the picture: DC-AE '
              f'{psnr(a, x.numpy()):.1f} dB, Lite {psnr(b, x.numpy()):.1f} dB; Lite against DC-AE {psnr(a, b):.1f} dB', flush=True)
        row = [img] + [Image.fromarray(((v[0].transpose(1, 2, 0) + 1) * 127.5).round().astype(np.uint8)) for v in (a, b)]
        sheet = Image.new('RGB', (3072, 1024))
        for i, im in enumerate(row):
            sheet.paste(im, (i * 1024, 0))
        sheet.save(os.path.join(out, key + '.jpg'), quality=92)


if __name__ == '__main__':
    main()
