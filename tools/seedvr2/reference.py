#!/usr/bin/env python3
"""The check's truth for tools/seedvr2/convert.py: SeedVR2-3B's one-step path for one 128 px tile,
run by the community code's own modules (github.com/numz/ComfyUI-SeedVR2_VideoUpscaler, the code
the Ultra test ran), in fp32 on the CPU, on the same tiles and noise as convert.py --check.

    python tools/seedvr2/reference.py --numz CLI_DIR --dit DIT.safetensors --vae VAE.safetensors \\
        --pos-emb pos_emb.pt --tiles TILES --out REF_DIR [--tiny]

Writes REF_DIR/<tile>.npz with lat (the scaled VAE latent), v (the DiT velocity) and out (the colour
fixed 512 px result in [-1, 1]), each channels first. --tiny reads the small random model that
convert.py --save-tiny wrote and sizes the modules from it.

The steps are the CLI's (src/core/generation_phases.py and infer.py) for a single image: torchvision
bicubic resize with antialias to 512 px, VAE encode (posterior mode) x 0.9152, NaDiT on
[noise, latent, 1] with the positive prompt embedding at t = 1000 (cfg 1, one step), x0 = noise - v,
VAE decode of x0 / 0.9152, wavelet colour fix against the resized input. The noise is convert.py's
fixed tile noise instead of the CLI's seeded one.
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_into(module, path):
    """Copies a safetensors file into [module] tensor by tensor (fp32; no second full copy)."""
    from safetensors import safe_open
    own = dict(module.named_parameters())
    own.update(dict(module.named_buffers()))
    loaded, unexpected = set(), []
    with safe_open(path, framework='pt') as f:
        for k in f.keys():
            if k not in own:
                unexpected.append(k)
                continue
            t = f.get_tensor(k)
            if own[k].shape != t.shape:
                raise SystemExit(f'{path}: {k} is {tuple(t.shape)}, the module has {tuple(own[k].shape)}')
            with torch.no_grad():
                own[k].copy_(t.float())
            loaded.add(k)
    missing = [k for k in own if k not in loaded and 'freqs' not in k and 'cached' not in k]
    print(f'{os.path.basename(path)}: {len(loaded)} tensors; missing {missing[:8]}{"..." if len(missing) > 8 else ""}; '
          f'unexpected {unexpected[:8]}', flush=True)
    if missing:
        raise SystemExit('weights missing from the file')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--numz', required=True)
    ap.add_argument('--dit', required=True)
    ap.add_argument('--vae', required=True)
    ap.add_argument('--pos-emb', required=True)
    ap.add_argument('--tiles', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--count', type=int, default=8)
    ap.add_argument('--tiny', action='store_true')
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    sys.path.insert(0, HERE)
    import convert as C
    sys.path.insert(0, os.path.abspath(a.numz))
    from omegaconf import OmegaConf
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TVF
    from src.models.dit_3b.nadit import NaDiT
    from src.models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper
    from src.utils.color_fix import wavelet_reconstruction

    os.makedirs(a.out, exist_ok=True)
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    from safetensors import safe_open
    with safe_open(a.dit, framework='pt') as f:
        shapes = {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}
    with safe_open(a.vae, framework='pt') as f:
        vae_shapes = {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}

    cfg = OmegaConf.load(os.path.join(a.numz, 'configs_3b', 'main.yaml'))
    mc = cfg.dit.model
    if a.tiny:
        dim = shapes['vid_in.proj.weight'][0]
        hd = shapes['blocks.0.attn.norm_q.vid.weight'][0]
        mc.vid_dim = dim
        mc.head_dim = hd
        mc.heads = shapes['blocks.0.attn.proj_qkv.vid.weight'][0] // 3 // hd
        mc.num_layers = 1 + max(int(k.split('.')[1]) for k in shapes if k.startswith('blocks.'))
        mc.mm_layers = sum(1 for i in range(mc.num_layers) if f'blocks.{i}.attn.proj_qkv.vid.weight' in shapes)
        mc.txt_in_dim = shapes['txt_in.weight'][1]
        mc.rope_dim = hd
    kw = OmegaConf.to_container(mc, resolve=True)
    kw.pop('__object__', None)
    dit = NaDiT(**kw).eval()
    load_into(dit, a.dit)

    vcfg = OmegaConf.to_container(OmegaConf.load(os.path.join(
        a.numz, 'src', 'models', 'video_vae_v3', 's8_c16_t4_inflation_sd3.yaml')), resolve=True)
    if a.tiny:
        vcfg['block_out_channels'] = [vae_shapes[f'encoder.down_blocks.{i}.resnets.0.conv2.weight'][0] for i in range(4)]
    vae = VideoAutoencoderKLWrapper(**vcfg, freeze_encoder=False).eval()
    load_into(vae, a.vae)

    pos = torch.load(a.pos_emb, map_location='cpu', weights_only=True).float()
    noise = C.tile_noise()                                           # (1, 16, 64, 64)
    noise_cl = noise[0].permute(1, 2, 0)                             # (64, 64, 16)
    for name, x in C.tiles_raw(a.tiles, a.count):
        # The CLI's input transform: NaResize (torchvision bicubic, antialias), clamp, normalize.
        r = TVF.resize((x + 1) / 2, C.TILE * C.SCALE, InterpolationMode.BICUBIC, antialias=True).clamp(0, 1) * 2 - 1
        mine = C.upscale(x)
        print(f'{name}: input resize, convert.upscale vs torchvision max {float((mine - r).abs().max()):.1e}', flush=True)
        z = vae.encode(r).latent                                     # (1, 16, 64, 64), posterior mode
        lat = z * C.VAE_SCALING
        lat_cl = lat[0].permute(1, 2, 0)
        vid = torch.cat([noise_cl, lat_cl, torch.ones_like(lat_cl[..., :1])], dim=-1).reshape(-1, noise_cl.shape[-1] * 2 + 1)
        side = noise_cl.shape[0]
        out = dit(vid=vid, txt=pos, vid_shape=torch.tensor([[1, side, side]]), txt_shape=torch.tensor([[pos.shape[0]]]),
                  timestep=torch.tensor([C.T_STEP])).vid_sample
        v = out.reshape(side, side, -1).permute(2, 0, 1)[None]
        x0 = noise - v
        dec = vae.decode(x0 / C.VAE_SCALING).sample
        res = wavelet_reconstruction(dec, r).clamp(-1, 1)
        np.savez(os.path.join(a.out, name + '.npz'), lat=lat[0].numpy(), v=v[0].numpy(), out=res[0].numpy())
        print(f'{name}: done', flush=True)


if __name__ == '__main__':
    main()
