#!/usr/bin/env python3
"""Converts TSD-SR (one-step diffusion super-resolution on Stable Diffusion 3 Medium, CVPR 2025)
to one ncnn model for Brekwol Upscale's experimental Max mode.

    python tools/tsdsr/convert.py --sd3 SD3_DIR --lora LORA_DIR --embeds EMBED_DIR --out OUT \
        [--check FACES_OR_PHOTOS_DIR] [--blocks-per-piece 2]
    python tools/tsdsr/convert.py --tiny --out OUT --check DIR     # random small model: tests the
                                                                    # conversion, no weights needed

The ncnn model (tsdsr-x4.param / .bin):
    in0   128x128 RGB in [-1, 1] (one tile of the small photo)
    out0  512x512 RGB in [-1, 1]
One step, as in test/test_tsdsr.py of github.com/Microtreei/TSD-SR: bicubic 4x, SD3 VAE encoder
(with TSD-SR's LoRA), latent x0 = mean * scaling, x0 - transformer(x0, t=1000, the default prompt
embeddings), SD3 VAE decoder.

Changes from that script, all exact except the first:
  * The VAE posterior's mean is used, not a random sample (the app must be repeatable).
  * Both LoRAs are merged into the weights (W += B @ A; alpha = rank, so the scale is 1).
  * Everything that depends only on the fixed timestep, pooled embedding and prompt (time
    embedding, every block's AdaLN shift/scale/gate, the projected prompt) is computed once here
    and stored as constants.
  * Attention is written as matmul + softmax, and the output unpatchify as a pixel shuffle with
    reordered proj_out rows: ncnn has neither SDPA in every build nor 6-D tensors.

pnnx can't hold a 2B model in memory, so the network is exported in pieces (encoder + patch
embedding, groups of transformer blocks, output + decoder) and the ncnn graphs joined into one
(merge()). --check runs the PyTorch model and the joined ncnn model on 128 px tiles of the photos
in DIR, reports the error and the largest activation (fp16 storage on the phone needs < 65504),
and writes side-by-side images.

TSD-SR: Apache-2.0. SD3 Medium: Stability AI Non-Commercial Research Community License — the
converted files must not be redistributed publicly; they go to a private place for personal use.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TILE = 128
SCALE = 4
TIMESTEP = 1000.0


# ---------------------------------------------------------------------------------------------
# Loading

def merge_lora(model, path):
    """Adds TSD-SR's LoRA deltas into [model]'s weights. Keys look like
    transformer.<module>.lora_A.weight (the authors' loader prefixes "transformer." for the VAE too)."""
    from safetensors.torch import load_file
    sd = load_file(path)
    pairs = {}
    for k, v in sd.items():
        k = k.removeprefix('transformer.')
        for part in ('.lora_A.weight', '.lora_B.weight'):
            if k.endswith(part):
                pairs.setdefault(k[: -len(part)], {})[part[6]] = v.float()
    modules = dict(model.named_modules())
    done = 0
    for name, ab in pairs.items():
        m = modules.get(name)
        if m is None:
            raise SystemExit(f'LoRA module {name} not in the model')
        a, b = ab['A'], ab['B']
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                m.weight += b @ a
            elif isinstance(m, nn.Conv2d):
                m.weight += torch.einsum('or,rikl->oikl', b[:, :, 0, 0], a)
            else:
                raise SystemExit(f'LoRA on {type(m).__name__} ({name})')
        done += 1
    print(f'{os.path.basename(path)}: merged {done} LoRA pairs')


def load_real(sd3, lora, embeds):
    from diffusers import AutoencoderKL, SD3Transformer2DModel
    tr = SD3Transformer2DModel.from_pretrained(sd3, subfolder='transformer', torch_dtype=torch.float32)
    vae = AutoencoderKL.from_pretrained(sd3, subfolder='vae', torch_dtype=torch.float32)
    merge_lora(tr, os.path.join(lora, 'transformer.safetensors'))
    merge_lora(vae, os.path.join(lora, 'vae.safetensors'))
    prompt = torch.load(os.path.join(embeds, 'prompt_embeds.pt'), map_location='cpu').float()
    pooled = torch.load(os.path.join(embeds, 'pool_embeds.pt'), map_location='cpu').float()
    return tr.eval(), vae.eval(), prompt, pooled


def load_tiny():
    """Random weights in SD3's layout, small: tests the conversion without the 8 GB download."""
    from diffusers import AutoencoderKL, SD3Transformer2DModel
    torch.manual_seed(0)
    tr = SD3Transformer2DModel(sample_size=64, patch_size=2, in_channels=16, num_layers=3, attention_head_dim=64,
                               num_attention_heads=2, joint_attention_dim=96, caption_projection_dim=128,
                               pooled_projection_dim=32, out_channels=16, pos_embed_max_size=96)
    vae = AutoencoderKL(in_channels=3, out_channels=3, down_block_types=('DownEncoderBlock2D',) * 4,
                        up_block_types=('UpDecoderBlock2D',) * 4, block_out_channels=(32, 32, 64, 64),
                        layers_per_block=1, latent_channels=16, norm_num_groups=32, scaling_factor=1.5305,
                        shift_factor=0.0609, use_quant_conv=False, use_post_quant_conv=False)
    # Random weights explode through many layers; keep the activations in a realistic range.
    with torch.no_grad():
        for p in list(tr.parameters()) + list(vae.parameters()):
            p.mul_(0.5)
    prompt = torch.randn(1, 154, 96)
    pooled = torch.randn(1, 32)
    return tr.eval(), vae.eval(), prompt, pooled


# ---------------------------------------------------------------------------------------------
# The network, rewritten in pieces with the constants folded in

def attention(attn, h, c, n_img):
    """SD3 joint attention ([image, prompt] tokens together), as JointAttnProcessor2_0."""
    for name in ('norm_q', 'norm_k', 'norm_added_q', 'norm_added_k'):
        if getattr(attn, name, None) is not None:
            raise SystemExit('qk-norm (SD3.5) is not handled')
    heads = attn.heads
    q = torch.cat([attn.to_q(h), attn.add_q_proj(c)], dim=1)
    k = torch.cat([attn.to_k(h), attn.add_k_proj(c)], dim=1)
    v = torch.cat([attn.to_v(h), attn.add_v_proj(c)], dim=1)
    d = q.shape[-1] // heads
    q = q.view(1, -1, heads, d).transpose(1, 2)
    k = k.view(1, -1, heads, d).transpose(1, 2)
    v = v.view(1, -1, heads, d).transpose(1, 2)
    w = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * (d ** -0.5), dim=-1)
    o = torch.matmul(w, v).transpose(1, 2).reshape(1, -1, heads * d)
    oh, oc = o[:, :n_img], o[:, n_img:]
    oh = attn.to_out[0](oh)
    oc = attn.to_add_out(oc) if getattr(attn, 'to_add_out', None) is not None and not attn.context_pre_only else None
    return oh, oc


def layer_norm(x, eps=1e-6):
    return F.layer_norm(x, (x.shape[-1],), eps=eps)


class Block(nn.Module):
    """One JointTransformerBlock with its AdaLN modulation precomputed for the fixed temb."""

    def __init__(self, block, temb):
        super().__init__()
        assert not getattr(block, 'use_dual_attention', False)
        self.attn, self.ff = block.attn, block.ff
        self.pre_only = block.context_pre_only
        with torch.no_grad():
            m = block.norm1.linear(block.norm1.silu(temb)).chunk(6, dim=1)
            for name, t in zip(('shift_msa', 'scale_msa', 'gate_msa', 'shift_mlp', 'scale_mlp', 'gate_mlp'), m):
                self.register_buffer(name, t[:, None].clone())
            if self.pre_only:
                scale, shift = block.norm1_context.linear(block.norm1_context.silu(temb)).chunk(2, dim=1)
                self.register_buffer('c_scale', scale[:, None].clone())
                self.register_buffer('c_shift', shift[:, None].clone())
            else:
                self.ff_context = block.ff_context
                m = block.norm1_context.linear(block.norm1_context.silu(temb)).chunk(6, dim=1)
                for name, t in zip(('c_shift_msa', 'c_scale_msa', 'c_gate_msa', 'c_shift_mlp', 'c_scale_mlp', 'c_gate_mlp'), m):
                    self.register_buffer(name, t[:, None].clone())

    def forward(self, h, c):
        nh = layer_norm(h) * (1 + self.scale_msa) + self.shift_msa
        if self.pre_only:
            nc = layer_norm(c) * (1 + self.c_scale) + self.c_shift
        else:
            nc = layer_norm(c) * (1 + self.c_scale_msa) + self.c_shift_msa
        ah, ac = attention(self.attn, nh, nc, h.shape[1])
        h = h + self.gate_msa * ah
        h = h + self.gate_mlp * self.ff(layer_norm(h) * (1 + self.scale_mlp) + self.shift_mlp)
        if self.pre_only:
            return h, None
        c = c + self.c_gate_msa * ac
        c = c + self.c_gate_mlp * self.ff_context(layer_norm(c) * (1 + self.c_scale_mlp) + self.c_shift_mlp)
        return h, c


class Encode(nn.Module):
    """LR tile in [-1, 1] -> (x0 latent, image tokens)."""

    def __init__(self, tr, vae):
        super().__init__()
        self.encoder = vae.encoder
        assert getattr(vae, 'quant_conv', None) is None
        self.scaling = float(vae.config.scaling_factor)
        self.proj = tr.pos_embed.proj
        with torch.no_grad():
            lat = TILE * SCALE // 8
            z = torch.zeros(1, tr.config.in_channels, lat, lat)
            pos = tr.pos_embed(z) - self.proj.bias[None, None, :]
        self.register_buffer('pos', pos.clone())

    def forward(self, x):
        x = F.interpolate(x, scale_factor=SCALE, mode='bicubic', align_corners=False).clamp(-1, 1)
        moments = self.encoder(x)
        x0 = moments[:, : moments.shape[1] // 2] * self.scaling
        h = self.proj(x0).flatten(2).transpose(1, 2) + self.pos
        return x0, h


class Blocks(nn.Module):
    """A run of transformer blocks; the first piece starts from the constant projected prompt."""

    def __init__(self, blocks, c0=None):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        if c0 is not None:
            self.register_buffer('c0', c0.clone())
        self.first = c0 is not None

    def forward(self, h, c=None):
        if self.first:
            c = self.c0
        for b in self.blocks:
            h, c = b(h, c)
        return (h, c) if c is not None else h


class Decode(nn.Module):
    """(image tokens, x0) -> 512 px RGB in [-1, 1]."""

    def __init__(self, tr, vae, temb):
        super().__init__()
        self.decoder = vae.decoder
        assert getattr(vae, 'post_quant_conv', None) is None
        self.scaling = float(vae.config.scaling_factor)
        p, ch = tr.config.patch_size, tr.out_channels
        with torch.no_grad():
            scale, shift = tr.norm_out.linear(tr.norm_out.silu(temb)).chunk(2, dim=1)
            # proj_out rows are (p, q, c); pixel_shuffle wants (c, p, q).
            idx = torch.tensor([(pp * p + qq) * ch + cc for cc in range(ch) for pp in range(p) for qq in range(p)])
            self.proj = nn.Linear(tr.proj_out.in_features, tr.proj_out.out_features)
            self.proj.weight.copy_(tr.proj_out.weight[idx])
            self.proj.bias.copy_(tr.proj_out.bias[idx])
        self.register_buffer('scale', scale[:, None].clone())
        self.register_buffer('shift', shift[:, None].clone())
        self.p = p
        self.side = TILE * SCALE // 8 // p

    def forward(self, h, x0):
        h = self.proj(layer_norm(h) * (1 + self.scale) + self.shift)            # (1, N, C*p*p)
        pred = F.pixel_shuffle(h.transpose(1, 2).reshape(1, -1, self.side, self.side), self.p)
        z = (x0 - pred) / self.scaling
        return self.decoder(z).clamp(-1, 1)


class Pieces:
    def __init__(self, tr, vae, prompt, pooled, per_piece):
        with torch.no_grad():
            temb = tr.time_text_embed(torch.tensor([TIMESTEP]), pooled)
            c0 = tr.context_embedder(prompt)
        blocks = [Block(b, temb) for b in tr.transformer_blocks]
        self.encode = Encode(tr, vae).eval()
        self.groups = [Blocks(blocks[i:i + per_piece], c0 if i == 0 else None).eval()
                       for i in range(0, len(blocks), per_piece)]
        self.decode = Decode(tr, vae, temb).eval()

    def __call__(self, x):
        x0, h = self.encode(x)
        c = None
        for g in self.groups:
            out = g(h, c) if c is not None else g(h)
            h, c = out if isinstance(out, tuple) else (out, None)
        return self.decode(h, x0)


def reference(tr, vae, prompt, pooled, x):
    """test_tsdsr.py's main() for one 128 px tile, with the posterior mean: the check's truth."""
    with torch.no_grad():
        up = F.interpolate((x + 1) / 2, scale_factor=SCALE, mode='bicubic', align_corners=False) * 2 - 1
        up = up.clamp(-1, 1)
        x0 = vae.encode(up).latent_dist.mean * vae.config.scaling_factor
        pred = tr(hidden_states=x0, timestep=torch.tensor([TIMESTEP]), encoder_hidden_states=prompt,
                  pooled_projections=pooled, return_dict=False)[0]
        return vae.decode((x0 - pred) / vae.config.scaling_factor, return_dict=False)[0].clamp(-1, 1)


# ---------------------------------------------------------------------------------------------
# Export: one pnnx run per piece, then one joined ncnn graph

def piece_specs(p):
    """(name, module, example inputs, global input blobs, global output blobs) for each piece."""
    x = torch.rand(1, 3, TILE, TILE) * 2 - 1
    with torch.no_grad():
        x0, h = p.encode(x)
    specs = [('encode', p.encode, (x,), ['in0'], ['x0', 'h0'])]
    c = None
    for i, g in enumerate(p.groups):
        ins = (h,) if c is None else (h, c)
        gin = [f'h{i}'] + ([f'c{i}'] if c is not None else [])
        with torch.no_grad():
            out = g(*ins)
        h, c = out if isinstance(out, tuple) else (out, None)
        gout = [f'h{i + 1}'] + ([f'c{i + 1}'] if c is not None else [])
        specs.append((f'blocks{i}', g, ins, gin, gout))
    specs.append(('decode', p.decode, (h, x0), [f'h{len(p.groups)}', 'x0'], ['out0']))
    return specs


def read_param(path):
    with open(path) as f:
        lines = [l.split() for l in f.read().splitlines() if l.strip()]
    assert lines[0] == ['7767517'], path
    layers = []
    for t in lines[2:]:
        nin, nout = int(t[2]), int(t[3])
        layers.append({'type': t[0], 'name': t[1], 'in': t[4:4 + nin], 'out': t[4 + nin:4 + nin + nout],
                       'params': t[4 + nin + nout:]})
    return layers


def merge(pieces, out_param):
    """Joins the pieces' ncnn graphs. pieces: [(param path, global inputs, global outputs)]. A
    piece's inK/outK become the global names; its Input layers go, except the model's in0. Blobs
    used by more than one layer get a Split, as ncnn expects. The .bin files are concatenated in
    the same order (Input and Split layers have no weights)."""
    layers = []
    for idx, (param, gin, gout) in enumerate(pieces):
        def rename(b):
            if b.startswith('in') and b[2:].isdigit():
                return gin[int(b[2:])]
            if b.startswith('out') and b[3:].isdigit():
                return gout[int(b[3:])]
            return f'p{idx}_{b}'
        for l in read_param(param):
            if l['type'] == 'Input' and rename(l['out'][0]) != 'in0':
                continue
            layers.append({**l, 'name': f'p{idx}_{l["name"]}', 'in': [rename(b) for b in l['in']],
                           'out': [rename(b) for b in l['out']]})
    consumers = {}
    for li, l in enumerate(layers):
        for j, b in enumerate(l['in']):
            consumers.setdefault(b, []).append((li, j))
    inserts = {}
    for b, uses in consumers.items():
        if len(uses) < 2:
            continue
        names = [f'{b}_s{k}' for k in range(len(uses))]
        for (li, j), n in zip(uses, names):
            layers[li]['in'][j] = n
        producer = next(i for i, l in enumerate(layers) if b in l['out'])
        inserts.setdefault(producer, []).append({'type': 'Split', 'name': f'split_{b}', 'in': [b], 'out': names, 'params': []})
    final = []
    for i, l in enumerate(layers):
        final.append(l)
        final.extend(inserts.get(i, []))
    blobs = {b for l in final for b in l['out']}
    with open(out_param, 'w') as f:
        f.write(f'7767517\n{len(final)} {len(blobs)}\n')
        for l in final:
            f.write(' '.join([l['type'], l['name'], str(len(l['in'])), str(len(l['out']))] + l['in'] + l['out'] + l['params']) + '\n')
    return len(final)


def free(module):
    """Drops a piece's weights once it is exported: the full model doesn't fit next to pnnx."""
    for m in module.modules():
        for name, prm in list(m.named_parameters(recurse=False)):
            setattr(m, name, nn.Parameter(torch.empty(0), requires_grad=False))
        for name in list(m._buffers):
            m._buffers[name] = torch.empty(0)


def export(p, out_dir, keep=None, release=False):
    """Traces every piece, runs pnnx on each in its own process, joins them into tsdsr-x4.*.
    With [release], each piece's weights are freed after its export (the model is unusable after)."""
    import pnnx
    exe = shutil.which('pnnx') or os.path.join(os.path.dirname(pnnx.__file__), 'pnnx')
    work = keep or tempfile.mkdtemp()
    os.makedirs(work, exist_ok=True)
    joined = []
    specs = piece_specs(p)
    for name, module, ins, gin, gout in specs:
        with torch.no_grad():
            torch.jit.trace(module, ins, check_trace=False).save(os.path.join(work, f'{name}.pt'))
        shapes = 'inputshape=' + ','.join('[' + ','.join(map(str, t.shape)) + ']' for t in ins)
        r = subprocess.run([exe, f'{name}.pt', shapes, 'fp16=1'], cwd=work, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(os.path.join(work, f'{name}.ncnn.param')):
            print(r.stdout[-3000:], r.stderr[-3000:])
            raise SystemExit(f'pnnx failed on {name} (exit {r.returncode})')
        for ext in ('pt', 'pnnx.bin', 'pnnx.onnx'):
            fp = os.path.join(work, f'{name}.{ext}')
            if os.path.exists(fp):
                os.remove(fp)
        joined.append((os.path.join(work, f'{name}.ncnn.param'), gin, gout))
        print(f'  {name}: {os.path.getsize(os.path.join(work, f"{name}.ncnn.bin")) // 2**20} MB', flush=True)
        if release:
            free(module)
    n = merge(joined, os.path.join(out_dir, 'tsdsr-x4.param'))
    with open(os.path.join(out_dir, 'tsdsr-x4.bin'), 'wb') as out:
        for param, _, _ in joined:
            with open(param.replace('.param', '.bin'), 'rb') as f:
                shutil.copyfileobj(f, out)
            os.remove(param.replace('.param', '.bin'))  # disk: the runner can't hold two copies
    print(f'wrote tsdsr-x4.param ({n} layers) and tsdsr-x4.bin ({os.path.getsize(os.path.join(out_dir, "tsdsr-x4.bin")) // 2**20} MB)')
    if keep is None:
        shutil.rmtree(work)


# ---------------------------------------------------------------------------------------------
# Check

def tiles(folder, n=6):
    from PIL import Image
    out = []
    for name in sorted(os.listdir(folder)):
        im = Image.open(os.path.join(folder, name)).convert('RGB')
        if min(im.size) < TILE:
            continue
        x0, y0 = (im.width - TILE) // 2, (im.height - TILE) // 2
        t = np.asarray(im.crop((x0, y0, x0 + TILE, y0 + TILE)), np.float32).transpose(2, 0, 1) / 127.5 - 1
        out.append((name, torch.from_numpy(t)[None]))
        if len(out) == n:
            break
    return out


def max_activation(p, x):
    """Largest |value| any module produces: fp16 storage on the phone overflows above 65504."""
    worst = [0.0, '']

    def hook(name):
        def f(_, __, out):
            for t in (out if isinstance(out, tuple) else (out,)):
                if isinstance(t, torch.Tensor):
                    v = float(t.abs().max())
                    if v > worst[0]:
                        worst[:] = [v, name]
        return f
    handles = []
    for root in [p.encode, *p.groups, p.decode]:
        for name, m in root.named_modules():
            handles.append(m.register_forward_hook(hook(f'{type(root).__name__}.{name}')))
    with torch.no_grad():
        p(x)
    for h in handles:
        h.remove()
    return worst


def run_ncnn(out_dir, x):
    import ncnn
    n = ncnn.Net()
    n.opt.use_fp16_storage = n.opt.use_fp16_packed = False
    n.opt.use_fp16_arithmetic = False
    n.opt.num_threads = os.cpu_count() or 4
    n.load_param(os.path.join(out_dir, 'tsdsr-x4.param'))
    n.load_model(os.path.join(out_dir, 'tsdsr-x4.bin'))
    a = np.ascontiguousarray(x[0].numpy())
    m = ncnn.Mat(a)
    ex = n.create_extractor()
    ex.input('in0', m)
    _, o = ex.extract('out0')
    y = np.array(o).copy()
    del ex, m, a, n
    return y


def references(tr, vae, prompt, pooled, p, folder):
    """PyTorch results for the check, taken before the export frees the weights."""
    refs = []
    for name, x in tiles(folder):
        ref = reference(tr, vae, prompt, pooled, x)[0].numpy()
        with torch.no_grad():
            mine = p(x)[0].numpy()
        act, where = max_activation(p, x)
        refs.append((name, x, ref, float(np.abs(mine - ref).max()), act, where))
        print(f'  {name}: pieces vs reference max {refs[-1][3]:.1e}; largest activation {act:.0f} ({where})', flush=True)
    return refs


def check(refs, out_dir):
    from PIL import Image
    rows, worst, report = [], 0.0, []
    for name, x, ref, e_split, act, where in refs:
        got = run_ncnn(out_dir, x)
        d = np.abs(got - ref)
        line = (f'{name}: pieces vs reference max {e_split:.1e}; ncnn mean {d.mean():.1e} max {d.max():.1e} '
                f'on [-1, 1]; largest activation {act:.0f} ({where})')
        print('  ' + line, flush=True)
        report.append(line)
        worst = max(worst, float(d.mean()))
        img = lambda a: Image.fromarray(np.clip((a.transpose(1, 2, 0) + 1) * 127.5 + 0.5, 0, 255).astype(np.uint8))
        lr = img(x[0].numpy()).resize((TILE * SCALE,) * 2, Image.BICUBIC)
        row = Image.new('RGB', (TILE * SCALE * 3, TILE * SCALE))
        for k, im in enumerate((lr, img(ref), img(got))):
            row.paste(im, (k * TILE * SCALE, 0))
        rows.append(row)
        os.makedirs(os.path.join(out_dir, 'ncnn-tiles'), exist_ok=True)
        img(got).save(os.path.join(out_dir, 'ncnn-tiles', os.path.splitext(name)[0] + '.png'))
    if rows:
        sheet = Image.new('RGB', (rows[0].width, sum(r.height for r in rows)))
        for k, r in enumerate(rows):
            sheet.paste(r, (0, k * r.height))
        sheet.save(os.path.join(out_dir, 'check-tsdsr.jpg'), quality=88)
    with open(os.path.join(out_dir, 'check-tsdsr.txt'), 'w') as f:
        f.write('\n'.join(report) + f'\nworst mean error {worst:.3g}\n')
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sd3')
    ap.add_argument('--lora')
    ap.add_argument('--embeds')
    ap.add_argument('--tiny', action='store_true')
    ap.add_argument('--out', required=True)
    ap.add_argument('--check')
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--blocks-per-piece', type=int, default=2)
    ap.add_argument('--keep-pieces')
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    os.makedirs(a.out, exist_ok=True)
    tr, vae, prompt, pooled = load_tiny() if a.tiny else load_real(a.sd3, a.lora, a.embeds)
    p = Pieces(tr, vae, prompt, pooled, a.blocks_per_piece)
    refs = references(tr, vae, prompt, pooled, p, a.check) if a.check else []
    if not a.check_only:
        del tr  # the pieces hold every weight they need
        export(p, a.out, a.keep_pieces, release=True)
        del p
    if a.check:
        import gc
        gc.collect()
        worst = check(refs, a.out)
        with open(os.path.join(a.out, 'check.json'), 'w') as f:
            json.dump({'worst_mean_error': worst}, f)
        if worst > 0.02:
            raise SystemExit(f'ncnn output differs from PyTorch (mean error {worst:.3g})')
        print('check passed')


if __name__ == '__main__':
    main()
