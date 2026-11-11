#!/usr/bin/env python3
"""Converts SANA-Sprint (text to image, 2 steps) to ncnn pieces for Brekwol Upscale's Create screen.

    python sana_export.py --repo Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers --out dist \
        --quantizer ncnnllm2int --check
    python sana_export.py --random --out /tmp/sana-test --check      # self-test with tiny random models

What the app runs (engine/create.cpp follows tools/create/sana_ncnn.py, which is the reference):

1. Text. Gemma's tokenizer (gemma-vocab.bin, engine/gemma_tokenizer.hpp) turns the "complex human
   instruction" plus the prompt into ids, with <bos> first. The instruction never changes, so its
   first P tokens are run here once: their keys and values in every layer are baked into the text
   pieces, and the phone runs only the tokens after them (the prompt; usually a few dozen).
   Their embeddings come from gemma-embed.bin (int8 rows with a scale each; x sqrt(hidden) as
   Gemma does). text-NN pieces take x (n x hidden), cos and sin (n x head_dim, positions P..)
   and an additive mask (n x (P + n)); the last piece ends with Gemma's final norm.
   The transformer is conditioned on token 0 (constant: text-h0.bin) and on every token from the
   instruction's last one on, as SanaSprintPipeline.encode_prompt selects them; padding is
   dropped (the pipeline masks it out).
2. Image. SCM sampling, 2 steps (timesteps 1.5708, 1.3, 0), with the guidance embedding at 4.5:
   sana-dit takes the latent (32 x 32 x 32 for 1024 px), the SCM timestep, the guidance value and
   the text (L x 2304) and gives the model output; the arithmetic between the steps is in
   create.json and sana_ncnn.py.
3. Pixels. The DC-AE decoder turns latent / scaling_factor into RGB in [-1, 1]: sana-vae-head
   (its attention stages, whole, up to 128 px) and sana-vae-tail (convolutions only) in tiles of
   vae_tile head pixels with vae_margin around each, so 1024 px fits in a phone's memory.

Outputs, into --out: gemma-vocab.bin, gemma-embed.bin, text-h0.bin, text-NN.param/.bin,
sana-dit.param/.bin, sana-vae-head.param/.bin, sana-vae-tail.param/.bin, create.json.

Needs torch, diffusers, transformers, sentencepiece, numpy, pnnx (and ncnn for --check).
"""
import argparse
import copy
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MASK_NEG = -1.0e4  # masked attention logits (after Gemma's softcap, which keeps them within +-50)


# ---------------------------------------------------------------- Gemma-2 text encoder, prefix-cached

def rms(x, w, eps):
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * (1.0 + w)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class GemmaLayers(nn.Module):
    """Gemma-2 decoder layers [first, last) run on n new tokens after P cached ones.

    forward(x [1, n, H], cos [1, n, D], sin [1, n, D], mask [1, n, P + n]) -> x [1, n, H]
    (plus the final norm when [final]). Keys and values of the P cached tokens are constants.
    """

    def __init__(self, model, first, last, prefix_kv, final):
        super().__init__()
        cfg = model.config
        # fp32 copies of just these layers: the model itself stays in bf16 (half the memory).
        self.layers = nn.ModuleList(copy.deepcopy(l).float() for l in model.layers[first:last])
        self.heads = cfg.num_attention_heads
        self.kv_heads = cfg.num_key_value_heads
        self.dim = cfg.head_dim
        self.scale = cfg.query_pre_attn_scalar ** -0.5
        self.softcap = cfg.attn_logit_softcapping
        self.eps = cfg.rms_norm_eps
        self.final = final
        self.norm_w = nn.Parameter(model.norm.weight.detach().float().clone()) if final else None
        for i, (k, v) in enumerate(prefix_kv[first:last]):
            self.register_buffer(f'pk{i}', k.clone())  # [1, kv_heads, P, D]
            self.register_buffer(f'pv{i}', v.clone())

    def forward(self, x, cos, sin, mask):
        n = x.shape[1]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        rep = self.heads // self.kv_heads
        for i, layer in enumerate(self.layers):
            a = layer.self_attn
            h = rms(x, layer.input_layernorm.weight, self.eps)
            q = a.q_proj(h).view(1, n, self.heads, self.dim).transpose(1, 2)
            k = a.k_proj(h).view(1, n, self.kv_heads, self.dim).transpose(1, 2)
            v = a.v_proj(h).view(1, n, self.kv_heads, self.dim).transpose(1, 2)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            k = torch.cat((getattr(self, f'pk{i}'), k), dim=2)
            v = torch.cat((getattr(self, f'pv{i}'), v), dim=2)
            # Grouped-query attention without repeat_interleave (not in ncnn): query head h uses
            # key/value head h // rep, so group the query heads by their key/value head.
            q = q.reshape(self.kv_heads, rep * n, self.dim)
            s = torch.matmul(q, k.squeeze(0).transpose(1, 2)) * self.scale  # [kv, rep*n, P+n]
            if self.softcap:
                s = torch.tanh(s / self.softcap) * self.softcap
            s = s.reshape(self.kv_heads, rep, n, -1) + mask
            p = torch.softmax(s, dim=-1).reshape(self.kv_heads, rep * n, -1)
            o = torch.matmul(p, v.squeeze(0))  # [kv, rep*n, D]
            o = o.reshape(self.heads, n, self.dim).transpose(0, 1).reshape(1, n, self.heads * self.dim)
            o = a.o_proj(o)
            x = x + rms(o, layer.post_attention_layernorm.weight, self.eps)
            h = rms(x, layer.pre_feedforward_layernorm.weight, self.eps)
            m = layer.mlp
            h = m.down_proj(nn.functional.gelu(m.gate_proj(h), approximate='tanh') * m.up_proj(h))
            x = x + rms(h, layer.post_feedforward_layernorm.weight, self.eps)
        if self.final:
            x = rms(x, self.norm_w, self.eps)
        return x


def rope(cfg, positions):
    """cos, sin [len(positions), head_dim] (Gemma's default RoPE)."""
    d = cfg.head_dim
    theta = getattr(cfg, 'rope_theta', None) or cfg.rope_parameters['rope_theta']
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    f = torch.outer(torch.tensor(positions, dtype=torch.float32), inv)
    emb = torch.cat((f, f), dim=-1)
    return emb.cos(), emb.sin()


def causal_mask(n, start):
    """Additive mask [n, start + n]: token i (at position start + i) sees positions <= its own."""
    m = torch.full((n, start + n), MASK_NEG)
    for i in range(n):
        m[i, :start + i + 1] = 0
    return m


@torch.no_grad()
def run_layers(model, x, start, prefix_kv=None, collect=False):
    """Runs every layer on x [1, n, H] at positions start..; returns (out, per-layer (k, v) if collect)."""
    cfg = model.config
    n = x.shape[1]
    cos, sin = rope(cfg, range(start, start + n))
    mask = causal_mask(n, start).unsqueeze(0)
    kv = []
    out = x
    if prefix_kv is None:
        prefix_kv = [(torch.zeros(1, cfg.num_key_value_heads, 0, cfg.head_dim),) * 2] * len(model.layers)
    for li in range(len(model.layers)):
        piece = GemmaLayers(model, li, li + 1, prefix_kv, final=False)
        if collect:
            # Keys and values after RoPE, recomputed the same way the piece does.
            layer = piece.layers[0]
            a = layer.self_attn
            h = rms(out, layer.input_layernorm.weight, cfg.rms_norm_eps)
            k = a.k_proj(h).view(1, n, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
            v = a.v_proj(h).view(1, n, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
            k = k * cos.unsqueeze(0) + rotate_half(k) * sin.unsqueeze(0)
            kv.append((torch.cat((prefix_kv[li][0], k), 2), torch.cat((prefix_kv[li][1], v), 2)))
        out = piece(out, cos.unsqueeze(0), sin.unsqueeze(0), mask)
        del piece
    out = rms(out, model.norm.weight.float(), cfg.rms_norm_eps)
    return out, kv


# ---------------------------------------------------------------- export helpers

def _movedim_as_permute(self, source, destination):
    n = self.dim()
    dims = list(range(n))
    d = dims.pop(source % n)
    dims.insert(destination % n, d)
    return self.permute(*dims)


def _repeat_as_concat(self, repeats, dim=None, output_size=None):
    assert isinstance(repeats, int) and dim is not None
    shape = list(self.shape)
    d = dim % len(shape)
    # Each slice along [d] repeated in place, through 4-D tensors only: [1, C, H, W] (dim 1) goes
    # [1, C, 1, H*W], [1, C, r, H*W], [1, C*r, H, W]. A 5-D stack or unsqueeze came out of pnnx
    # as a reshape that ncnn either lacks or runs wrong at an untraced size.
    assert len(shape) == 4 and d == 1, 'only the channel repeat of DC-AE'
    y = self.reshape(shape[0], shape[1], 1, shape[2] * shape[3])
    y = torch.cat([y] * repeats, dim=2)
    return y.reshape(shape[0], shape[1] * repeats, shape[2], shape[3])


def pnnx_export(module, inputs, inputs2, stem, work, fp16=True):
    """Traces [module] and converts it with pnnx; [inputs2] gives the dynamic second shapes."""
    os.makedirs(work, exist_ok=True)
    name = os.path.basename(stem).replace('-', '_')  # pnnx writes '-' as '_'
    pt = os.path.join(work, name + '.pt')
    module = module.eval()
    # ncnn has neither movedim (DC-AE's RMSNorm2d) nor repeat_interleave (DC-AE's shortcuts):
    # trace them as the permutes and concatenations they amount to.
    movedim, repeat = torch.Tensor.movedim, torch.Tensor.repeat_interleave
    torch.Tensor.movedim = _movedim_as_permute
    torch.Tensor.repeat_interleave = _repeat_as_concat
    try:
        with torch.no_grad():
            ts = torch.jit.trace(module, inputs, check_trace=False)
    finally:
        torch.Tensor.movedim, torch.Tensor.repeat_interleave = movedim, repeat
    ts.save(pt)
    shape = ','.join('[' + ','.join(str(d) for d in t.shape) + ']' for t in inputs)
    args = ['pnnx', pt, f'inputshape={shape}', f'fp16={1 if fp16 else 0}']
    if inputs2 is not None:
        args.append('inputshape2=' + ','.join('[' + ','.join(str(d) for d in t.shape) + ']' for t in inputs2))
    r = subprocess.run(args, cwd=work, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(os.path.join(work, name + '.ncnn.param')):
        print(r.stdout[-4000:], r.stderr[-4000:])
        raise SystemExit(f'pnnx failed on {name}')
    shutil.move(os.path.join(work, name + '.ncnn.param'), stem + '.param')
    shutil.move(os.path.join(work, name + '.ncnn.bin'), stem + '.bin')
    for f in os.listdir(work):
        if f.startswith(name):
            p = os.path.join(work, f)
            (shutil.rmtree if os.path.isdir(p) else os.remove)(p)


class VaeHead(nn.Module):
    """The decoder up to its first local-only stage: conv_in and the attention stages."""

    def __init__(self, dec, k):
        super().__init__()
        self.dec, self.k = dec, k

    def forward(self, z):
        d = self.dec
        h = d.conv_in(z)
        if d.in_shortcut:
            h = h + z.repeat_interleave(d.in_shortcut_repeats, dim=1)
        for i in reversed(range(self.k, len(d.up_blocks))):
            h = d.up_blocks[i](h)
        return h


class VaeTail(nn.Module):
    """The rest: convolution stages, pixel shuffles and per-pixel norms only, so it can run in tiles."""

    def __init__(self, dec, k):
        super().__init__()
        self.dec, self.k = dec, k

    def forward(self, h):
        d = self.dec
        for i in reversed(range(self.k)):
            h = d.up_blocks[i](h)
        h = d.norm_out(h.movedim(1, -1)).movedim(-1, 1)
        return d.conv_out(d.conv_act(h))


def vae_split(dec):
    """How many of the decoder's top stages are local (ResBlocks and upsampling convolutions):
    those form the tail. Returns (k, head scale, tail scale, tail channels, tail margin): the margin
    is the tail's reach in head pixels, rounded up, plus one."""
    from diffusers.models.autoencoders.autoencoder_dc import DCUpBlock2d, ResBlock
    n = len(dec.up_blocks)
    k = 0
    while k < n and all(isinstance(m, (DCUpBlock2d, ResBlock)) for m in dec.up_blocks[k]):
        k += 1
    assert k > 0, 'the decoder has no local stages to tile'

    def ups(i):
        return sum(1 for m in dec.up_blocks[i] if isinstance(m, DCUpBlock2d))

    head_scale = 2 ** sum(ups(i) for i in range(k, n))
    tail_scale = 2 ** sum(ups(i) for i in range(k))
    reach, scale = 0.0, 1

    def conv(c):
        assert c.stride == (1, 1) and c.kernel_size[0] == c.kernel_size[1]
        return (c.kernel_size[0] // 2) * c.dilation[0]

    for i in reversed(range(k)):
        for m in dec.up_blocks[i]:
            if isinstance(m, DCUpBlock2d):
                if m.interpolate:
                    scale *= m.factor
                    reach += conv(m.conv) / scale
                else:
                    reach += conv(m.conv) / scale
                    scale *= m.factor
            else:
                reach += (conv(m.conv1) + conv(m.conv2)) / scale
    out = dec.conv_out
    if isinstance(out, DCUpBlock2d):
        reach += conv(out.conv) / scale
        scale *= out.factor
        tail_scale *= out.factor
    else:
        reach += conv(out) / scale
    channels = dec.up_blocks[k - 1][0].conv.in_channels if isinstance(dec.up_blocks[k - 1][0], DCUpBlock2d) \
        else dec.up_blocks[k - 1][0].conv1.in_channels
    return k, head_scale, tail_scale, channels, math.ceil(reach) + 1


def run_ncnn(stem, inputs):
    import ncnn
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = int(os.environ.get('OMP_NUM_THREADS', '4'))
    net.load_param(stem + '.param')
    net.load_model(stem + '.bin')
    ex = net.create_extractor()
    keep = [np.ascontiguousarray(a, dtype=np.float32) for a in inputs]  # ncnn.Mat shares this memory
    for name, a in zip(net.input_names(), keep):
        ex.input(name, ncnn.Mat(a))
    r, out = ex.extract(sorted(net.output_names())[0])
    assert r == 0
    return np.array(out)


def export_vae(vae, side, out, work):
    """Exports the DC-AE decoder as sana-vae-head and sana-vae-tail and checks them, the tail run in
    tiles as the app runs it, against PyTorch's whole decoder at the full [side].

    At 1024 px the whole decoder needs over 13 GB in ncnn (and pnnx more than 29 GB to convert
    it). The attention stages work on small images (128 px and less) and stay whole in the head;
    the tail is convolutions, pixel shuffles and per-pixel norms only, so tiles with a margin of
    its reach give exactly the whole image. Both are traced at small sizes (two sizes make the
    shapes dynamic) and run here at sizes they were not traced at."""
    import sana_ncnn
    dec = vae.decoder
    lc = vae.config.latent_channels
    k, head_scale, tail_scale, channels, margin = vae_split(dec)
    hs = side * head_scale
    tile = max(hs // 4, 1)
    h_small = max(side // 4, 4)
    h_mid = max(side // 2, h_small + 2)
    t_small = max(hs // 8, 4)
    t_mid = t_small + max(t_small // 2, 2)
    head, tail = VaeHead(dec, k).eval(), VaeTail(dec, k).eval()
    pnnx_export(head, (torch.randn(1, lc, h_small, h_small),), (torch.randn(1, lc, h_mid, h_mid),),
                os.path.join(out, 'sana-vae-head'), work)
    pnnx_export(tail, (torch.randn(1, channels, t_small, t_small),), (torch.randn(1, channels, t_mid, t_mid),),
                os.path.join(out, 'sana-vae-tail'), work)

    torch.manual_seed(1)
    z = torch.randn(1, lc, side, side)
    want = vae.decode(z, return_dict=False)[0][0].numpy()
    h = run_ncnn(os.path.join(out, 'sana-vae-head'), [z[0].numpy()])
    assert h.shape == (channels, hs, hs), h.shape
    got = sana_ncnn.tiled(lambda x: run_ncnn(os.path.join(out, 'sana-vae-tail'), [x]), h, tile, margin, tail_scale)
    assert got.shape == want.shape, (got.shape, want.shape)
    mse = float(np.mean((np.clip(got, -1, 1) - np.clip(want, -1, 1)) ** 2))
    psnr = 10 * np.log10(4 / max(mse, 1e-12))  # pixels in [-1, 1]: peak-to-peak 2
    print(f'sana-vae at {side}x{side}: head traced at {h_small} and {h_mid}, tail at {t_small} and {t_mid}, '
          f'run in tiles of {tile} with margin {margin}: PSNR {psnr:.1f} dB against PyTorch')
    if psnr < 30:
        raise SystemExit('sana-vae does not match PyTorch at the full size')
    return {'vae_head_scale': head_scale, 'vae_tail_scale': tail_scale, 'vae_tail_channels': channels,
            'vae_tile': tile, 'vae_margin': margin}


def quantize(stem, tool, bits, block):
    """Block-quantizes the Gemm / Linear weights of [stem] in place (ncnnllm2int)."""
    q = stem + '.q'
    r = subprocess.run([tool, stem + '.param', stem + '.bin', q + '.param', q + '.bin', 'method=minmax',
                        f'bits={bits}', f'block={block}'], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(q + '.bin'):
        print((r.stdout + r.stderr)[-3000:])
        raise SystemExit(f'quantizer failed on {stem}')
    os.replace(q + '.param', stem + '.param')
    os.replace(q + '.bin', stem + '.bin')


def write_embed(weight, path):
    """int8 rows with one fp32 scale each: "GEMB", u32 1, u32 rows, u32 cols, f32 scales, i8 data.
    Works in chunks of rows (the fp32 table is 2.4 GB). Returns (scales, int8 rows)."""
    rows, cols = weight.shape
    scale = np.empty(rows, np.float32)
    q = np.empty((rows, cols), np.int8)
    for i in range(0, rows, 8192):
        w = weight[i:i + 8192].detach().float().numpy()
        sc = np.abs(w).max(axis=1) / 127.0
        sc[sc == 0] = 1.0
        scale[i:i + 8192] = sc
        q[i:i + 8192] = np.clip(np.round(w / sc[:, None]), -127, 127).astype(np.int8)
    with open(path, 'wb') as f:
        f.write(b'GEMB' + struct.pack('<III', 1, rows, cols))
        f.write(scale.astype('<f4').tobytes())
        f.write(q.tobytes())
    return scale, q


# ---------------------------------------------------------------- models

class Parts:
    """The pipeline's parts, loaded one at a time (in fp32 the whole pipeline is ~13 GB, which
    stopped the 16 GB GitHub runner): drop() each one once it is exported."""

    def __init__(self, args):
        self.repo = args.repo
        self.pipe = random_models(args.vocab_model) if args.random else None

    def get(self, name):
        if self.pipe is not None:
            return getattr(self.pipe, name)
        from diffusers import AutoencoderDC, SanaTransformer2DModel, SCMScheduler
        from transformers import AutoTokenizer, Gemma2Model
        kind = {'tokenizer': AutoTokenizer, 'text_encoder': Gemma2Model, 'transformer': SanaTransformer2DModel,
                'vae': AutoencoderDC, 'scheduler': SCMScheduler}[name]
        if name == 'tokenizer':
            return kind.from_pretrained(self.repo, subfolder=name)
        if name == 'scheduler':
            return kind.from_pretrained(self.repo, subfolder=name)
        # Gemma stays in bf16 (5 GB, not 10): its pieces are copied to fp32 two layers at a time.
        dtype = torch.bfloat16 if name == 'text_encoder' else torch.float32
        return kind.from_pretrained(self.repo, subfolder=name, dtype=dtype).eval()


def drop(*objs):
    import gc
    del objs
    gc.collect()


def random_models(vocab_model):
    """Tiny random SANA-Sprint with Gemma's real tokenizer, for --random."""
    from diffusers import AutoencoderDC, SanaSprintPipeline, SanaTransformer2DModel, SCMScheduler
    from transformers import Gemma2Config, Gemma2Model, GemmaTokenizer
    torch.manual_seed(0)
    folder = tempfile.mkdtemp()
    shutil.copy(vocab_model, os.path.join(folder, 'tokenizer.model'))
    tok = GemmaTokenizer.from_pretrained(folder, add_bos_token=True)
    cfg = Gemma2Config(vocab_size=256000, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=16, query_pre_attn_scalar=16,
                       sliding_window=4096, attn_logit_softcapping=50.0)
    te = Gemma2Model(cfg).eval()
    for p in te.parameters():
        p.data.normal_(0, 0.05)
    tr = SanaTransformer2DModel(in_channels=8, out_channels=8, num_attention_heads=2, attention_head_dim=16,
                                num_layers=2, num_cross_attention_heads=2, cross_attention_head_dim=16,
                                cross_attention_dim=32, caption_channels=64, sample_size=8,
                                guidance_embeds=True, qk_norm='rms_norm_across_heads', timestep_scale=1.0).eval()
    vae = AutoencoderDC(latent_channels=8, attention_head_dim=8,
                        encoder_block_types=('ResBlock', 'ResBlock', 'EfficientViTBlock'),
                        decoder_block_types=('ResBlock', 'ResBlock', 'EfficientViTBlock'),
                        encoder_block_out_channels=(8, 16, 16), decoder_block_out_channels=(8, 16, 16),
                        encoder_layers_per_block=(1, 1, 1), decoder_layers_per_block=(1, 1, 1),
                        encoder_qkv_multiscales=((), (), (5,)), decoder_qkv_multiscales=((), (), (5,)),
                        scaling_factor=0.41).eval()
    sch = SCMScheduler()
    return SanaSprintPipeline(tokenizer=tok, text_encoder=te, transformer=tr, vae=vae, scheduler=sch)


def instruction():
    import inspect
    from diffusers import SanaSprintPipeline
    return inspect.signature(SanaSprintPipeline.__call__).parameters['complex_human_instruction'].default


# ---------------------------------------------------------------- main

def write_meta(out, part, values):
    with open(os.path.join(out, f'meta-{part}.json'), 'w') as f:
        json.dump(values, f, indent=1)


def merge_meta(out):
    """create.json from meta-text.json and meta-image.json, once both exist."""
    parts = [os.path.join(out, f'meta-{p}.json') for p in ('text', 'image')]
    if not all(os.path.exists(p) for p in parts):
        return
    meta = {'version': 2}  # 2: the decoder as head + tiled tail
    for p in parts:
        meta.update(json.load(open(p)))
        os.remove(p)
    with open(os.path.join(out, 'create.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print('wrote create.json')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo')
    ap.add_argument('--random', action='store_true')
    ap.add_argument('--vocab-model', help="Gemma's tokenizer.model (SentencePiece)", required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--layers-per-piece', type=int, default=2)
    ap.add_argument('--text-tokens', type=int, default=128, help='tokens per text run (after the cached prefix)')
    ap.add_argument('--quantizer', help='ncnnllm2int; without it the pieces keep fp16 weights')
    ap.add_argument('--bits', type=int, default=8)
    ap.add_argument('--block', type=int, default=64)
    ap.add_argument('--quantize-dit', action='store_true')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--stage', choices=['text', 'image', 'all'], default='all')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    work = tempfile.mkdtemp()
    torch.set_grad_enabled(False)

    parts = Parts(a)
    # --stage text / image: one part per run (the workflow runs them as separate jobs, each on a
    # fresh runner); both leave meta-*.json, merged into create.json once both are there.
    if a.stage in ('text', 'all'):
        tok = parts.get('tokenizer')
        te = parts.get('text_encoder')
        cfg = te.config
        chi = '\n'.join(instruction())

        # Tokenizer: the app's vocabulary, and the instruction ids HF's tokenizer gives.
        import gemma_vocab
        sys.argv = ['gemma_vocab.py', a.vocab_model, os.path.join(a.out, 'gemma-vocab.bin')]
        gemma_vocab.main()
        chi_ids = tok(chi, add_special_tokens=True).input_ids
        prefix = len(chi_ids) - 2  # the last instruction tokens may join the prompt's first word
        print(f'instruction: {len(chi_ids)} tokens, {prefix} cached')

        # Embedding rows (int8) and the cached prefix.
        emb_scale, emb_q = write_embed(te.embed_tokens.weight, os.path.join(a.out, 'gemma-embed.bin'))
        if not a.random:
            te.embed_tokens = None  # 1.2 GB that nothing needs any more (the tiny check still uses it)
        normalizer = float(cfg.hidden_size) ** 0.5

        def embed(ids):
            return torch.from_numpy(emb_q[ids].astype(np.float32) * emb_scale[ids, None] * np.float32(normalizer)).unsqueeze(0)

        x0 = embed(chi_ids[:prefix])
        h_prefix, kv = run_layers(te, x0, 0, collect=True)
        np.asarray(h_prefix[0, 0], dtype='<f4').tofile(os.path.join(a.out, 'text-h0.bin'))

        # Text pieces.
        layers = len(te.layers)
        pieces = []
        n = a.text_tokens  # fixed: ncnn's dynamic shapes broke at untraced lengths; the app pads
        for first in range(0, layers, a.layers_per_piece):
            last = min(layers, first + a.layers_per_piece)
            m = GemmaLayers(te, first, last, kv, final=last == layers)
            stem = os.path.join(a.out, f'text-{len(pieces):02d}')

            def sample(n):
                cos, sin = rope(cfg, range(prefix, prefix + n))
                return (torch.zeros(1, n, cfg.hidden_size), cos.unsqueeze(0), sin.unsqueeze(0),
                        causal_mask(n, prefix).unsqueeze(0))
            # --random: fp32 weights, so the check tests the graph (tiny random weights amplify fp16
            # rounding far beyond what a trained model does); the real run checks fp16 / quantized.
            pnnx_export(m, sample(n), None, stem, work, fp16=not a.random)
            if a.quantizer:
                quantize(stem, a.quantizer, a.bits, a.block)
            pieces.append(os.path.basename(stem))
            print('exported', pieces[-1], f'layers {first}..{last - 1}')

        m = None  # the last piece holds layers of te
        del te, kv, h_prefix, emb_q
        drop()
        write_meta(a.out, 'text', {
            'instruction': chi,
            'prefix_ids': chi_ids[:prefix],
            'instruction_tokens': len(chi_ids),
            'text_tokens': a.text_tokens,
            'max_prompt_tokens': a.text_tokens - (len(chi_ids) - prefix),
            'hidden': cfg.hidden_size,
            'head_dim': cfg.head_dim,
            'rope_theta': getattr(cfg, 'rope_theta', None) or cfg.rope_parameters['rope_theta'],
            'normalizer': normalizer,
            'mask_neg': MASK_NEG,
            'text_pieces': pieces,
        })

    if a.stage in ('image', 'all'):
        tr = parts.get('transformer')

        # Transformer: (latent, scm timestep, guidance, text) -> model output.
        class Dit(nn.Module):
            # The model's own unpatchify goes through a 6-D reshape, which ncnn has not; with patch
            # size 1 it is a transpose, done here on proj_out's output (the rest is traced away).
            def __init__(self, t):
                super().__init__()
                assert t.config.patch_size == 1
                self.t = t
                self.last = None
                t.proj_out.register_forward_hook(lambda mod, inp, out: setattr(self, 'last', out))

            def forward(self, x, t, g, text):
                self.t(x, encoder_hidden_states=text, timestep=t, guidance=g, return_dict=False)
                return self.last.transpose(1, 2).reshape(1, -1, x.shape[2], x.shape[3])  # [1, H*W, C] -> [1, C, H, W]

        lc = tr.config.in_channels
        side = tr.config.sample_size
        dit_in = (torch.randn(1, lc, side, side), torch.tensor([0.5]), torch.tensor([4.5]),
                  torch.randn(1, 20, tr.config.caption_channels))
        dit_in2 = (torch.randn(1, lc, side, side), torch.tensor([0.5]), torch.tensor([4.5]),
                   torch.randn(1, 40, tr.config.caption_channels))
        pnnx_export(Dit(tr), dit_in, dit_in2, os.path.join(a.out, 'sana-dit'), work)
        if a.quantizer and a.quantize_dit:
            quantize(os.path.join(a.out, 'sana-dit'), a.quantizer, a.bits, a.block)
        print('exported sana-dit')

        guidance = 4.5 * tr.config.guidance_embeds_scale
        del tr
        drop()
        vae = parts.get('vae')
        tiles = export_vae(vae, side, a.out, work)
        print('exported sana-vae-head and sana-vae-tail')

        vae_scale = 2 ** (len(vae.config.decoder_block_out_channels) - 1)
        scaling = vae.config.scaling_factor
        del vae
        drop()
        sch = parts.get('scheduler').config
        write_meta(a.out, 'image', {
            'latent_channels': lc,
            'latent_side': side,
            'vae_scale': vae_scale,
            'scaling_factor': scaling,
            'sigma_data': sch.sigma_data,
            'timesteps': [1.57080, 1.3, 0.0],
            'guidance': guidance,
            **tiles,
        })

    merge_meta(a.out)

    if a.check and os.path.exists(os.path.join(a.out, 'create.json')):
        # The real model's check runs in its own process (sana_ncnn.py --check), with nothing
        # else in memory; the tiny random one right here.
        import sana_ncnn
        pipe = parts.pipe
        if pipe is None:
            from diffusers import SanaSprintPipeline
            pipe = SanaSprintPipeline.from_pretrained(a.repo, dtype=torch.float32)
        sana_ncnn.check(pipe, a.out, a.vocab_model)
    shutil.rmtree(work)
    for f in sorted(os.listdir(a.out)):
        print(f, os.path.getsize(os.path.join(a.out, f)))


if __name__ == '__main__':
    main()
