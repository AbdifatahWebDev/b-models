#!/usr/bin/env python3
"""FLUX.2 klein 4B -> ncnn pieces for Brekwol Upscale's edit model (see PLAN.md).

    python klein_export.py --stage text --repo black-forest-labs/FLUX.2-klein-4B --out DIST \
        --quantizer ncnnllm2int [--bits 8 --block 64]
    python klein_export.py --random --stage text --out DIST --check      # tiny self-test

Stage text (this file so far): the Qwen3 text encoder as the pipeline uses it
(Flux2KleinPipeline._get_qwen3_prompt_embeds): the chat-templated prompt padded to TOKENS tokens,
the hidden states after layers 9, 18 and 27 joined per token. Layers 0-26 go out in pieces of
three (qwen-00 .. qwen-08), each (x, cos, sin, mask) -> x, so every third piece's output is one
of the three wanted states; weights block-quantized by ncnnllm2int, activations fp32. The
embedding rows are int8 with a scale per row (qwen-embed.bin, tools/create's format).
meta-text.json has what the phone needs; klein_ncnn.py runs the pieces as the phone does.
"""
import argparse
import copy
import json
import os
import sys
import tempfile

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'create'))
from sana_export import MASK_NEG, pnnx_export, quantize, rope, rotate_half, write_embed  # noqa: E402

LAYERS_OUT = (9, 18, 27)


def rms(x, w, eps):
    """Qwen3RMSNorm (plain weight; Gemma's, in sana_export.py, scales by 1 + weight)."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


class QwenLayers(nn.Module):
    """Qwen3 decoder layers [first, last) on n tokens.

    forward(x [1, n, H], cos [1, n, D], sin [1, n, D], mask [1, n, n]) -> x [1, n, H]
    """

    def __init__(self, model, first, last):
        super().__init__()
        cfg = model.config
        self.layers = nn.ModuleList(copy.deepcopy(l).float() for l in model.layers[first:last])
        self.heads = cfg.num_attention_heads
        self.kv_heads = cfg.num_key_value_heads
        self.dim = cfg.head_dim
        self.scale = cfg.head_dim ** -0.5
        self.eps = cfg.rms_norm_eps

    def forward(self, x, cos, sin, mask):
        n = x.shape[1]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        rep = self.heads // self.kv_heads
        for layer in self.layers:
            a = layer.self_attn
            h = rms(x, layer.input_layernorm.weight, self.eps)
            # Qwen3 normalises queries and keys per head before RoPE.
            q = rms(a.q_proj(h).view(1, n, self.heads, self.dim), a.q_norm.weight, self.eps).transpose(1, 2)
            k = rms(a.k_proj(h).view(1, n, self.kv_heads, self.dim), a.k_norm.weight, self.eps).transpose(1, 2)
            v = a.v_proj(h).view(1, n, self.kv_heads, self.dim).transpose(1, 2)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            # Grouped-query attention without repeat_interleave (see tools/create/sana_export.py).
            q = q.reshape(self.kv_heads, rep * n, self.dim)
            s = torch.matmul(q, k.squeeze(0).transpose(1, 2)) * self.scale  # [kv, rep*n, n]
            s = s.reshape(self.kv_heads, rep, n, n) + mask
            p = torch.softmax(s, dim=-1).reshape(self.kv_heads, rep * n, n)
            o = torch.matmul(p, v.squeeze(0))
            o = o.reshape(self.heads, n, self.dim).transpose(0, 1).reshape(1, n, self.heads * self.dim)
            x = x + a.o_proj(o)
            h = rms(x, layer.post_attention_layernorm.weight, self.eps)
            m = layer.mlp
            x = x + m.down_proj(nn.functional.silu(m.gate_proj(h)) * m.up_proj(h))
        return x


def prompt_mask(n, real):
    """Additive mask [n, n] as HF builds it for a right-padded prompt of [real] tokens: token i sees
    the real tokens up to its own position (a padding token sees every real token)."""
    m = torch.full((n, n), MASK_NEG)
    for i in range(n):
        m[i, :min(i + 1, real)] = 0
    return m


def chat(tokenizer, prompt):
    return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def reference(model, tokenizer, prompt, tokens):
    """The pipeline's prompt_embeds [tokens, 3 * H] and its input ids and count of real tokens."""
    t = tokenizer(chat(tokenizer, prompt), return_tensors='pt', padding='max_length', truncation=True, max_length=tokens)
    with torch.no_grad():
        out = model(input_ids=t.input_ids, attention_mask=t.attention_mask, output_hidden_states=True, use_cache=False)
    hs = torch.stack([out.hidden_states[k].float() for k in LAYERS_OUT], dim=1)  # [1, 3, n, H]
    return hs.permute(0, 2, 1, 3).reshape(tokens, -1).numpy(), t.input_ids[0].tolist(), int(t.attention_mask.sum())


# ---------------------------------------------------------------- transformer (Flux2Transformer2DModel)

def rotation(d):
    """R with x @ R = (-x1, x0, -x3, x2, ...): the pair rotation of apply_rotary_emb as one matmul
    (its unbind and stack have no ncnn counterpart)."""
    r = torch.zeros(d, d)
    for i in range(0, d, 2):
        r[i + 1, i] = -1.0
        r[i, i + 1] = 1.0
    return r


def tnorm(x, norm):
    """torch.nn.RMSNorm over the last dimension."""
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + (norm.eps or 1e-6))
    return y if norm.weight is None else y * norm.weight


class Attend(nn.Module):
    """Attention as Flux2AttnProcessor does it, written for ncnn: q, k, v [1, N, heads·d] (q and
    k already normed per head by the caller) -> [1, N, heads·d]."""

    def __init__(self, heads, dim):
        super().__init__()
        self.heads, self.dim = heads, dim
        self.register_buffer('rot', rotation(dim))

    def split(self, x):
        return x.reshape(1, -1, self.heads, self.dim)

    def forward(self, q, k, v, cos, sin):
        # q, k, v: [1, N, heads, d]; cos, sin: [1, N, d]
        cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
        q = q * cos + torch.matmul(q, self.rot) * sin
        k = k * cos + torch.matmul(k, self.rot) * sin
        q, k, v = (t.squeeze(0).transpose(0, 1) for t in (q, k, v))  # [heads, N, d]
        p = torch.softmax(torch.matmul(q, k.transpose(1, 2)) * (self.dim ** -0.5), dim=-1)
        o = torch.matmul(p, v)  # [heads, N, d]
        return o.transpose(0, 1).reshape(1, -1, self.heads * self.dim)


def mods(m, sets, d):
    """Flux2Modulation.split: [1, 3·sets·d] -> sets × (shift, scale, gate), each [1, 1, d]."""
    m = m.unsqueeze(1)
    return [tuple(m[:, :, (3 * i + j) * d:(3 * i + j + 1) * d] for j in range(3)) for i in range(sets)]


class KleinTime(nn.Module):
    """t [1] (the scheduler's timestep / 1000) -> [1, 16·D]: temb, the double blocks' image and
    text modulations (6·D each) and the single blocks' (3·D)."""

    def __init__(self, tr):
        super().__init__()
        self.embed = tr.time_guidance_embed
        self.img, self.txt, self.single = (tr.double_stream_modulation_img, tr.double_stream_modulation_txt,
                                           tr.single_stream_modulation)

    def forward(self, t):
        temb = self.embed(t * 1000, None)
        return torch.cat((temb, self.img(temb), self.txt(temb), self.single(temb)), dim=-1)


class KleinDouble(nn.Module):
    """Double-stream block [index] on the joined stream x = [text (nt) | image].

    forward(x [1, N, D], mod_img [1, 6D], mod_txt [1, 6D], cos, sin [1, N, d]) -> x. The first
    block takes (text [1, nt, J], image [1, Ni, C], ...) and embeds them."""

    def __init__(self, tr, index, nt):
        super().__init__()
        self.b = tr.transformer_blocks[index]
        self.first = index == 0
        self.xe, self.ce = (tr.x_embedder, tr.context_embedder) if self.first else (None, None)
        self.nt, self.d = nt, tr.inner_dim
        a = self.b.attn
        self.att = Attend(a.heads, a.head_dim)

    def run(self, e, h, mi, mt, cos, sin):
        b, a, at = self.b, self.b.attn, self.att
        (s1, c1, g1), (s2, c2, g2) = mods(mi, 2, self.d)
        (ts1, tc1, tg1), (ts2, tc2, tg2) = mods(mt, 2, self.d)
        nh = (1 + c1) * b.norm1(h) + s1
        ne = (1 + tc1) * b.norm1_context(e) + ts1
        q = torch.cat((tnorm(at.split(a.add_q_proj(ne)), a.norm_added_q), tnorm(at.split(a.to_q(nh)), a.norm_q)), dim=1)
        k = torch.cat((tnorm(at.split(a.add_k_proj(ne)), a.norm_added_k), tnorm(at.split(a.to_k(nh)), a.norm_k)), dim=1)
        v = torch.cat((at.split(a.add_v_proj(ne)), at.split(a.to_v(nh))), dim=1)
        o = at(q, k, v, cos, sin)
        oe, oh = torch.tensor_split(o, (self.nt,), dim=1)  # pnnx lost a plain o[:, nt:] slice
        h = h + g1 * a.to_out[0](oh)
        h = h + g2 * b.ff(b.norm2(h) * (1 + c2) + s2)
        e = e + tg1 * a.to_add_out(oe)
        e = e + tg2 * b.ff_context(b.norm2_context(e) * (1 + tc2) + ts2)
        return torch.cat((e, h), dim=1)

    def forward(self, *args):
        if self.first:
            text, img, mi, mt, cos, sin = args
            return self.run(self.ce(text), self.xe(img), mi, mt, cos, sin)
        x, mi, mt, cos, sin = args
        e, h = torch.tensor_split(x, (self.nt,), dim=1)
        return self.run(e, h, mi, mt, cos, sin)


class KleinSingle(nn.Module):
    """Single-stream blocks [first, last) on the joined stream; with [final] also the output
    layers: forward(x, mod [1, 3D], cos, sin[, temb [1, D]]) -> x, or [1, N - nt, C] when final."""

    def __init__(self, tr, first, last, nt, final):
        super().__init__()
        self.blocks = nn.ModuleList(tr.single_transformer_blocks[first:last])
        self.nt, self.d, self.final = nt, tr.inner_dim, final
        self.norm_out, self.proj_out = (tr.norm_out, tr.proj_out) if final else (None, None)
        a = self.blocks[0].attn
        self.att = Attend(a.heads, a.head_dim)

    def forward(self, x, mod, cos, sin, temb=None):
        ((shift, scale, gate),) = mods(mod, 1, self.d)
        at = self.att
        for b in self.blocks:
            a = b.attn
            y = a.to_qkv_mlp_proj((1 + scale) * b.norm(x) + shift)
            d = self.d
            q, k, v, m = y[:, :, :d], y[:, :, d:2 * d], y[:, :, 2 * d:3 * d], y[:, :, 3 * d:]
            o = at(tnorm(at.split(q), a.norm_q), tnorm(at.split(k), a.norm_k), at.split(v), cos, sin)
            half = a.mlp_hidden_dim
            m = nn.functional.silu(m[:, :, :half]) * m[:, :, half:]
            x = x + gate * a.to_out(torch.cat((o, m), dim=-1))
        if self.final:
            e, h = torch.tensor_split(x, (self.nt,), dim=1)
            # The text half is not needed any more, but an unused split output would become a
            # second output of the piece: tie it in with a zero.
            x = self.proj_out(self.norm_out(h, temb)) + e[:, :1, :1] * 0.0
        return x


def random_transformer(joint):
    from diffusers import Flux2Transformer2DModel
    torch.manual_seed(1)
    tr = Flux2Transformer2DModel(patch_size=1, in_channels=8, num_layers=2, num_single_layers=4, attention_head_dim=16,
                                 num_attention_heads=2, joint_attention_dim=joint, timestep_guidance_channels=32,
                                 mlp_ratio=3.0, axes_dims_rope=(4, 4, 4, 4), rope_theta=2000, guidance_embeds=False).eval()
    for p in tr.parameters():
        if p.dim() > 1:
            nn.init.normal_(p, std=0.05)
    return tr


def export_transformer(tr, nt, a, work, singles_per_piece):
    """Writes klein-time, klein-dNN and klein-sNN; returns the meta and checks them when asked."""
    import klein_ncnn
    cfg = tr.config
    D, C, J, hd = tr.inner_dim, cfg.in_channels, cfg.joint_attention_dim, cfg.attention_head_dim
    fp16 = not a.random

    def stem(name):
        return os.path.join(a.out, name)

    def q(name):
        if a.quantizer:
            quantize(stem(name), a.quantizer, a.bits, a.block)
        print(f'exported {name}', flush=True)

    def f32(m):
        # An fp32 copy of just this piece: the model itself may be in bf16 (half the memory).
        return copy.deepcopy(m).float().eval()

    # Two image-token counts make the sequence length dynamic; the check runs a third.
    n1, n2 = (16, 24) if a.random else (64, 96)

    def rot(n):
        return torch.randn(1, nt + n, hd), torch.randn(1, nt + n, hd)

    pnnx_export(f32(KleinTime(tr)), (torch.tensor([0.5]),), None, stem('klein-time'), work, fp16=fp16)
    q('klein-time')
    doubles, singles = [], []
    for i in range(len(tr.transformer_blocks)):
        m = f32(KleinDouble(tr, i, nt))

        def sample(n):
            head = (torch.randn(1, nt, J), torch.randn(1, n, C)) if i == 0 else (torch.randn(1, nt + n, D),)
            return (*head, torch.randn(1, 6 * D), torch.randn(1, 6 * D), *rot(n))
        doubles.append(f'klein-d{i:02d}')
        pnnx_export(m, sample(n1), sample(n2), stem(doubles[-1]), work, fp16=fp16)
        q(doubles[-1])
    total = len(tr.single_transformer_blocks)
    for first in range(0, total, singles_per_piece):
        last = min(total, first + singles_per_piece)
        final = last == total
        m = f32(KleinSingle(tr, first, last, nt, final))

        def sample(n):
            return (torch.randn(1, nt + n, D), torch.randn(1, 3 * D), *rot(n)) + ((torch.randn(1, D),) if final else ())
        singles.append(f'klein-s{len(singles):02d}')
        pnnx_export(m, sample(n1), sample(n2), stem(singles[-1]), work, fp16=fp16)
        q(singles[-1])
    meta = {'inner_dim': D, 'latent_channels': C, 'joint_dim': J, 'rope_axes': list(cfg.axes_dims_rope),
            'rope_theta_image': float(cfg.rope_theta), 'double_pieces': doubles, 'single_pieces': singles}
    with open(os.path.join(a.out, 'meta-image.json'), 'w') as f:
        json.dump(meta, f, indent=1)

    if a.check:
        # One model call against the pieces: text, a 6×5 image and a 4×6 reference (a length that
        # was not traced), as the pipeline joins them.
        torch.manual_seed(3)
        text = torch.randn(1, nt, J)
        img, ref = torch.randn(1, 30, C), torch.randn(1, 24, C)
        ids = klein_ncnn.ids_text(nt), np.concatenate((klein_ncnn.ids_image(6, 5, 0), klein_ncnn.ids_image(4, 6, 10)))
        t = 0.7
        dt = next(tr.parameters()).dtype
        want = tr(hidden_states=torch.cat((img, ref), dim=1).to(dt), encoder_hidden_states=text.to(dt),
                  timestep=torch.tensor([t], dtype=dt), img_ids=torch.from_numpy(ids[1])[None],
                  txt_ids=torch.from_numpy(ids[0])[None], guidance=None, return_dict=False)[0][0].float().numpy()
        got = klein_ncnn.Pieces(a.out).transformer(text[0].numpy(), torch.cat((img, ref), dim=1)[0].numpy(), ids[1], t)
        cos = float((got * want).sum() / (np.linalg.norm(got) * np.linalg.norm(want) + 1e-12))
        err = float(np.abs(got - want).max())
        print(f'transformer check ({nt} text + {got.shape[0]} image tokens): cosine {cos:.5f}, max difference {err:.4f}, '
              f'output range {np.abs(want).max():.2f}')
        if cos < (0.999 if a.random else 0.97):
            raise SystemExit('the transformer pieces do not match PyTorch')
    return meta


# ---------------------------------------------------------------- VAE (AutoencoderKLFlux2)

class VaeEncode(nn.Module):
    """RGB in [-1, 1] [1, 3, H, W] -> the latent mean [1, C, H/8, W/8] (the pipeline's "argmax")."""

    def __init__(self, vae):
        super().__init__()
        self.encoder, self.quant, self.c = vae.encoder, vae.quant_conv, vae.config.latent_channels

    def forward(self, x):
        z = self.encoder(x)
        if self.quant is not None:
            z = self.quant(z)
        return torch.tensor_split(z, (self.c,), dim=1)[0] + torch.tensor_split(z, (self.c,), dim=1)[1][:, :1, :1, :1] * 0.0


class VaeDecode(nn.Module):
    """Latent [1, C, h, w] -> RGB in [-1, 1] [1, 3, 8h, 8w]."""

    def __init__(self, vae):
        super().__init__()
        self.decoder, self.post = vae.decoder, vae.post_quant_conv

    def forward(self, z):
        return self.decoder(z if self.post is None else self.post(z))


def random_vae():
    from diffusers import AutoencoderKLFlux2
    torch.manual_seed(2)
    return AutoencoderKLFlux2(block_out_channels=(8, 16), down_block_types=('DownEncoderBlock2D',) * 2,
                              up_block_types=('UpDecoderBlock2D',) * 2, layers_per_block=1, latent_channels=2,
                              norm_num_groups=4).eval()


def export_vae(vae, a, work):
    """klein-enc and klein-dec, traced at two small sizes and checked at a third; meta-vae.json has
    the batch-norm statistics the latents are normalised with after the 2x2 patchify."""
    import klein_ncnn
    vae = vae.float()
    cfg = vae.config
    c = cfg.latent_channels
    f = 2 ** (len(cfg.block_out_channels) - 1)
    s1, s2, s3 = (2, 3, 4) if a.random else (8, 12, 16)  # latent sides: traced, traced, checked
    pnnx_export(VaeEncode(vae), (torch.randn(1, 3, s1 * f, s1 * f),), (torch.randn(1, 3, s2 * f, s3 * f),),
                os.path.join(a.out, 'klein-enc'), work, fp16=not a.random)
    print('exported klein-enc', flush=True)
    pnnx_export(VaeDecode(vae), (torch.randn(1, c, s1, s1),), (torch.randn(1, c, s2, s3),),
                os.path.join(a.out, 'klein-dec'), work, fp16=not a.random)
    print('exported klein-dec', flush=True)
    std = torch.sqrt(vae.bn.running_var + cfg.batch_norm_eps)
    meta = {'vae_channels': c, 'vae_scale': f, 'bn_mean': vae.bn.running_mean.tolist(), 'bn_std': std.tolist()}
    with open(os.path.join(a.out, 'meta-vae.json'), 'w') as fh:
        json.dump(meta, fh)
    if a.check:
        torch.manual_seed(5)
        x = torch.rand(1, 3, s3 * f, s2 * f) * 2 - 1
        z = VaeEncode(vae)(x)
        y = VaeDecode(vae)(z)
        p = klein_ncnn.Pieces(a.out)
        gz = p.run(p.net('klein-enc'), [x[0].numpy()])
        gy = p.run(p.net('klein-dec'), [z[0].numpy()])
        ez = float(np.abs(gz - z[0].numpy()).max() / (np.abs(z[0].numpy()).max() + 1e-9))
        ey = float(np.abs(gy - y[0].numpy()).max())
        print(f'vae check at {s3 * f}x{s2 * f}: encoder relative difference {ez:.4f}, decoder max difference {ey:.4f} (pixels in [-1, 1])')
        if gz.shape != z[0].shape or gy.shape != y[0].shape or ez > 0.05 or ey > 0.05:
            raise SystemExit('the VAE does not match PyTorch')
    return meta


def random_text():
    """A tiny random Qwen3 and a real byte-level BPE tokenizer with Qwen's chat template."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(['a red fox sitting in fresh snow in a forest', 'user assistant think dress the woman in image'] * 4,
                           trainers.BpeTrainer(vocab_size=400, special_tokens=['<|endoftext|>', '<|im_start|>', '<|im_end|>', '<think>', '</think>'],
                                               initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token='<|endoftext|>', padding_side='right')
    tok.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                         "{% if add_generation_prompt %}<|im_start|>assistant\n{% if enable_thinking is defined and not enable_thinking %}"
                         "<think>\n\n</think>\n\n{% endif %}{% endif %}")
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=28,  # one more than used: HF applies the final norm to the last state
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
    model = Qwen3Model(cfg).eval()
    for p in model.parameters():  # livelier than the default init, still tame over 27 layers
        if p.dim() > 1:
            nn.init.normal_(p, std=0.08)
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='black-forest-labs/FLUX.2-klein-4B')
    ap.add_argument('--out', required=True)
    ap.add_argument('--stage', choices=['text', 'image', 'vae', 'all'], default='text')
    ap.add_argument('--singles-per-piece', type=int, default=4)
    ap.add_argument('--random', action='store_true', help='tiny random models: tests the export and the pieces')
    ap.add_argument('--tokens', type=int, default=512, help="prompt tokens, padded (the pipeline's max_sequence_length)")
    ap.add_argument('--layers-per-piece', type=int, default=3)
    ap.add_argument('--quantizer', help='path to ncnnllm2int')
    ap.add_argument('--bits', type=int, default=8)
    ap.add_argument('--block', type=int, default=64)
    ap.add_argument('--check', action='store_true', help='run the pieces against PyTorch')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    work = tempfile.mkdtemp(prefix='klein-')
    torch.set_grad_enabled(False)
    assert all(k % a.layers_per_piece == 0 for k in LAYERS_OUT), 'the wanted layers must end pieces'

    if a.random:
        a.tokens = min(a.tokens, 48)
    if a.stage in ('vae', 'all'):
        if a.random:
            vae = random_vae()
        else:
            from diffusers import AutoencoderKLFlux2
            vae = AutoencoderKLFlux2.from_pretrained(a.repo, subfolder='vae', torch_dtype=torch.float32).eval()
        export_vae(vae, a, work)
        if a.stage == 'vae':
            return
    if a.stage in ('image', 'all'):
        if a.random:
            tr = random_transformer(joint=3 * 64)
        else:
            from diffusers import Flux2Transformer2DModel
            tr = Flux2Transformer2DModel.from_pretrained(a.repo, subfolder='transformer', torch_dtype=torch.bfloat16).eval()
        print(f'transformer: {len(tr.transformer_blocks)} double + {len(tr.single_transformer_blocks)} single blocks, '
              f'width {tr.inner_dim}, heads {tr.config.num_attention_heads} x {tr.config.attention_head_dim}')
        export_transformer(tr, a.tokens, a, work, a.singles_per_piece)
        if a.stage == 'image':
            return

    if a.random:
        model, tok = random_text()
        random_text_models = (model, tok)
    else:
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.repo, subfolder='tokenizer')
        model = AutoModel.from_pretrained(a.repo, subfolder='text_encoder', torch_dtype=torch.bfloat16).eval()
    cfg = model.config
    print(f'text encoder: {type(model).__name__}, {len(model.layers)} layers, hidden {cfg.hidden_size}, '
          f'heads {cfg.num_attention_heads}/{cfg.num_key_value_heads} x {cfg.head_dim}, vocab {cfg.vocab_size}, '
          f'padding {tok.padding_side}, pad id {tok.pad_token_id}')
    assert tok.padding_side == 'right', 'the pieces assume right padding'
    template = chat(tok, 'PROMPT')
    head, tail = template.split('PROMPT')
    print(f'chat template: {head!r} + prompt + {tail!r}')

    # The check's reference first, while the whole model is still here.
    prompt = 'a red fox sitting in fresh snow in a forest, sharp focus on the eyes'
    want, ids, real = reference(model, tok, prompt, a.tokens) if a.check else (None, None, None)

    emb_scale, emb_q = write_embed(model.embed_tokens.weight, os.path.join(a.out, 'qwen-embed.bin'))
    n, H = a.tokens, cfg.hidden_size
    last = max(LAYERS_OUT)
    pieces = []
    # A second reference for the check: the same layers in fp32 in PyTorch, fed the int8 embedding
    # rows. Against it the pieces show only what quantizing the weights costs; the model itself
    # runs in bf16, whose rounding shows up on the padding tokens.
    x32, kept32 = None, []
    if a.check:
        x32 = torch.from_numpy(emb_q[ids].astype(np.float32) * emb_scale[ids, None]).unsqueeze(0)
        cos32, sin32 = rope(cfg, range(n))
        mask32 = prompt_mask(n, real).unsqueeze(0)
    for first in range(0, last, a.layers_per_piece):
        m = QwenLayers(model, first, first + a.layers_per_piece)
        if a.check:
            x32 = m(x32, cos32.unsqueeze(0), sin32.unsqueeze(0), mask32)
            if first + a.layers_per_piece in LAYERS_OUT:
                kept32.append(x32[0].numpy().copy())
        stem = os.path.join(a.out, f'qwen-{len(pieces):02d}')
        cos, sin = rope(cfg, range(n))
        # --random: fp32 weights, so the check tests the graph, not rounding (as in sana_export.py).
        pnnx_export(m, (torch.zeros(1, n, H), cos.unsqueeze(0), sin.unsqueeze(0), prompt_mask(n, n // 2).unsqueeze(0)),
                    None, stem, work, fp16=not a.random)
        if a.quantizer:
            quantize(stem, a.quantizer, a.bits, a.block)
        pieces.append(os.path.basename(stem))
        del m
        print(f'exported {pieces[-1]} (layers {first}-{first + a.layers_per_piece - 1})', flush=True)

    meta = {
        'version': 1, 'text_tokens': n, 'hidden': H, 'head_dim': cfg.head_dim,
        'rope_theta': float(getattr(cfg, 'rope_theta', None) or cfg.rope_parameters['rope_theta']),
        'mask_neg': MASK_NEG, 'text_pieces': pieces,
        'text_outputs': [k // a.layers_per_piece - 1 for k in LAYERS_OUT],  # pieces whose output is kept
        'chat_head': head, 'chat_tail': tail, 'pad_id': tok.pad_token_id,
    }
    with open(os.path.join(a.out, 'meta-text.json'), 'w') as f:
        json.dump(meta, f, indent=1)

    if a.check:
        import gc
        import klein_ncnn
        if not a.random:
            del model  # 8 GB the check does not need
            gc.collect()
        got = klein_ncnn.Pieces(a.out).text(ids, real)
        want32 = np.concatenate(kept32, axis=1)

        def cosines(x, y):
            return [float(np.dot(x[i], y[i]) / (np.linalg.norm(x[i]) * np.linalg.norm(y[i]) + 1e-12)) for i in range(n)]

        def report(name, c):
            print(f'{name}: real tokens min {min(c[:real]):.5f} mean {np.mean(c[:real]):.5f}; '
                  f'padding min {min(c[real:] or [1]):.5f} mean {np.mean(c[real:] or [1]):.5f}')

        print(f'text check, {real} real tokens of {n}')
        quant = cosines(got, want32)
        report('pieces against fp32 PyTorch (cost of quantizing)', quant)
        report('pieces against the model as loaded', cosines(got, want))
        report('fp32 PyTorch against the model as loaded', cosines(want32, want))
        norms = np.linalg.norm(want32.reshape(n, len(LAYERS_OUT), -1), axis=2)
        print('norms per kept layer, real tokens mean', norms[:real].mean(0).round(1), 'padding mean', norms[real:].mean(0).round(1))
        # The padding tokens are judged by their mean: they are small (a tenth of a real token's
        # norm in klein 4B) and unstable, bf16 against fp32 alone moves single ones to 0.76.
        if min(quant[:real]) < (0.999 if a.random else 0.98) or np.mean(quant[real:] or [1]) < 0.9:
            raise SystemExit('the text pieces do not match PyTorch')

    if a.random and a.check and a.stage == 'all':
        # The whole pipeline on the tiny models: words to picture, then an edit with a reference.
        import klein_ncnn
        from diffusers import FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline
        from PIL import Image
        model, tok = random_text_models
        sched = FlowMatchEulerDiscreteScheduler(base_image_seq_len=256, base_shift=0.5, max_image_seq_len=4096, max_shift=1.15,
                                                shift=3.0, time_shift_type='exponential', use_dynamic_shifting=True)
        pipe = Flux2KleinPipeline(scheduler=sched, vae=vae, text_encoder=model, tokenizer=tok, transformer=tr, is_distilled=True)
        pipe.set_progress_bar_config(disable=True)
        p = klein_ncnn.Pieces(a.out)
        emb, ids_, real_ = reference(model, tok, 'a red fox in fresh snow', a.tokens)
        text = p.text(ids_, real_)
        rng = np.random.default_rng(7)
        ref_img = Image.fromarray(rng.integers(0, 255, (64, 80, 3), dtype=np.uint8))
        ref = np.asarray(ref_img, np.float32).transpose(2, 0, 1) / 127.5 - 1
        for name, refs, pil in (('create', [], None), ('edit', [ref], [ref_img])):
            noise = torch.randn(1, tr.config.in_channels, 64 // 4, 64 // 4, generator=torch.Generator().manual_seed(11))
            want = pipe(image=pil, prompt_embeds=torch.from_numpy(emb)[None], latents=noise.clone(), height=64, width=64,
                        num_inference_steps=4, guidance_scale=1.0, output_type='pt').images[0].numpy() * 2 - 1
            got = np.clip(p.generate(text, noise[0].numpy(), refs), -1, 1)
            err = float(np.abs(got - want).max())
            print(f'pipeline check, {name}: max difference {err:.4f} (pixels in [-1, 1])')
            if got.shape != want.shape or err > 0.03:
                raise SystemExit('the pieces do not make the picture the pipeline makes')


if __name__ == '__main__':
    main()
